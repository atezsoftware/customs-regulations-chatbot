"""Guarded DEV repair for three previously reviewed multi-row replacements."""

import hashlib
import os
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select

from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import AmendmentBatch, AmendmentProposal, RegulatoryChunk
from onyx.db.regulatory_chunks import is_hierarchical_aggregate_chunk
from onyx.regulatory.amendments.draft_integrity import explicit_replacement_body
from onyx.regulatory.amendments.pipeline import _chunk_to_review_dict

BATCH_ID = 299
SOURCE_SHA256 = "30f7be2484b800594679c3e9f81eac71fb8f4fa18e7d193bcbd9fbe00be2c70b"
FILE_ID = "014fbc6a-2da2-4845-9df9-f4afc625587a"
EXPECTED = {
    440: {
        "index": 0,
        "kind": "clause_pair",
        "article": "3",
        "hash": "b4a941e0ea76d18a4b4a71c268b0610291e828186441edc3d690f1807e95688a",
        "ids": {
            "rc_6dbdeab7151644c0117dcbc6b2d6e71af913abfe",
            "rc_b0496615f66f861ff57646905f239eea85f2eca5",
        },
    },
    435: {
        "index": 2,
        "kind": "article",
        "article": "4",
        "hash": "200780182c029fd1a99d3f495044728bf8dca94e473677845341cd64d12a6a83",
        "ids": {
            "rc_53ca26620c835ef632bc473c89afb541098c65b9",
            "rc_036a60d3142133ab6797477a9964e6ed6f2391a6",
            "rc_6eb5e0799a3e5c64b8a933cee63a84d2cbcbce7c",
            "rc_5118c55aa1d40d618acd8c43d5879dc62f9adcc8",
        },
    },
    436: {
        "index": 4,
        "kind": "article",
        "article": "6",
        "hash": "2fba54444b291fcc16bc1a5e7a304d757877db8959ee8e7cbe62614424743774",
        "ids": {
            "rc_836c6ecaf8cbc015c9b275f18f43774d9928710b",
            "rc_9099a31ac2161b16ec75c36ad57161eebb98ab62",
            "rc_92a622d50595962d792f60d7d4f4663b26cce687",
            "rc_39db3feea37b4a1100c6b243eca4498ab75f9477",
            "rc_52798951a209cb121edd75daec5e4644006894e9",
            "rc_baf86718a51692e421f6c4cf724e4803047e7152",
        },
    },
}


def main() -> None:
    from onyx.regulatory.amendments.annexes import config as annex_config

    if os.environ.get("POSTGRES_DB") != "customs-regulations-dev":
        raise RuntimeError("Scope repair is restricted to DEV")
    if annex_config.REGULATORY_ANNEX_ENVIRONMENT != "dev":
        raise RuntimeError("Scope repair requires DEV publication")
    SqlEngine.init_engine(pool_size=1, max_overflow=0)
    with get_session_with_current_tenant() as session:
        batch = session.scalar(
            select(AmendmentBatch)
            .where(AmendmentBatch.id == BATCH_ID)
            .with_for_update()
        )
        if (
            batch is None
            or batch.status != "analyzed"
            or batch.superseded_by_batch_id is not None
            or hashlib.sha256(batch.raw_text.encode()).hexdigest() != SOURCE_SHA256
        ):
            raise RuntimeError("Batch 299 authority changed")
        repaired = []
        for proposal_id, expected in EXPECTED.items():
            proposal = session.scalar(
                select(AmendmentProposal)
                .where(AmendmentProposal.id == proposal_id)
                .with_for_update()
            )
            if (
                proposal is None
                or proposal.batch_id != BATCH_ID
                or proposal.instruction_index != expected["index"]
                or proposal.status != "pending"
                or proposal.chunk_changes
                or proposal.old_chunk_id not in expected["ids"]
                or proposal.old_chunk_snapshot.get("replacement_scope") is not None
            ):
                raise RuntimeError(f"Proposal {proposal_id} review state changed")
            draft = dict(proposal.new_chunk_draft)
            if (
                str(draft.get("user_file_id")) != FILE_ID
                or (draft.get("metadata") or {}).get("article_no")
                != expected["article"]
                or hashlib.sha256(str(draft.get("text", "")).encode()).hexdigest()
                != expected["hash"]
            ):
                raise RuntimeError(f"Proposal {proposal_id} source text changed")
            if proposal_id == 436:
                if (
                    proposal.instruction_text
                    != "MADDE 4- Aynı Tebliğin 6 ncı maddesi yürürlükten kaldırılmıştır."
                    or draft["text"] != "**MADDE 6-** (Mülga)"
                ):
                    raise RuntimeError("Article 6 repeal authority changed")
            elif explicit_replacement_body(proposal.instruction_text) != draft["text"]:
                raise RuntimeError(f"Proposal {proposal_id} quoted body changed")
            rows = list(
                session.scalars(
                    select(RegulatoryChunk)
                    .where(
                        RegulatoryChunk.id.in_(expected["ids"]),
                        RegulatoryChunk.status == "active",
                    )
                    .order_by(RegulatoryChunk.position)
                    .with_for_update()
                )
            )
            if {row.id for row in rows} != expected["ids"] or any(
                str(row.user_file_id) != FILE_ID for row in rows
            ):
                raise RuntimeError(f"Proposal {proposal_id} source scope changed")
            scoped_rows = list(
                session.scalars(
                    select(RegulatoryChunk).where(
                        RegulatoryChunk.user_file_id == UUID(FILE_ID),
                        RegulatoryChunk.status == "active",
                        RegulatoryChunk.chunk_metadata["article_no"].astext
                        == expected["article"],
                    )
                )
            )
            scoped_rows = [
                row
                for row in scoped_rows
                if not is_hierarchical_aggregate_chunk(row)
                and (
                    expected["kind"] == "article"
                    or (
                        row.chunk_metadata.get("paragraph_no") == "1"
                        and row.chunk_metadata.get("clause_label") in {"a", "b"}
                    )
                )
            ]
            if {row.id for row in scoped_rows} != expected["ids"]:
                raise RuntimeError(f"Proposal {proposal_id} has extra active source")
            primary = next(row for row in rows if row.id == proposal.old_chunk_id)
            if (
                proposal.old_chunk_snapshot.get("text") != primary.text
                or proposal.old_chunk_snapshot.get("metadata") != primary.chunk_metadata
            ):
                raise RuntimeError(f"Proposal {proposal_id} frozen target changed")
            metadata = dict(primary.chunk_metadata)
            metadata.pop("clause_label", None)
            metadata.pop("subclause_label", None)
            if expected["kind"] == "article":
                metadata.pop("paragraph_no", None)
            heading_path = list(primary.heading_path[:-1])
            metadata["heading_path"] = heading_path
            scope = {
                "kind": expected["kind"],
                "article_no": expected["article"],
                "member_snapshots": [_chunk_to_review_dict(row) for row in rows],
                "result_chunk_type": (
                    "article" if expected["kind"] == "article" else "paragraph"
                ),
                "result_metadata": metadata,
                "result_heading_path": heading_path,
            }
            if expected["kind"] == "clause_pair":
                scope["paragraph_no"] = "1"
                scope["clause_labels"] = ["a", "b"]
            proposal.old_chunk_snapshot = {
                **proposal.old_chunk_snapshot,
                "replacement_scope": scope,
            }
            proposal.new_chunk_draft = {
                **draft,
                "position": primary.position,
                "chunk_type": scope["result_chunk_type"],
                "heading_path": heading_path,
                "metadata": metadata,
            }
            repaired.append(proposal_id)
        batch.analysis_log = [
            *batch.analysis_log,
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "reviewed_replacement_scope_repaired",
                "proposal_ids": repaired,
            },
        ][-2000:]
        if os.environ.get("AMENDMENT_SCOPE_DRY_RUN") == "1":
            session.rollback()
            print("CHECKED_SCOPES", repaired)
        else:
            session.commit()
            print("REPAIRED_SCOPES", repaired)


if __name__ == "__main__":
    main()
