"""One-time, guarded DEV replay of the existing reviewed amendment proposals."""

import hashlib
import os
import sys
from uuid import UUID

from onyx.background.celery.tasks.regulatory_amendments.tasks import (
    enqueue_amendment_proposal_approval,
)
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.regulatory_amendments import (
    get_batch,
    get_proposal,
    queue_amendment_proposal_approval,
    reset_amendment_proposal_approval,
)
from shared_configs.contextvars import get_current_tenant_id

EXPECTED: dict[int, tuple[int, str, int]] = {
    440: (299, "b4a941e0ea76d18a4b4a71c268b0610291e828186441edc3d690f1807e95688a", 0),
    435: (299, "200780182c029fd1a99d3f495044728bf8dca94e473677845341cd64d12a6a83", 0),
    436: (299, "2fba54444b291fcc16bc1a5e7a304d757877db8959ee8e7cbe62614424743774", 0),
    437: (299, "862d095116b4fc47bad44f53fa35375746379468f73dc6e9e9b5084b249018e7", 0),
    438: (299, "b731ac863475cfc4405031945b0fbf129bfb5327517e7e8ae311a0ee0202c0ce", 0),
    441: (299, "23dfb8e916d111ea8ee6e4f5c8df11a3d8fb2e498039eb70633b51ecb30fb0fe", 0),
    439: (299, "758c3b4bdf345cec553958d194f302c566f7d559e2ef59e406c2abb986a34a48", 0),
    443: (299, "0cec6f3ed0cf25fdeed344032c924d7e4b842faa03f3465824d8d9cb2164b7ba", 0),
    442: (299, "1fe5608e7be48a77328be8bc26b79f1d439592d6b7b18ae47137bf02e57c82b4", 0),
    481: (299, "aee0fe3b82d815e238190a6f6ebc64be66cbfe244d9caf741c3a9569c1a2601c", 0),
    482: (306, "4a7c7ac54d7c2f2e98e3fc1098b607ea96b96bdc5826b8f8815e9d5c723aa444", 0),
    462: (309, "dcb40be9e4d2009764a65a9cdd6f44dc471562dc9ff2138cde056d81f1459be9", 0),
    465: (309, "1f7454149079474b14e3eebb024f3cd70cc08f1fb4734f7394e149d68d019242", 0),
    470: (310, "3fc4e3e9d68f3b9be74400646ef9a69b76a658d2eaf905b4dcb3192b97c08fec", 4),
}


def main(proposal_id: int) -> None:
    from onyx.regulatory.amendments.annexes import config as annex_config

    if os.environ.get("POSTGRES_DB") != "customs-regulations-dev":
        raise RuntimeError("This replay is restricted to the DEV database")
    if annex_config.REGULATORY_ANNEX_ENVIRONMENT != "dev":
        raise RuntimeError("This replay is restricted to DEV publication")
    SqlEngine.init_engine(pool_size=1, max_overflow=0)
    expected_batch, expected_hash, expected_changes = EXPECTED[proposal_id]
    with get_session_with_current_tenant() as session:
        proposal = get_proposal(session, proposal_id)
        if proposal is None or proposal.batch_id != expected_batch:
            raise RuntimeError("Proposal identity changed")
        batch = get_batch(session, expected_batch)
        if batch is None or batch.status != "analyzed":
            raise RuntimeError("Batch is not analyzed")
        if (
            hashlib.sha256(
                str(proposal.new_chunk_draft.get("text", "")).encode()
            ).hexdigest()
            != expected_hash
        ):
            raise RuntimeError("Reviewed proposal text changed")
        if len(proposal.chunk_changes or []) != expected_changes:
            raise RuntimeError("Reviewed change count changed")
        if proposal_id in {440, 435, 436}:
            scope = proposal.old_chunk_snapshot.get("replacement_scope") or {}
            expected_identity = {
                440: (0, "3", "clause_pair", "paragraph", 2),
                435: (2, "4", "article", "article", 4),
                436: (4, "6", "article", "article", 6),
            }[proposal_id]
            instruction_index, article_no, kind, chunk_type, member_count = (
                expected_identity
            )
            draft = proposal.new_chunk_draft
            if (
                proposal.instruction_index != instruction_index
                or str(draft.get("user_file_id"))
                != "014fbc6a-2da2-4845-9df9-f4afc625587a"
                or scope.get("kind") != kind
                or scope.get("article_no") != article_no
                or scope.get("result_chunk_type") != chunk_type
                or len(scope.get("member_snapshots") or []) != member_count
                or proposal.old_chunk_id
                not in {item.get("id") for item in scope["member_snapshots"]}
                or draft.get("chunk_type") != chunk_type
                or (draft.get("metadata") or {}).get("article_no") != article_no
                or draft.get("effective_start_date") != "2026-10-01"
            ):
                raise RuntimeError("Reviewed replacement scope changed")
        if proposal_id in {437, 438}:
            from onyx.regulatory.amendments.draft_integrity import explicit_added_body

            draft = proposal.new_chunk_draft
            expected_index, article_no = {437: (3, "4/A"), 438: (6, "6/B")}[proposal_id]
            if (
                proposal.instruction_index != expected_index
                or proposal.old_chunk_id is not None
                or str(draft.get("user_file_id"))
                != "014fbc6a-2da2-4845-9df9-f4afc625587a"
                or (draft.get("metadata") or {}).get("article_no") != article_no
                or draft.get("effective_start_date") != "2026-10-01"
                or explicit_added_body(proposal.instruction_text) != draft.get("text")
            ):
                raise RuntimeError("Reviewed inserted article changed")
        if proposal_id == 481:
            draft = proposal.new_chunk_draft
            if (
                proposal.instruction_index != 5
                or proposal.old_chunk_id is not None
                or str(draft.get("user_file_id"))
                != "014fbc6a-2da2-4845-9df9-f4afc625587a"
                or (draft.get("metadata") or {}).get("article_no") != "6/A"
                or (draft.get("insertion_order") or {}).get("after_article_no") != "6"
                or proposal.instruction_text.split("“", 1)[1] != draft.get("text")
            ):
                raise RuntimeError("Reviewed 6/A insertion identity changed")
        if proposal_id == 482:
            draft = proposal.new_chunk_draft
            quoted = proposal.instruction_text.split('"', 2)[1]
            if (
                proposal.instruction_index != 0
                or proposal.old_chunk_id
                != "rc_cf2191636b0857c03551c430312de392f2820bf5"
                or str(draft.get("user_file_id"))
                != "c7bed4e2-77a2-4609-bb52-3fdffcd15cb1"
                or (draft.get("metadata") or {}).get("document_number") != "2018/13"
                or draft.get("text", "").split("\n", 1)[1] != quoted
            ):
                raise RuntimeError("Reviewed 2018/13 paragraph identity changed")
        if proposal_id == 443:
            draft = proposal.new_chunk_draft
            if (
                proposal.instruction_index != 1
                or proposal.old_chunk_id is not None
                or str(draft.get("user_file_id"))
                != "014fbc6a-2da2-4845-9df9-f4afc625587a"
                or (draft.get("metadata") or {}).get("article_no") != "3"
                or (draft.get("metadata") or {}).get("clause_label") != "e"
                or draft.get("effective_start_date") != "2026-10-01"
                or proposal.instruction_text.split("“", 1)[1].rsplit("”", 1)[0]
                != draft.get("text")
            ):
                raise RuntimeError("Reviewed Article 3 clause e identity changed")
        if proposal_id == 442:
            from onyx.regulatory.amendments.draft_integrity import explicit_added_body

            draft = proposal.new_chunk_draft
            if (
                proposal.instruction_index != 8
                or proposal.old_chunk_id is not None
                or str(draft.get("user_file_id"))
                != "014fbc6a-2da2-4845-9df9-f4afc625587a"
                or (draft.get("metadata") or {}).get("article_no") != "GEÇİCİ 2"
                or draft.get("effective_start_date") != "2026-10-01"
                or explicit_added_body(proposal.instruction_text) != draft.get("text")
            ):
                raise RuntimeError("Reviewed temporary Article 2 identity changed")
        if proposal_id == 439:
            draft = proposal.new_chunk_draft
            old_text = (proposal.old_chunk_snapshot or {}).get("text")
            if (
                proposal.instruction_index != 7
                or proposal.old_chunk_id
                != "rc_89c2cbd8f02fda473189ef601fc32bb7c49cc414"
                or str(draft.get("user_file_id"))
                != "014fbc6a-2da2-4845-9df9-f4afc625587a"
                or (draft.get("metadata") or {}).get("article_no") != "GEÇİCİ 1"
                or draft.get("effective_start_date") != "2026-08-01"
                or not isinstance(old_text, str)
                or old_text.count("31/7/2026") != 1
                or old_text.replace("31/7/2026", "31/1/2027") != draft.get("text")
            ):
                raise RuntimeError("Reviewed temporary Article 1 date change altered")
        if proposal_id == 441:
            from onyx.regulatory.amendments.draft_integrity import explicit_added_body

            draft = proposal.new_chunk_draft
            if (
                proposal.instruction_index != 9
                or proposal.old_chunk_id is not None
                or str(draft.get("user_file_id"))
                != "014fbc6a-2da2-4845-9df9-f4afc625587a"
                or (draft.get("metadata") or {}).get("article_no") != "GEÇİCİ 3"
                or draft.get("effective_start_date") != "2026-10-01"
                or explicit_added_body(proposal.instruction_text) != draft.get("text")
            ):
                raise RuntimeError("Reviewed temporary Article 3 identity changed")
        if proposal.status != "pending":
            print(f"SKIP proposal={proposal_id} status={proposal.status}")
            return
        if proposal_id in {437, 438}:
            from onyx.db.regulatory_amendment_order import load_amendment_order
            from onyx.regulatory.amendments.insertion_order import plan_insertion

            dependency_id = 435 if proposal_id == 437 else 436
            dependency = get_proposal(session, dependency_id)
            before_id = (
                "rc_12f207233f855a9bb1cafb54558769aa3781d602"
                if proposal_id == 437
                else "rc_6ad57aba8059172544852055e09238f77f6fcf94"
            )
            prior = proposal.new_chunk_draft["insertion_order"]
            if (
                dependency is None
                or dependency.status != "approved"
                or prior.get("after_chunk_id")
                != (
                    "rc_5118c55aa1d40d618acd8c43d5879dc62f9adcc8"
                    if proposal_id == 437
                    else "rc_baf86718a51692e421f6c4cf724e4803047e7152"
                )
                or prior.get("before_chunk_id")
                != (
                    before_id
                    if proposal_id == 437
                    else "rc_89c2cbd8f02fda473189ef601fc32bb7c49cc414"
                )
            ):
                raise RuntimeError("Inserted article predecessor review changed")
            current = plan_insertion(
                load_amendment_order(
                    session, UUID(proposal.new_chunk_draft["user_file_id"])
                ),
                article_no="4/A" if proposal_id == 437 else "6/B",
                paragraph_no=None,
                clause_label=None,
            )
            expected_after = (
                dependency.applied_new_chunk_id
                if proposal_id == 437
                else "rc_22ac9f73b247206ee9fef6dedd6f377407929f9f"
            )
            if (
                current.after_chunk_id != expected_after
                or current.before_chunk_id != before_id
            ):
                raise RuntimeError("Inserted article boundary changed")
            proposal.new_chunk_draft = {
                **proposal.new_chunk_draft,
                "position": current.position,
                "insertion_order": current.model_dump(mode="json"),
            }
        if proposal_id == 441:
            from onyx.db.regulatory_amendment_order import load_amendment_order
            from onyx.regulatory.amendments.insertion_order import plan_insertion

            prior = proposal.new_chunk_draft["insertion_order"]
            first = get_proposal(session, 439)
            second = get_proposal(session, 442)
            if (
                first is None
                or second is None
                or first.status != "approved"
                or second.status != "approved"
                or prior.get("after_chunk_id")
                != "rc_89c2cbd8f02fda473189ef601fc32bb7c49cc414"
                or prior.get("before_chunk_id")
                != "rc_0f0090df74a75c1f9e35adbaf0df6d3593e6d6ca"
            ):
                raise RuntimeError("Temporary provision predecessor review changed")
            current = plan_insertion(
                load_amendment_order(
                    session, UUID(proposal.new_chunk_draft["user_file_id"])
                ),
                article_no="GEÇİCİ 3",
                paragraph_no=None,
                clause_label=None,
            )
            if (
                current.after_chunk_id != second.applied_new_chunk_id
                or current.before_chunk_id
                != "rc_0f0090df74a75c1f9e35adbaf0df6d3593e6d6ca"
            ):
                raise RuntimeError("Temporary provision insertion boundary changed")
            proposal.new_chunk_draft = {
                **proposal.new_chunk_draft,
                "position": current.position,
                "insertion_order": current.model_dump(mode="json"),
            }
        queue_amendment_proposal_approval(session, proposal, decided_by=None)
        session.commit()
    try:
        enqueue_amendment_proposal_approval(
            proposal_id=proposal_id, tenant_id=get_current_tenant_id()
        )
    except Exception:
        with get_session_with_current_tenant() as session:
            reset_amendment_proposal_approval(session, proposal_id=proposal_id)
        raise
    print(f"QUEUED proposal={proposal_id} batch={expected_batch}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {
        "435",
        "436",
        "437",
        "438",
        "439",
        "440",
        "441",
        "442",
        "443",
        "462",
        "465",
        "470",
        "481",
        "482",
    }:
        raise SystemExit("Specify exactly one approved DEV proposal ID")
    main(int(sys.argv[1]))
