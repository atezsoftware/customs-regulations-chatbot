"""One-time, guarded DEV replay of the existing reviewed amendment proposals."""

import hashlib
import os
import sys

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
                or (draft.get("metadata") or {}).get("document_number")
                != "2018/13"
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
                or old_text.replace("31/7/2026", "31/1/2027")
                != draft.get("text")
            ):
                raise RuntimeError("Reviewed temporary Article 1 date change altered")
        if proposal.status != "pending":
            print(f"SKIP proposal={proposal_id} status={proposal.status}")
            return
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
    if len(sys.argv) != 2 or sys.argv[1] not in {"439", "442", "443", "462", "465", "470", "481", "482"}:
        raise SystemExit("Specify exactly one approved DEV proposal ID")
    main(int(sys.argv[1]))
