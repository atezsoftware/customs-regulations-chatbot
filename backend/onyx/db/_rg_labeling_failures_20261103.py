"""Export exact failed source/context and retained ACK evidence without inference."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import _new_document_rg_operator as scope
import _rg_labeling_retry_20261102 as retry
from sqlalchemy import select, text

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    vertex_batch_submission_key,
)
from onyx.regulatory.labeling.intake_abstention_repair import build_abstention_repair
from onyx.regulatory.labeling.provider import TaxonomyDefinition
from onyx.regulatory.labeling.vertex_direct import _StoredResult
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id

RUN_ID = UUID("b6fd5ae5-5b51-4c8a-a332-910e86732b0a")


def retained_ack(
    session: Any, *, binding: str, submission: str, request: VertexBatchRequest
) -> dict[str, Any]:
    suffix = submission.removeprefix("regulatory-labeling-")
    scope.require(
        bool(scope.re.fullmatch(r"[0-9a-f]{64}", suffix)),
        "ACK submission identity changed",
    )
    file_id = (
        f"regulatory-labeling-direct/{binding}/{suffix}/{request.request_hash}.json"
    )
    raw = retry.retained_ack_bytes(session, file_id, binding)
    ack = _StoredResult.model_validate_json(raw)
    line = json.loads(ack.line)
    scope.require(
        ack.binding_sha256 == binding
        and ack.submission_key == submission
        and ack.request_hash == request.request_hash
        and VertexBatchRequest.model_validate(ack.request_payload) == request
        and hashlib.sha256(ack.line.encode()).hexdigest() == ack.line_sha256
        and line.get("key") == request.request_hash
        and line.get("request") == request.to_generate_content_request(),
        "Retained ACK correlation changed",
    )
    response = retry.complete_ack_context(ack.line, request.request_hash)
    return {
        "file_id": file_id,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "request_hash": request.request_hash,
        "request_echo_verified": True,
        "sole_candidate_finish_reason": "STOP",
        "line_sha256": ack.line_sha256,
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    scope.require(
        args.expected_image_sha == args.observed_image_sha == retry.RETRY_IMAGE_SHA,
        "Exact final deployed image is required",
    )
    scope.require(
        POSTGRES_DB == "customs-regulations-dev"
        and scope.config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
        and get_current_tenant_id() == "public",
        "Exact DEV tenant is required",
    )
    set_is_ee_based_on_env_variable()
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        session.execute(text("SET LOCAL statement_timeout = '20000'"))
        batch = session.get_one(scope.RegulatoryDocumentIntake, scope.BATCH)
        scope.require(
            batch.labeling_run_id == RUN_ID
            and batch.stage == "labeling"
            and batch.status == "preparing",
            "The frozen failed intake pointer changed",
        )
        scope.verify_scope(session, batch)
        scope.unpublished_epoch_scope(session)
        scope.require(
            retry.canonical_source_sha256(session) == retry.RETRY_SOURCE_SHA256,
            "The failed canonical source changed",
        )
        run = session.get_one(RegulatoryLabelingRun, RUN_ID)
        scope.require(
            run.status == "completed_with_errors"
            and run.stage == "finished"
            and (
                run.total_chunks,
                run.completed_chunks,
                run.failed_chunks,
                run.stale_chunks,
                run.derived_chunks,
                run.unresolved_derived_chunks,
            )
            == (651, 640, 11, 0, 46, 0),
            "The terminal failure set changed",
        )
        binding = LabelingProviderBinding.model_validate(run.provider_binding)
        taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
        failed = list(
            session.scalars(
                select(RegulatoryLabelingItem)
                .where(
                    RegulatoryLabelingItem.run_id == RUN_ID,
                    RegulatoryLabelingItem.status == "failed",
                )
                .order_by(RegulatoryLabelingItem.regulatory_chunk_id)
            )
        )
        all_items = {
            item.regulatory_chunk_id: item
            for item in session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == RUN_ID
                )
            )
        }
        shards = {
            shard.id: shard
            for shard in session.scalars(
                select(RegulatoryLabelingShard).where(
                    RegulatoryLabelingShard.run_id == RUN_ID
                )
            )
        }
        canonical = {
            row.id: row
            for row in capture_canonical_scope(session, UUID(scope.PDF_FILE))
        }
        scope.require(
            len(failed) == 11
            and all(str(item.user_file_id) == scope.PDF_FILE for item in failed),
            "The exact PDF failed set changed",
        )
        records = []
        for item in failed:
            retry.failed_request_is_current(
                session, run=run, item=item, frozen=all_items
            )
            scope.require(
                item.request_payload is not None and item.shard_id is not None,
                "A frozen failed request is missing",
            )
            assert item.request_payload is not None and item.shard_id is not None
            base = VertexBatchRequest.model_validate(item.request_payload)
            shard = shards[item.shard_id]
            base_ack = retained_ack(
                session,
                binding=binding.fingerprint,
                submission=shard.submission_key,
                request=base,
            )
            repair = build_abstention_repair(
                base_request=base,
                rejected_response=base_ack["response"],
                text=item.text_snapshot,
                chunk_id=item.regulatory_chunk_id,
                source_hash=item.canonical_text_sha256 or "",
                taxonomy=taxonomy,
                provider_binding=run.provider_binding,
            )
            submission = vertex_batch_submission_key(
                [repair.request],
                tenant_id="public",
                job_id=run.id,
                output_prefix=f"regulatory-labeling/{run.id}/abstention-repair/{shard.id}/{item.id}",
                submission_attempt=1,
            ).replace("regulatory-context-", "regulatory-labeling-", 1)
            repair_ack = retained_ack(
                session,
                binding=binding.fingerprint,
                submission=submission,
                request=repair.request,
            )
            records.append(
                {
                    "item_id": str(item.id),
                    "chunk_id": item.regulatory_chunk_id,
                    "error": item.error,
                    "canonical": canonical[item.regulatory_chunk_id].model_dump(
                        mode="json"
                    ),
                    "text_snapshot": item.text_snapshot,
                    "source_snapshot": item.source_snapshot,
                    "context_snapshot": item.context_snapshot,
                    "context_sha256": item.context_sha256,
                    "canonical_text_sha256": item.canonical_text_sha256,
                    "base_request_hash": item.request_hash,
                    "base_request_payload": item.request_payload,
                    "base_request_payload_sha256": publication_digest(
                        item.request_payload
                    ),
                    "base_ack": base_ack,
                    "repair_request_hash": repair.request.request_hash,
                    "repair_request_payload": repair.request.model_dump(
                        mode="json", exclude_computed_fields=True
                    ),
                    "repair_ack": repair_ack,
                }
            )
        report = {
            "operator_action": "failed_labeling_sources_exported_read_only",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(run.id),
            "expected_image_sha": args.expected_image_sha,
            "canonical_source_sha256": retry.RETRY_SOURCE_SHA256,
            "provider_binding_fingerprint": binding.fingerprint,
            "model": run.model,
            "failed_count": len(records),
            "provider_calls": 0,
            "records": records,
        }
        session.rollback()
    output = args.output_path
    scope.require(
        output.parent.is_dir()
        and output.parent.name.startswith("rg-labeling-failures."),
        "A private diagnostic output directory is required",
    )
    output.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"}
            | {
                "output": str(output),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "output_bytes": output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
