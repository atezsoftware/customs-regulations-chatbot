"""Bounded fictional end-to-end release proof through the existing frontend."""

import hashlib
import json
import re
import signal
import time
from typing import Any
from uuid import UUID

import httpx

from onyx.db.regulatory_annex_acceptance import (
    CanaryRun,
    CreationIntent,
    RetainedArtifact,
    bootstrap_canary_file,
    canary_file_state,
    canary_index_names,
    cleanup_empty_canary_scope,
    issue_canary_token,
    read_canary_worker_failure,
    recover_creation_intents,
    reserve_canary,
    retained_canary_audit,
    revoke_canary_token,
    save_canary,
)
from onyx.regulatory.amendments.annexes.dev_acceptance import (
    load_fixtures,
    safe_failure_detail,
)

DEV_FRONTEND = "https://dev-customs-regulations.singlewindow.io"
DEV_ADMIN = "kubilay.payci@atez.com"
NOTICE = """Temsili Oran Yonetmeligi
MADDE 1 - Yonetmeligin EK-1 Oran Tablosu, devam sayfasi ve dipnotuyla birlikte,
ekli kaynakta yer alan iki sayfalik yeni tablo ile degistirilmistir.
MADDE 2 - Bu temsili degisiklik 10/09/2026 tarihinde yururluge girer.
Yalniz yazilim testi icindir; hukuki bir kaynak degildir."""


def bootstrap_original(run: CanaryRun) -> None:
    from onyx.configs.constants import DocumentSource
    from onyx.connectors.models import Document, TextSection
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure
    from onyx.regulatory.writer_publication import chunk_owned_file, republish_user_file
    from shared_configs.contextvars import get_current_tenant_id

    content = load_fixtures()["old.pdf"]
    bootstrap_canary_file(run, content)
    parsed = extract_annex_structure(content, "application/pdf")
    if parsed.page_count != 4 or parsed.issues != ["vision_model_unavailable"]:
        raise ValueError("fictional_native_baseline_incomplete")
    native = "\n".join(item.text for item in parsed.elements if item.text)
    document = Document(
        id=str(run.file_id),
        source=DocumentSource.FILE,
        semantic_identifier="Temsili Oran Yonetmeligi",
        sections=[TextSection(text=native, link="")],
        metadata={},
    )
    tenant = get_current_tenant_id()
    chunk_owned_file(run.file_id, tenant, [document])
    if republish_user_file(run.file_id, tenant, include_chunked=True) <= 0:
        raise ValueError("fictional_baseline_publication_empty")
    deadline = time.monotonic() + 30
    while not (snapshot := index_evidence(run)):
        if time.monotonic() >= deadline:
            raise TimeoutError("fictional_baseline_search_visibility_deadline")
        time.sleep(1)
    run.evidence["physical_indices"] = json.dumps(
        {item["index"]: item["index_uuid"] for item in snapshot}, sort_keys=True
    )
    run.phase = "baseline_published"
    save_canary(run)


def request_json(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, "/api" + path, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def wait_package(
    client: httpx.Client, run: CanaryRun, deadline: float
) -> dict[str, Any]:
    while time.monotonic() < deadline:
        result = request_json(
            client,
            "GET",
            f"/regulatory/amendments/source-packages/{run.package_id}",
            params={"document_set_id": run.document_set_id},
        )
        if result["status"] == "ready":
            return result
        if result["status"] in {"failed", "blocked"}:
            run.evidence["source_package_status"] = result["status"]
            for issue in result.get("issues", []):
                detail = (
                    issue.get("failure_detail") if isinstance(issue, dict) else None
                )
                if isinstance(detail, str) and len(detail) <= 4000:
                    run.evidence["worker_failure"] = detail
                    break
            save_canary(run)
            raise ValueError("fictional_source_package_failed")
        time.sleep(1)
    raise TimeoutError("fictional_source_package_deadline")


def wait_review(
    client: httpx.Client, run: CanaryRun, deadline: float, *, approved: bool = False
) -> dict[str, Any]:
    while time.monotonic() < deadline:
        reviews = request_json(
            client, "GET", f"/regulatory/amendments/batches/{run.batch_id}/annex-groups"
        )
        if reviews:
            if len(reviews) != 1:
                raise ValueError("fictional_canary_requires_one_group")
            review = reviews[0]
            if review["status"] == ("approved" if approved else "pending"):
                return review
            if review["status"] in {"blocked", "failed", "rejected"}:
                raise ValueError("fictional_annex_review_not_ready")
        batches = request_json(
            client,
            "GET",
            "/regulatory/amendments/batches",
            params={"document_set_id": run.document_set_id},
        )
        batch = next((item for item in batches if item["id"] == run.batch_id), None)
        if batch is None or batch["status"] == "failed":
            receipt = read_canary_worker_failure(run)
            if receipt is not None:
                run.evidence["worker_failure"] = receipt[1]
                save_canary(run)
            raise ValueError("fictional_amendment_batch_failed")
        time.sleep(1)
    raise TimeoutError("fictional_annex_review_deadline")


def prepare_review(
    client: httpx.Client,
    run: CanaryRun,
    deadline: float,
    *,
    source: bytes | None = None,
) -> dict[str, Any]:
    content = source if source is not None else load_fixtures()["new.pdf"]
    if run.package_id is None:
        result = request_json(
            client,
            "POST",
            "/regulatory/amendments/source-packages/upload",
            data={
                "document_set_id": str(run.document_set_id),
                "idempotency_key": "annex-canary-" + str(run.run_id),
            },
            files={"file": ("fictional-update.pdf", content, "application/pdf")},
        )
        run.package_id = UUID(result["id"])
        save_canary(run)
    package = wait_package(client, run, deadline)
    for asset in package["assets"]:
        response = client.get(
            f"/api/regulatory/amendments/source-packages/{run.package_id}/assets/{asset['id']}",
            params={"document_set_id": run.document_set_id},
        )
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != asset["sha256"]:
            raise ValueError("fictional_immutable_source_hash_mismatch")
    run.evidence["source_assets"] = len(package["assets"])
    if run.batch_id is None:
        result = request_json(
            client,
            "POST",
            "/regulatory/amendments/analyze",
            json={
                "document_set_id": run.document_set_id,
                "source_package_id": str(run.package_id),
                "raw_text": NOTICE,
            },
        )
        run.batch_id = int(result["id"])
        save_canary(run)
    review = wait_review(client, run, deadline)
    record_vision_roles(run, review)
    run.review_id = UUID(review["id"])
    run.phase = "review_ready"
    run.evidence["review_sha256"] = review["review_sha256"]
    save_canary(run)
    return review


def record_vision_roles(run: CanaryRun, review: dict[str, Any]) -> None:
    roles: list[dict[str, str | int]] = []
    for side in ("old", "new"):
        extraction = review["review_payload"][side + "_extraction"]
        for position, element in enumerate(extraction["elements"]):
            role = element.get("table_role", "unknown")
            if element["extraction_method"] != "vision" or role == "unknown":
                continue
            if element["kind"] != "table_cell" or role not in {"column_header", "data"}:
                raise ValueError("canary_invalid_vision_table_role")
            receipt: dict[str, str | int] = {"side": side, "table_role": role}
            original_locator = element["locator"]
            original_position = position
            source_hash = extraction["source_sha256"]
            view_payload = extraction.get("evidence_view")
            if view_payload is not None:
                from onyx.regulatory.amendments.annexes.models import AnnexEvidenceView

                try:
                    view = AnnexEvidenceView.model_validate(view_payload)
                    mappings = [
                        item
                        for item in view.element_mappings
                        if item.view_position == position
                    ]
                    if len(mappings) != 1:
                        raise ValueError("mapping_not_unique")
                    mapping = mappings[0]
                    if not 0 <= mapping.parent_index < len(view.parents):
                        raise ValueError("mapping_parent_invalid")
                    parent = view.parents[mapping.parent_index]
                    if not 0 <= mapping.original_position < parent.element_count:
                        raise ValueError("mapping_position_invalid")
                except ValueError as exc:
                    raise ValueError("canary_vision_role_mapping_invalid") from exc
                source_hash = parent.sha256
                original_position = mapping.original_position
                original_locator = mapping.original_locator.model_dump(mode="json")
                receipt.update(
                    {
                        "source_file_id": parent.file_id,
                        "view_sha256": view.sha256,
                        "view_position": position,
                        "view_locator_sha256": hashlib.sha256(
                            json.dumps(element["locator"], sort_keys=True).encode()
                        ).hexdigest(),
                    }
                )
            locator_json = json.dumps(original_locator, sort_keys=True)
            receipt.update(
                {
                    "source_sha256": source_hash,
                    "original_position": original_position,
                    "original_locator": locator_json,
                    "original_locator_sha256": hashlib.sha256(
                        locator_json.encode()
                    ).hexdigest(),
                }
            )
            roles.append(receipt)
    if len(roles) > 100 or {item["table_role"] for item in roles} != {
        "column_header",
        "data",
    }:
        raise ValueError("canary_explicit_vision_header_and_data_required")
    run.vision_roles = roles
    save_canary(run)


def approve_review(
    client: httpx.Client, run: CanaryRun, review: dict[str, Any], deadline: float
) -> dict[str, Any]:
    draft = review["review_payload"]
    comparison = draft["comparison"]
    changes = comparison["changes"]
    if (
        len(changes) != 1
        or [item["text"] for item in changes[0]["old"]] != ["5%"]
        or [item["text"] for item in changes[0]["new"]] != ["7%"]
    ):
        raise ValueError("fictional_expected_single_rate_change_required")
    run.phase = "approval_requested"
    save_canary(run)
    request_json(
        client,
        "POST",
        f"/regulatory/amendments/batches/{run.batch_id}/annex-groups/{run.review_id}/approve",
        json={"expected_review_sha256": review["review_sha256"]},
    )
    approved = wait_review(client, run, deadline, approved=True)
    run.phase = "approved"
    run.evidence["publication_generation"] = approved["publication_generation"]
    for name, count in draft["publication"]["counts"].items():
        run.evidence[name] = int(count)
    save_canary(run)
    return approved


def cleanup_canary(run: CanaryRun) -> None:
    from onyx.db.regulatory_writer_publication import writer_file_exists
    from onyx.regulatory.writer_publication import (
        delete_owned_file,
        request_owned_file_deletion,
    )
    from shared_configs.contextvars import get_current_tenant_id

    resolved = recover_creation_intents(run)
    run.retained_objects = retained_canary_audit(run)
    run.phase = "cleanup_incomplete"
    run.evidence["cleanup_complete"] = False
    run.evidence.pop("cleanup_live_projections", None)
    save_canary(run)
    tenant = get_current_tenant_id()
    for identifier in [run.file_id, *run.markdown_file_ids]:
        if writer_file_exists(identifier, tenant):
            request_owned_file_deletion(identifier, tenant)
            delete_owned_file(identifier, tenant)
    if canary_file_state(run)["file_exists"]:
        raise ValueError("fictional_cleanup_file_still_present")
    deadline = time.monotonic() + 30
    snapshot = index_evidence(run)
    while any(not item["tombstone"] for item in snapshot):
        if time.monotonic() >= deadline:
            raise ValueError("fictional_cleanup_live_search_projection_remains")
        time.sleep(1)
        snapshot = index_evidence(run)
    cleanup_empty_canary_scope(run)
    run.retained_objects = retained_canary_audit(run) + [
        RetainedArtifact(
            kind="tombstone",
            id=item["id"],
            index_name=item["index"],
            index_uuid=item["index_uuid"],
        )
        for item in snapshot
    ]
    if len(run.retained_objects) > 512:
        raise ValueError("canary_retained_object_bound_exceeded")
    run.evidence["retained_tombstones"] = len(snapshot)
    save_canary(run)
    if not resolved:
        raise ValueError("canary_creation_unresolved")
    run.evidence["cleanup_live_projections"] = 0
    run.evidence["cleanup_complete"] = True
    run.phase = "cleaned"
    save_canary(run)


def index_evidence(run: CanaryRun) -> list[dict[str, Any]]:
    from onyx.document_index.elasticsearch.client import ElasticsearchClient

    saved = run.evidence.get("physical_indices")
    expected: dict[str, str] = json.loads(saved) if isinstance(saved, str) else {}
    names = list(expected) if expected else canary_index_names()
    output: list[dict[str, Any]] = []
    with ElasticsearchClient() as transport:
        client = transport.publication_client()
        for name in names:
            metadata = client.indices.get(index=name)
            if set(metadata) != {name}:
                raise ValueError("canary_requires_physical_index")
            if (
                expected
                and metadata[name]["settings"]["index"]["uuid"] != expected[name]
            ):
                raise ValueError("canary_physical_index_changed")
            result = client.search(
                index=name,
                size=1000,
                track_total_hits=True,
                query={
                    "terms": {
                        "document_id": [
                            str(run.file_id),
                            *[str(value) for value in run.markdown_file_ids],
                        ]
                    }
                },
            )
            if result["hits"]["total"]["value"] > 1000:
                raise ValueError("canary_projection_inventory_exceeds_bound")
            for hit in result["hits"]["hits"]:
                source = hit["_source"]
                vector = source.get("content_vector", [])
                output.append(
                    {
                        "index": name,
                        "index_uuid": metadata[name]["settings"]["index"]["uuid"],
                        "id": hit["_id"],
                        "ordinal": source["chunk_index"],
                        "tombstone": bool(source.get("publication_tombstone")),
                        "hidden": bool(source.get("hidden")),
                        "vector_dimension": len(vector),
                        "vector_sha256": hashlib.sha256(
                            json.dumps(vector).encode()
                        ).hexdigest(),
                        "source_sha256": hashlib.sha256(
                            json.dumps(source, sort_keys=True).encode()
                        ).hexdigest(),
                    }
                )
    return output


def create_canary_chat(client: httpx.Client, run: CanaryRun, *, purpose: str) -> UUID:
    if purpose not in {"dated 2026-09-09", "dated 2026-09-10", "markdown"}:
        raise ValueError("canary_chat_purpose_not_supported")
    marker = run.name + " / " + purpose
    if any(
        item.kind == "chat" and item.marker == marker for item in run.creation_intents
    ):
        raise ValueError("canary_chat_creation_already_intended")
    intent = CreationIntent(kind="chat", marker=marker)
    run.creation_intents.append(intent)
    save_canary(run)
    session = request_json(
        client, "POST", "/chat/create-chat-session", json={"description": marker}
    )
    chat_id = UUID(session["chat_session_id"])
    intent.artifact_id = chat_id
    run.chat_ids.append(chat_id)
    save_canary(run)
    return chat_id


def upload_canary_markdown(client: httpx.Client, run: CanaryRun) -> dict[str, Any]:
    marker = "ANNEXCANARY" + run.run_id.hex
    if any(item.kind == "markdown" for item in run.creation_intents):
        raise ValueError("canary_markdown_creation_already_intended")
    intent = CreationIntent(kind="markdown", marker=marker + ".md")
    run.creation_intents.append(intent)
    save_canary(run)
    result = request_json(
        client,
        "POST",
        f"/manage/admin/document-set/{run.document_set_id}/file/upload",
        files={
            "files": (
                intent.marker,
                "# Fictional canary\nVerification marker: " + marker,
                "text/markdown",
            )
        },
    )
    if result["rejected_files"] or len(result["user_files"]) != 1:
        raise ValueError("ordinary_markdown_upload_failed")
    file = result["user_files"][0]
    identifier = UUID(file["id"])
    intent.artifact_id = identifier
    run.markdown_file_ids.append(identifier)
    save_canary(run)
    return file


def chat_canary(client: httpx.Client, run: CanaryRun, *, as_of: str, rate: str) -> None:
    tools = request_json(client, "GET", "/tool")
    search = next(
        (tool for tool in tools if tool.get("display_name") == "Internal Search"), None
    )
    if search is None:
        raise ValueError("canary_search_tool_missing")
    chat_id = create_canary_chat(client, run, purpose="dated " + as_of)
    result = request_json(
        client,
        "POST",
        "/chat/send-chat-message",
        json={
            "chat_session_id": str(chat_id),
            "message": f"{as_of} tarihinde Temsili Oran Yonetmeligi EK-1 tablosunda1001.10 kodunun orani nedir? Kaynagi goster.",
            "internal_search_filters": {
                "document_set": [run.name],
                "as_of_date": as_of,
            },
            "forced_tool_id": search["id"],
            "stream": False,
        },
    )
    if (
        result.get("error_msg")
        or rate not in result["answer"]
        or not result["top_documents"]
        or not result["citation_info"]
    ):
        raise ValueError("canary_dated_chat_evidence_failed")
    if not any(
        str(run.file_id) in str(document.get("document_id", ""))
        for document in result["top_documents"]
    ):
        raise ValueError("canary_chat_did_not_retrieve_owned_file")
    run.evidence["chat_" + as_of] = True
    save_canary(run)


def require_markdown_chat_evidence(
    response: dict[str, Any], identifier: UUID, marker: str
) -> None:
    marker_pattern = r"\bANNEXCANARY[0-9a-f]{32}\b"
    owned_chunks: set[int] = set()
    body_markers: set[str] = set()
    for document in response.get("top_documents", []):
        body = "\n".join(
            value
            for key in ("content", "blurb")
            if isinstance(value := document.get(key), str)
        )
        found = set(re.findall(marker_pattern, body))
        body_markers.update(found)
        chunk = document.get("chunk_ind")
        if (
            document.get("document_id") == str(identifier)
            and found == {marker}
            and type(chunk) is int
            and chunk >= 0
        ):
            owned_chunks.add(chunk)
    cited_owned_chunk = any(
        citation.get("document_id") == str(identifier)
        and type(citation.get("chunk_ind")) is int
        and citation["chunk_ind"] in owned_chunks
        for citation in response.get("citation_info", [])
    )
    if (
        response.get("error_msg")
        or set(re.findall(marker_pattern, response.get("answer", ""))) != {marker}
        or body_markers != {marker}
        or not cited_owned_chunk
    ):
        raise ValueError("ordinary_markdown_indexed_chat_failed")


def markdown_canary(client: httpx.Client, run: CanaryRun, deadline: float) -> None:
    marker = "ANNEXCANARY" + run.run_id.hex
    file = upload_canary_markdown(client, run)
    identifier = UUID(file["id"])
    while time.monotonic() < deadline:
        files = request_json(
            client, "GET", f"/manage/admin/document-set/{run.document_set_id}/files"
        )
        current = next(item for item in files if item["id"] == str(identifier))
        if current["status"] == "CHUNKED":
            request_json(
                client,
                "POST",
                f"/manage/admin/document-set/{run.document_set_id}/files/{identifier}/index",
            )
        elif current["status"] == "COMPLETED":
            break
        elif current["status"] in {"FAILED", "CANCELED"}:
            raise ValueError("ordinary_markdown_index_failed")
        time.sleep(1)
    else:
        raise TimeoutError("ordinary_markdown_index_deadline")
    tools = request_json(client, "GET", "/tool")
    search = next(
        (tool for tool in tools if tool.get("display_name") == "Internal Search"), None
    )
    if search is None:
        raise ValueError("canary_search_tool_missing")
    chat_id = create_canary_chat(client, run, purpose="markdown")
    response = request_json(
        client,
        "POST",
        "/chat/send-chat-message",
        json={
            "chat_session_id": str(chat_id),
            "message": "Return the verification marker from the indexed fictional canary document. Cite the source.",
            "internal_search_filters": {"document_set": [run.name]},
            "forced_tool_id": search["id"],
            "stream": False,
        },
    )
    require_markdown_chat_evidence(response, identifier, marker)
    run.evidence["ordinary_markdown_upload_index_chat"] = True
    save_canary(run)


def run_canary(release_sha: str) -> dict[str, Any]:
    from onyx.regulatory.amendments.annexes import config

    if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
        raise ValueError("canary_requires_explicit_creation_activation")
    run = reserve_canary(release_sha, admin_email=DEV_ADMIN)
    if run.phase == "cleaned" and run.evidence.get("acceptance_passed") is True:
        return {"canary": run.model_dump(mode="json"), "reused_completed_run": True}
    if run.phase != "reserved":
        raise ValueError("unfinished_canary_requires_owned_recovery")

    def deadline_expired(_signum: int, _frame: object) -> None:
        raise TimeoutError("fixed_canary_wall_time_limit")

    previous_handler = signal.signal(signal.SIGALRM, deadline_expired)
    signal.alarm(720)
    token: str | None = None
    failed = False
    operation = "token"
    try:
        token = issue_canary_token(run)
        with httpx.Client(
            base_url=DEV_FRONTEND,
            headers={"Authorization": "Bearer " + token},
            timeout=90,
            follow_redirects=False,
        ) as client:
            operation = "capabilities"
            request_json(client, "GET", "/regulatory/amendments/capabilities")
            operation = "baseline"
            bootstrap_original(run)
            deadline = time.monotonic() + 600
            operation = "source_review"
            review = prepare_review(client, run, deadline)
            operation = "approval"
            approve_review(client, run, review, deadline)
            operation = "historical_chat"
            chat_canary(client, run, as_of="2026-09-09", rate="5%")
            operation = "current_chat"
            chat_canary(client, run, as_of="2026-09-10", rate="7%")
            operation = "markdown"
            markdown_canary(client, run, deadline)
    except Exception as exc:
        failed = True
        run.evidence["failure"] = safe_failure_detail(operation, exc)
    finally:
        signal.alarm(120)
        try:
            cleanup_canary(run)
        except Exception as exc:
            failed = True
            run.evidence["cleanup_failure"] = safe_failure_detail("cleanup", exc)
        finally:
            try:
                if token is not None:
                    with httpx.Client(
                        base_url=DEV_FRONTEND,
                        headers={"Authorization": "Bearer " + token},
                        timeout=30,
                        follow_redirects=False,
                    ) as client:
                        for chat_id in run.chat_ids:
                            request_json(
                                client, "DELETE", f"/chat/delete-chat-session/{chat_id}"
                            )
            except Exception as exc:
                failed = True
                run.evidence["chat_cleanup_failure"] = safe_failure_detail(
                    "chat_cleanup", exc
                )
            finally:
                try:
                    revoke_canary_token(run)
                except Exception as exc:
                    failed = True
                    run.evidence["token_cleanup_failure"] = safe_failure_detail(
                        "token_cleanup", exc
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_handler)
    if any(
        key in run.evidence
        for key in ("cleanup_failure", "chat_cleanup_failure", "token_cleanup_failure")
    ):
        run.phase = "cleanup_incomplete"
        run.evidence["cleanup_complete"] = False
    run.evidence["acceptance_passed"] = not failed
    save_canary(run)
    return {
        "status": "failed" if failed else "passed",
        "canary": run.model_dump(mode="json"),
        "retained": "private document-set, immutable source/review history, publication ownership and tombstones, expired PAT audit",
    }
