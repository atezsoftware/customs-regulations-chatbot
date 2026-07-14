"""
FastAPI server for FsExplorer web UI.

Provides a WebSocket endpoint for real-time workflow streaming
and serves the single-page HTML interface.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import (
    Depends,
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from .agent import (
    GEMINI_MAX_CONTEXT_TOKENS,
    FsExplorerAgent,
    LLMCallStats,
    clear_index_context,
    set_index_context,
    set_search_flags,
)
from .amendments import (
    AnalyzeAmendmentRequest,
    AnalyzeAmendmentResponse,
    ApproveProposalsRequest,
    ApproveProposalsResponse,
    DecideProposalRequest,
    ProposalFailure,
    ProposalRecord,
    analyze_amendment,
    flag_duplicate_targets_in_records,
)
from fs_explorer_shared.auth import internal_token_valid, require_internal_token
from fs_explorer_shared.embeddings import EmbeddingProvider
from .exploration_trace import ExplorationTrace, extract_cited_sources
from fs_explorer_shared.index_config import (
    corpus_root as resolve_corpus_root,
    resolve_database_url,
)
from .llm import get_llm_client
from .runs import RunRecord, get_run, new_run_id, register_run, remove_run
from .search import IndexedQueryEngine
from fs_explorer_shared.storage import PostgresStorage
from .workflow import (
    AskHumanEvent,
    ExplorationEndEvent,
    GoDeeperEvent,
    HumanAnswerEvent,
    InputEvent,
    ToolCallEvent,
    get_run_agent,
    new_workflow,
    resume_agent_run,
)

app = FastAPI(title="FsExplorer", description="AI-powered filesystem exploration")


class TaskRequest(BaseModel):
    """Request model for task submission."""

    task: str
    folder: str = "."
    use_index: bool = False
    database_url: str | None = None
    model: str | None = None
    temperature: float | None = None


class SearchRequest(BaseModel):
    """Request model for search queries."""

    corpus_folder: str
    query: str
    filters: str | None = None
    limit: int = 5
    database_url: str | None = None
    as_of_date: str | None = None


def _format_conversation_context(raw_context: Any) -> str:
    """Format short-lived frontend memory into prompt context."""
    if not isinstance(raw_context, list):
        return ""

    lines: list[str] = []
    for item in raw_context[-10:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        cleaned = " ".join(content.split())
        if not cleaned:
            continue
        if len(cleaned) > 1200:
            cleaned = f"{cleaned[:1200]}..."
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {cleaned}")

    if not lines:
        return ""

    return (
        "Previous conversation context from this browser session:\n"
        + "\n".join(lines)
        + "\n\nUse this context when it is relevant, but answer the current "
        "question using the selected folder and available tools."
    )


def _task_with_context(task: str, raw_context: Any) -> str:
    context = _format_conversation_context(raw_context)
    if not context:
        return task
    return f"{context}\n\nCurrent question:\n{task}"


def _status_for_tool(tool_name: str, tool_input: dict[str, Any]) -> dict[str, str]:
    """Return compact UI status copy for a tool call."""
    if tool_name == "semantic_search":
        query = tool_input.get("query")
        return {
            "label": "Searching indexed chunks",
            "detail": str(query) if query else "Running indexed retrieval",
        }
    if tool_name in {"parse_file", "preview_file", "get_document", "read"}:
        target = (
            tool_input.get("file_path")
            or tool_input.get("doc_id")
            or tool_input.get("directory")
            or "document"
        )
        return {
            "label": "Reading indexed chunks",
            "detail": _display_target(str(target)),
        }
    if tool_name == "scan_folder":
        return {
            "label": "Scanning indexed documents",
            "detail": _display_target(str(tool_input.get("directory", ""))),
        }
    if tool_name in {"grep", "glob"}:
        return {
            "label": "Searching indexed text",
            "detail": _display_target(str(tool_input.get("pattern", ""))),
        }
    return {"label": "Using indexed retrieval", "detail": ""}


def _display_target(value: str) -> str:
    """Keep user-facing status copy free from local paths and raw storage details."""
    if not value:
        return ""
    cleaned = value.replace("\\", "/")
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    cleaned = re.sub(r"^\d+-", "", cleaned)
    cleaned = cleaned.replace("_", " ")
    if cleaned.startswith("doc_"):
        return "indexed document"
    return cleaned[:120]


def _source_links(
    *,
    cited_sources: list[str],
    referenced_documents: list[str],
    root_directory: str,
) -> dict[str, str]:
    candidates: list[Path] = [Path(path).resolve() for path in referenced_documents]
    root = Path(root_directory).resolve()

    # If the final answer cites a document that was not in a direct tool call,
    # fall back to a quick filename search under the selected folder.
    known_names = {path.name for path in candidates}
    for source in cited_sources:
        source_name = Path(source).name
        if source_name in known_names:
            continue
        for path in root.rglob(source_name):
            if path.is_file():
                candidates.append(path.resolve())
                known_names.add(path.name)
                break

    links: dict[str, str] = {}
    for source in cited_sources:
        source_name = Path(source).name
        for path in candidates:
            if path.name == source or path.name == source_name or source in str(path):
                links[source] = f"/api/document?path={str(path)}"
                break
    return links


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI HTML file."""
    html_path = Path(__file__).parent / "ui.html"
    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8"), status_code=200
        )
    return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)


@app.get("/api/document")
async def open_document(path: str):
    """Serve a local source document for citation links."""
    document_path = Path(path).expanduser().resolve()
    if not document_path.exists() or not document_path.is_file():
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return FileResponse(str(document_path), filename=document_path.name)


@app.get("/api/folders")
async def list_folders(path: str = "."):
    """
    List folders in the given path.
    Returns list of folder names and current path info.
    """
    try:
        base_path = Path(path).resolve()
        if not base_path.exists():
            return JSONResponse({"error": "Path not found"}, status_code=404)
        if not base_path.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)

        # Get folders (non-hidden)
        folders = sorted(
            [
                f.name
                for f in base_path.iterdir()
                if f.is_dir() and not f.name.startswith(".")
            ]
        )

        # Get parent path (if not at root)
        parent = str(base_path.parent) if base_path != base_path.parent else None

        return {
            "current": str(base_path),
            "parent": parent,
            "folders": folders,
            "files_count": len([f for f in base_path.iterdir() if f.is_file()]),
        }
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/search", dependencies=[Depends(require_internal_token)])
async def search_index(request: SearchRequest):
    """Search an indexed corpus and return ranked hits."""
    try:
        corpus_root = resolve_corpus_root(request.corpus_folder)

        resolved_database_url = resolve_database_url(request.database_url)
        storage = PostgresStorage(
            resolved_database_url, read_only=True, initialize=False
        )
        corpus_id = storage.get_corpus_id(corpus_root)
        if corpus_id is None:
            storage.close()
            return JSONResponse(
                {"error": "No index found for this folder."}, status_code=404
            )

        embedding_provider: EmbeddingProvider | None = None
        if storage.has_embeddings(corpus_id=corpus_id):
            try:
                embedding_provider = EmbeddingProvider()
            except ValueError:
                pass

        engine = IndexedQueryEngine(storage, embedding_provider=embedding_provider)
        hits = engine.search(
            corpus_id=corpus_id,
            query=request.query,
            filters=request.filters,
            limit=request.limit,
            as_of_date=request.as_of_date,
        )
        storage.close()

        return {
            "corpus_folder": corpus_root,
            "query": request.query,
            "hits": [
                {
                    "doc_id": hit.doc_id,
                    "relative_path": hit.relative_path,
                    "absolute_path": hit.absolute_path,
                    "chunk_id": hit.chunk_id,
                    "position": hit.position,
                    "text": hit.text,
                    "chunk_type": hit.chunk_type,
                    "metadata": hit.metadata,
                    "semantic_score": hit.semantic_score,
                    "metadata_score": hit.metadata_score,
                    "score": hit.score,
                    "matched_by": hit.matched_by,
                }
                for hit in hits
            ],
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/index/document-chunks", dependencies=[Depends(require_internal_token)])
async def document_chunks(
    corpus_key: str, relative_path_prefix: str, database_url: str | None = None
):
    """Look up a document and its chunks by corpus + relative-path prefix.

    This is a pure Postgres read (`get_document_chunks_by_prefix`, owned by
    `fs_explorer_shared`) — duplicated here from `core-indexer` so it works
    wherever `core-api` is deployed, since `core-indexer` isn't wired into
    any deployed environment yet (see CLAUDE.md's "Deploy status" note).
    """
    try:
        corpus_root = resolve_corpus_root(corpus_key)
        resolved_database_url = resolve_database_url(database_url)
        storage = PostgresStorage(
            resolved_database_url, read_only=True, initialize=False
        )
        try:
            result = storage.get_document_chunks_by_prefix(
                corpus_root=corpus_root, relative_path_prefix=relative_path_prefix
            )
        finally:
            storage.close()

        if result is None:
            return {"document": None, "chunks": []}
        return result
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/amendments/analyze", dependencies=[Depends(require_internal_token)])
async def analyze_amendment_endpoint(request: AnalyzeAmendmentRequest):
    """Analyze pasted Resmi Gazete amendment text into reviewable proposals.

    Never writes to `core_chunks` — only creates a batch and pending
    proposals for an admin to review and individually approve/reject via
    the endpoints below.
    """
    storage: PostgresStorage | None = None
    try:
        corpus_root = resolve_corpus_root(request.corpus_folder)
        resolved_database_url = resolve_database_url(request.database_url)
        storage = PostgresStorage(resolved_database_url, initialize=False)

        corpus_id = storage.get_corpus_id(corpus_root)
        if corpus_id is None:
            return JSONResponse(
                {"error": "No index found for this folder."}, status_code=404
            )

        batch_id = storage.create_amendment_batch(
            corpus_id=corpus_id, raw_text=request.raw_text, created_by=None
        )

        try:
            embedding_provider: EmbeddingProvider | None = None
            if storage.has_embeddings(corpus_id=corpus_id):
                try:
                    embedding_provider = EmbeddingProvider()
                except ValueError:
                    pass

            result = await analyze_amendment(
                storage=storage,
                embedding_provider=embedding_provider,
                llm=get_llm_client(),
                corpus_id=corpus_id,
                raw_text=request.raw_text,
            )
        except Exception as exc:
            storage.update_amendment_batch(
                batch_id=batch_id, status="failed", error_message=str(exc)
            )
            raise

        storage.update_amendment_batch(
            batch_id=batch_id,
            status="analyzed",
            reference_date=result.reference_date,
        )
        proposal_ids = storage.create_amendment_proposals(
            batch_id=batch_id,
            proposals=[proposal.to_storage_dict() for proposal in result.proposals],
        )
        stored = [
            storage.get_amendment_proposal(proposal_id=proposal_id)
            for proposal_id in proposal_ids
        ]
        records = flag_duplicate_targets_in_records(
            [record for record in stored if record is not None]
        )

        return AnalyzeAmendmentResponse(
            batch_id=batch_id,
            reference_date=result.reference_date,
            proposals=[ProposalRecord(**record) for record in records],
            unmatched_instructions=[
                instruction.instruction_text
                for instruction in result.unmatched_instructions
            ],
        ).model_dump()
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if storage is not None:
            storage.close()


@app.get("/api/amendments/proposals", dependencies=[Depends(require_internal_token)])
async def list_amendment_proposals_endpoint(
    status: str | None = None,
    batch_id: str | None = None,
    corpus_folder: str | None = None,
    database_url: str | None = None,
):
    """List amendment proposals, optionally filtered by status/batch/corpus.

    Used both right after `analyze` and independently later, so proposals
    an admin didn't act on stay visible ("pending") until approved/deleted.
    """
    storage: PostgresStorage | None = None
    try:
        resolved_database_url = resolve_database_url(database_url)
        storage = PostgresStorage(
            resolved_database_url, read_only=True, initialize=False
        )
        corpus_id: str | None = None
        if corpus_folder:
            corpus_id = storage.get_corpus_id(resolve_corpus_root(corpus_folder))
            if corpus_id is None:
                return {"proposals": []}

        records = storage.list_amendment_proposals(
            status=status, batch_id=batch_id, corpus_id=corpus_id
        )
        records = flag_duplicate_targets_in_records(records)
        return {
            "proposals": [ProposalRecord(**record).model_dump() for record in records]
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if storage is not None:
            storage.close()


@app.get(
    "/api/amendments/batches/{batch_id}",
    dependencies=[Depends(require_internal_token)],
)
async def get_amendment_batch_endpoint(batch_id: str, database_url: str | None = None):
    """Fetch one amendment batch plus all of its proposals."""
    storage: PostgresStorage | None = None
    try:
        resolved_database_url = resolve_database_url(database_url)
        storage = PostgresStorage(
            resolved_database_url, read_only=True, initialize=False
        )
        batch = storage.get_amendment_batch(batch_id=batch_id)
        if batch is None:
            return JSONResponse({"error": "Batch not found."}, status_code=404)
        records = flag_duplicate_targets_in_records(
            storage.list_amendment_proposals(batch_id=batch_id, limit=1000)
        )
        return {
            "batch": batch,
            "proposals": [ProposalRecord(**record).model_dump() for record in records],
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if storage is not None:
            storage.close()


@app.post(
    "/api/amendments/proposals/approve",
    dependencies=[Depends(require_internal_token)],
)
async def approve_amendment_proposals_endpoint(request: ApproveProposalsRequest):
    """Approve selected proposals. Each is applied independently (its own
    DB transaction) so one conflict doesn't roll back the others — the
    caller gets back which succeeded and which failed, with why."""
    storage: PostgresStorage | None = None
    try:
        resolved_database_url = resolve_database_url(request.database_url)
        storage = PostgresStorage(resolved_database_url, initialize=False)

        embedding_provider: EmbeddingProvider | None = None
        try:
            embedding_provider = EmbeddingProvider()
        except ValueError:
            pass

        applied: list[ProposalRecord] = []
        failed: list[ProposalFailure] = []
        for proposal_id in request.proposal_ids:
            try:
                chunk = storage.approve_amendment_proposal(
                    proposal_id=proposal_id, decided_by=request.decided_by
                )
            except ValueError as exc:
                failed.append(ProposalFailure(proposal_id=proposal_id, reason=str(exc)))
                continue

            if embedding_provider is not None:
                try:
                    document = storage.get_document(doc_id=chunk["document_id"])
                    if document is not None:
                        embedding = embedding_provider.embed_texts([chunk["text"]])[0]
                        storage.store_chunk_embeddings(
                            corpus_id=document["corpus_id"],
                            chunk_embeddings=[(chunk["id"], embedding)],
                        )
                except Exception:
                    # Best-effort: an approved chunk without an embedding
                    # yet is still fully valid — it just won't surface via
                    # semantic search until a later embed pass fixes it up,
                    # the same tolerance the indexer already has for
                    # with_embeddings=false.
                    pass

            record = storage.get_amendment_proposal(proposal_id=proposal_id)
            if record is not None:
                applied.append(
                    ProposalRecord(**flag_duplicate_targets_in_records([record])[0])
                )

        return ApproveProposalsResponse(applied=applied, failed=failed).model_dump()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if storage is not None:
            storage.close()


@app.post(
    "/api/amendments/proposals/{proposal_id}/reject",
    dependencies=[Depends(require_internal_token)],
)
async def reject_amendment_proposal_endpoint(
    proposal_id: str, request: DecideProposalRequest
):
    storage: PostgresStorage | None = None
    try:
        resolved_database_url = resolve_database_url(request.database_url)
        storage = PostgresStorage(resolved_database_url, initialize=False)
        storage.reject_amendment_proposal(
            proposal_id=proposal_id, decided_by=request.decided_by
        )
        record = storage.get_amendment_proposal(proposal_id=proposal_id)
        if record is None:
            return JSONResponse({"error": "Proposal not found."}, status_code=404)
        return {
            "proposal": ProposalRecord(
                **flag_duplicate_targets_in_records([record])[0]
            ).model_dump()
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if storage is not None:
            storage.close()


@app.delete(
    "/api/amendments/proposals/{proposal_id}",
    dependencies=[Depends(require_internal_token)],
)
async def delete_amendment_proposal_endpoint(
    proposal_id: str, database_url: str | None = None
):
    storage: PostgresStorage | None = None
    try:
        resolved_database_url = resolve_database_url(database_url)
        storage = PostgresStorage(resolved_database_url, initialize=False)
        storage.delete_amendment_proposal(proposal_id=proposal_id)
        return {"deleted": True}
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if storage is not None:
            storage.close()


def _tool_call_ws_message(
    event: ToolCallEvent,
    *,
    step_number: int,
    trace: ExplorationTrace,
    index_storage: PostgresStorage | None,
) -> dict[str, Any]:
    resolved_document_path: str | None = None
    if event.tool_name == "get_document":
        doc_id = event.tool_input.get("doc_id")
        if index_storage is not None and isinstance(doc_id, str) and doc_id:
            document = index_storage.get_document(doc_id=doc_id)
            if document and not document["is_deleted"]:
                resolved_document_path = str(document["absolute_path"])
    trace.record_tool_call(
        step_number=step_number,
        tool_name=event.tool_name,
        tool_input=event.tool_input,
        resolved_document_path=resolved_document_path,
    )
    # status_label/status_detail below are sent as part of this single
    # tool_call event (not also as a standalone "status" event) so the
    # frontend doesn't end up with two adjacent research steps carrying the
    # identical label/detail text for what is really one action.
    tool_status = _status_for_tool(event.tool_name, event.tool_input)
    return {
        "type": "tool_call",
        "data": {
            "step": step_number,
            "tool_name": event.tool_name,
            "tool_input": event.tool_input,
            "reason": event.reason,
            "status_label": tool_status["label"],
            "status_detail": tool_status["detail"],
        },
    }


def _go_deeper_ws_message(
    event: GoDeeperEvent, *, step_number: int, trace: ExplorationTrace
) -> dict[str, Any]:
    trace.record_go_deeper(step_number=step_number, directory=event.directory)
    # No separate "status" event here either — the go_deeper event above
    # already carries the directory being inspected, which is what the
    # frontend displays for it.
    return {
        "type": "go_deeper",
        "data": {
            "step": step_number,
            "directory": event.directory,
            "reason": event.reason,
        },
    }


def _ask_human_ws_message(event: AskHumanEvent, *, step_number: int) -> dict[str, Any]:
    return {
        "type": "ask_human",
        "data": {
            "step": step_number,
            "question": event.question,
            "reason": event.reason,
        },
    }


def _register_if_resumable(
    *,
    run_id: str,
    agent: FsExplorerAgent | None,
    trace: ExplorationTrace | None,
    step_number: int,
    folder_path: Path,
    use_index: bool,
    enable_semantic: bool,
    enable_metadata: bool,
    index_folders: list[str],
    database_url: str | None,
    original_task: str,
) -> None:
    """Best-effort: keep an interrupted run resumable if it made real progress.

    Called from the `except` branch of both session functions below, right
    before the exception is re-raised for the outer handler's usual
    logging. Skips registration if the agent never got far enough to be
    worth resuming (no tool calls made yet) — nothing there for a "resume"
    to meaningfully continue.
    """
    if agent is None or trace is None or agent.step_count == 0:
        return
    register_run(
        RunRecord(
            run_id=run_id,
            agent=agent,
            trace=trace,
            step_number=step_number,
            folder=str(folder_path),
            use_index=use_index,
            enable_semantic=enable_semantic,
            enable_metadata=enable_metadata,
            index_folders=index_folders,
            database_url=database_url,
            original_task=original_task,
        )
    )


async def _finish_run(
    websocket: WebSocket,
    *,
    run_id: str,
    agent: FsExplorerAgent,
    trace: ExplorationTrace,
    step_number: int,
    folder_path: Path,
    use_index: bool,
    final_result: str,
    result_error: str | None,
    run_started_at: float,
    flush_llm_calls: Callable[[], Awaitable[None]],
) -> None:
    """Stream the final answer (if no error) and send the terminal `complete`
    event. Shared by a fresh run and a resumed one — by this point there is
    no meaningful difference between the two: both have an `agent` with a
    full chat history and a `trace` of everything gathered so far.
    """
    if not result_error:
        await websocket.send_json(
            {
                "type": "status",
                "data": {
                    "label": "Writing answer",
                    "detail": "Composing the final response",
                },
            }
        )
        await websocket.send_json({"type": "answer_start", "data": {}})
        streamed_parts: list[str] = []
        async for chunk in agent.stream_final_answer(fallback_answer=final_result):
            streamed_parts.append(chunk)
            await websocket.send_json({"type": "answer_delta", "data": {"text": chunk}})
        await flush_llm_calls()
        streamed_final = "".join(streamed_parts).strip()
        if streamed_final:
            final_result = streamed_final
        cited_sources = extract_cited_sources(final_result)
        referenced_documents = trace.sorted_documents()
        cited_source_links = (
            {}
            if use_index
            else _source_links(
                cited_sources=cited_sources,
                referenced_documents=referenced_documents,
                root_directory=str(folder_path),
            )
        )
        await websocket.send_json(
            {
                "type": "answer_done",
                "data": {
                    "final_result": final_result,
                    "cited_sources": cited_sources,
                    "cited_source_links": cited_source_links,
                },
            }
        )
    else:
        cited_sources = []
        referenced_documents = trace.sorted_documents()
        cited_source_links = {}

    usage = agent.token_usage
    _input_cost, _output_cost, total_cost = usage._calculate_cost()

    await websocket.send_json(
        {
            "type": "complete",
            "data": {
                "final_result": final_result,
                "error": result_error,
                "stats": {
                    "steps": step_number,
                    "api_calls": usage.api_calls,
                    "documents_scanned": usage.documents_scanned,
                    "documents_parsed": usage.documents_parsed,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "thinking_tokens": usage.thinking_tokens,
                    "total_tokens": usage.total_tokens,
                    "tool_result_chars": usage.tool_result_chars,
                    "estimated_cost": round(total_cost, 6),
                    "context_summaries": usage.context_summaries,
                    "model": getattr(agent._llm, "model", None),
                    "duration_ms": round((time.monotonic() - run_started_at) * 1000),
                    "context_usage_ratio": round(
                        usage.context_usage_ratio(GEMINI_MAX_CONTEXT_TOKENS), 4
                    ),
                },
                "trace": {
                    "step_path": trace.step_path,
                    "referenced_documents": referenced_documents,
                    "cited_sources": cited_sources,
                    "cited_source_links": cited_source_links,
                },
            },
        }
    )
    # Run reached a real terminal state (answer or error) — nothing further
    # for a later "resume" to do, so it must not still be resumable.
    remove_run(run_id)


async def _run_fresh_session(websocket: WebSocket, data: dict[str, Any]) -> None:
    """Start and drive a brand-new exploration run, end to end."""
    run_id = new_run_id()
    index_storage: PostgresStorage | None = None
    agent: FsExplorerAgent | None = None
    trace: ExplorationTrace | None = None
    step_number = 0
    folder_path = Path(".")
    use_index = False
    enable_semantic = False
    enable_metadata = False
    index_folders: list[str] = []
    resolved_database_url: str | None = None
    original_task = ""

    try:
        task = data.get("task", "")
        original_task = task
        folder = data.get("folder", ".")
        use_index = bool(data.get("use_index", False))
        raw_index_folders = data.get("index_folders")
        database_url = data.get("database_url")
        enable_semantic = bool(data.get("enable_semantic", False))
        enable_metadata = bool(data.get("enable_metadata", False))
        conversation_context = data.get("conversation_context")
        model = data.get("model")
        temperature = data.get("temperature")

        if not task:
            await websocket.send_json(
                {"type": "error", "data": {"message": "No task provided"}}
            )
            return

        # Validate folder only for raw filesystem mode. Indexed chat can use
        # virtual corpus keys backed entirely by PostgreSQL.
        folder_path = Path(folder).resolve()
        index_folder_values: list[str] = []
        if isinstance(raw_index_folders, list):
            index_folder_values = [
                str(item) for item in raw_index_folders if isinstance(item, str)
            ]
        elif isinstance(raw_index_folders, str):
            index_folder_values = [raw_index_folders]
        if not use_index and (not folder_path.exists() or not folder_path.is_dir()):
            await websocket.send_json(
                {"type": "error", "data": {"message": f"Invalid folder: {folder}"}}
            )
            return

        clear_index_context()
        if use_index:
            resolved_database_url = resolve_database_url(
                database_url if isinstance(database_url, str) else None
            )
            storage = PostgresStorage(resolved_database_url)
            candidate_index_folders = index_folder_values or [str(folder_path)]

            available_index_folders: list[str] = []
            for index_folder in candidate_index_folders:
                corpus_root = resolve_corpus_root(index_folder)
                if storage.get_corpus_id(corpus_root) is not None:
                    available_index_folders.append(corpus_root)

            if not available_index_folders:
                storage.close()
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {
                            "message": (
                                "No index found for the selected folder. "
                                "Run `explore index <folder>` first."
                            )
                        },
                    }
                )
                return
            index_storage = storage
            index_folders = available_index_folders
            set_index_context(available_index_folders, resolved_database_url)

        set_search_flags(
            enable_semantic=enable_semantic and use_index,
            enable_metadata=enable_metadata and use_index,
        )

        task = _task_with_context(str(task), conversation_context)
        trace = ExplorationTrace(root_directory=str(folder_path))

        resolved_model = model if isinstance(model, str) else None
        resolved_temperature = (
            float(temperature) if isinstance(temperature, (int, float)) else None
        )

        # Per-LLM-call token/timing observability. The hook runs inside the
        # workflow's internal worker tasks, not this handler's own task, so
        # it must not call websocket.send_json directly (concurrent writes
        # to the same socket from multiple tasks aren't safe) — it just
        # appends, and _flush_llm_calls() (called from this task only)
        # drains and sends them.
        run_started_at = time.monotonic()
        pending_llm_calls: list[LLMCallStats] = []

        async def _collect_llm_call(stats: LLMCallStats) -> None:
            pending_llm_calls.append(stats)

        async def _flush_llm_calls() -> None:
            while pending_llm_calls:
                stats = pending_llm_calls.pop(0)
                await websocket.send_json(
                    {
                        "type": "llm_call",
                        "data": {
                            "purpose": stats.purpose,
                            "model": stats.model,
                            "prompt_tokens": stats.prompt_tokens,
                            "completion_tokens": stats.completion_tokens,
                            "thinking_tokens": stats.thinking_tokens,
                            "duration_ms": round(stats.duration_ms),
                        },
                    }
                )

        # Send start event. The client should hold onto run_id and offer to
        # resume with it if the connection drops or the run errors out
        # before a `complete` event arrives.
        await websocket.send_json(
            {
                "type": "start",
                "data": {
                    "task": original_task,
                    "folder": str(folder_path),
                    "use_index": use_index,
                    "run_id": run_id,
                },
            }
        )
        await websocket.send_json(
            {
                "type": "status",
                "data": {
                    "label": "Thinking",
                    "detail": "Planning the first step",
                },
            }
        )

        # Run the workflow. Each connection gets its own workflow instance,
        # ResourceManager, and explicitly-constructed agent (model/
        # temperature/hook passed directly, not via module globals — see
        # new_workflow()'s docstring for why that matters under concurrent
        # requests). The agent is pre-registered into resource_manager
        # before any step runs, so get_run_agent() reliably returns it here
        # even while the run is still in progress (needed so a disconnect
        # mid-run can still register whatever the agent has gathered so far
        # — see _register_if_resumable() below).
        run_workflow, resource_manager = new_workflow(
            model=resolved_model,
            temperature=resolved_temperature,
            on_llm_call=_collect_llm_call,
        )
        agent = get_run_agent(resource_manager)
        handler = run_workflow.run(
            start_event=InputEvent(
                task=task,
                folder=str(folder_path),
                use_index=use_index,
                enable_semantic=enable_semantic and use_index,
                enable_metadata=enable_metadata and use_index,
            )
        )

        async for event in handler.stream_events():
            await _flush_llm_calls()
            if isinstance(event, ToolCallEvent):
                step_number += 1
                await websocket.send_json(
                    _tool_call_ws_message(
                        event,
                        step_number=step_number,
                        trace=trace,
                        index_storage=index_storage,
                    )
                )
            elif isinstance(event, GoDeeperEvent):
                step_number += 1
                await websocket.send_json(
                    _go_deeper_ws_message(event, step_number=step_number, trace=trace)
                )
            elif isinstance(event, AskHumanEvent):
                step_number += 1
                await websocket.send_json(
                    _ask_human_ws_message(event, step_number=step_number)
                )
                response_data = await websocket.receive_json()
                if response_data.get("type") == "human_response":
                    handler.ctx.send_event(
                        HumanAnswerEvent(response=response_data.get("response", ""))
                    )

        # Get final result
        result = await handler
        # Catch the LLM call that produced the terminal StopAction — it
        # isn't followed by another loop iteration to flush it.
        await _flush_llm_calls()

        await _finish_run(
            websocket,
            run_id=run_id,
            agent=agent,
            trace=trace,
            step_number=step_number,
            folder_path=folder_path,
            use_index=use_index,
            final_result=result.final_result or "",
            result_error=result.error,
            run_started_at=run_started_at,
            flush_llm_calls=_flush_llm_calls,
        )
    except Exception:
        _register_if_resumable(
            run_id=run_id,
            agent=agent,
            trace=trace,
            step_number=step_number,
            folder_path=folder_path,
            use_index=use_index,
            enable_semantic=enable_semantic,
            enable_metadata=enable_metadata,
            index_folders=index_folders,
            database_url=resolved_database_url,
            original_task=original_task,
        )
        raise
    finally:
        if index_storage is not None:
            index_storage.close()
        set_search_flags(enable_semantic=False, enable_metadata=False)
        clear_index_context()


async def _run_resume_session(websocket: WebSocket, run_id: str) -> None:
    """Continue a run that was previously interrupted (see `runs.py`).

    Reuses the same `FsExplorerAgent` the original connection was driving —
    its `_chat_history`/`_step_count` already hold everything gathered
    before the interruption — and drives it directly via
    `workflow.resume_agent_run()` instead of restarting through
    `InputEvent`/`start_exploration`.
    """
    index_storage: PostgresStorage | None = None
    record = get_run(run_id)
    if record is None:
        await websocket.send_json(
            {
                "type": "error",
                "data": {
                    "message": (
                        "No resumable run found for that run_id — it may have "
                        "already finished, expired, or the server restarted. "
                        "Please start a new question."
                    )
                },
            }
        )
        return
    remove_run(run_id)  # re-registered below only if interrupted again

    agent = record.agent
    trace = record.trace
    step_number = record.step_number
    folder_path = Path(record.folder)
    use_index = record.use_index
    enable_semantic = record.enable_semantic
    enable_metadata = record.enable_metadata
    index_folders = record.index_folders
    resolved_database_url = record.database_url
    original_task = record.original_task

    try:
        clear_index_context()
        if use_index:
            storage = PostgresStorage(resolved_database_url)
            index_storage = storage
            set_index_context(index_folders, resolved_database_url)

        set_search_flags(
            enable_semantic=enable_semantic and use_index,
            enable_metadata=enable_metadata and use_index,
        )

        run_started_at = time.monotonic()
        pending_llm_calls: list[LLMCallStats] = []

        async def _collect_llm_call(stats: LLMCallStats) -> None:
            pending_llm_calls.append(stats)

        async def _flush_llm_calls() -> None:
            while pending_llm_calls:
                stats = pending_llm_calls.pop(0)
                await websocket.send_json(
                    {
                        "type": "llm_call",
                        "data": {
                            "purpose": stats.purpose,
                            "model": stats.model,
                            "prompt_tokens": stats.prompt_tokens,
                            "completion_tokens": stats.completion_tokens,
                            "thinking_tokens": stats.thinking_tokens,
                            "duration_ms": round(stats.duration_ms),
                        },
                    }
                )

        # The agent's original hook closure was bound to the interrupted
        # connection's websocket — rebind it to this one before driving the
        # agent any further.
        agent.set_llm_call_hook(_collect_llm_call)

        await websocket.send_json(
            {
                "type": "start",
                "data": {
                    "task": original_task,
                    "folder": str(folder_path),
                    "use_index": use_index,
                    "run_id": run_id,
                    "resumed": True,
                },
            }
        )
        await websocket.send_json(
            {
                "type": "status",
                "data": {
                    "label": "Continuing",
                    "detail": "Resuming from where the run left off",
                },
            }
        )

        final_result = ""
        result_error: str | None = None

        gen = resume_agent_run(
            agent,
            use_index=use_index,
            current_directory=str(folder_path),
            initial_task=original_task,
        )
        send_value: str | None = None
        while True:
            try:
                event = await gen.asend(send_value)
            except StopAsyncIteration:
                break
            send_value = None
            await _flush_llm_calls()

            if isinstance(event, ToolCallEvent):
                step_number += 1
                await websocket.send_json(
                    _tool_call_ws_message(
                        event,
                        step_number=step_number,
                        trace=trace,
                        index_storage=index_storage,
                    )
                )
            elif isinstance(event, GoDeeperEvent):
                step_number += 1
                await websocket.send_json(
                    _go_deeper_ws_message(event, step_number=step_number, trace=trace)
                )
            elif isinstance(event, AskHumanEvent):
                step_number += 1
                await websocket.send_json(
                    _ask_human_ws_message(event, step_number=step_number)
                )
                response_data = await websocket.receive_json()
                if response_data.get("type") == "human_response":
                    send_value = response_data.get("response", "")
                else:
                    send_value = ""
            elif isinstance(event, ExplorationEndEvent):
                final_result = event.final_result or ""
                result_error = event.error

        await _flush_llm_calls()

        await _finish_run(
            websocket,
            run_id=run_id,
            agent=agent,
            trace=trace,
            step_number=step_number,
            folder_path=folder_path,
            use_index=use_index,
            final_result=final_result,
            result_error=result_error,
            run_started_at=run_started_at,
            flush_llm_calls=_flush_llm_calls,
        )
    except Exception:
        _register_if_resumable(
            run_id=run_id,
            agent=agent,
            trace=trace,
            step_number=step_number,
            folder_path=folder_path,
            use_index=use_index,
            enable_semantic=enable_semantic,
            enable_metadata=enable_metadata,
            index_folders=index_folders,
            database_url=resolved_database_url,
            original_task=original_task,
        )
        raise
    finally:
        if index_storage is not None:
            index_storage.close()
        set_search_flags(enable_semantic=False, enable_metadata=False)
        clear_index_context()


@app.websocket("/ws/explore")
async def websocket_explore(websocket: WebSocket):
    """
    WebSocket endpoint for real-time exploration streaming.

    Protocol:
    1. Client sends either:
       - {"task": "user question", ...} to start a new run, or
       - {"type": "resume", "run_id": "..."} to continue a run that was
         interrupted (dropped connection, manual stop, or an
         unrecoverable error) before it completed — see `runs.py` for
         what makes a run resumable.
    2. Server streams events: {"type": "...", "data": {...}}
    3. Final event: {"type": "complete", "data": {...}}

    The `start` event always carries a `run_id`. If the connection is lost
    or the run ends up erroring before a `complete` event, the client
    should offer to resume with that same run_id on a fresh connection.
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()

        if not internal_token_valid(data.get("internal_token")):
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": "Invalid or missing internal token."},
                }
            )
            return

        if data.get("type") == "resume":
            await _run_resume_session(websocket, str(data.get("run_id", "")))
        else:
            await _run_fresh_session(websocket, data)

    except WebSocketDisconnect:
        logger.warning("Client disconnected from /ws/explore before completion")
    except Exception as e:
        # Log unconditionally, before attempting to notify the client — if the
        # underlying cause is a broken connection, the send below will raise
        # the same kind of error again and, unguarded, would propagate out of
        # this handler and kill the socket with zero trace of what happened
        # (surfaces to the backend only as "Core stream closed before
        # completion", with nothing in these logs to explain why).
        logger.exception("Unhandled error in /ws/explore")
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            logger.warning(
                "Failed to deliver error event over /ws/explore; "
                "connection likely already broken"
            )


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
