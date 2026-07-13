"""
FsExplorer Agent for filesystem exploration using Google Gemini.

This module contains the agent that interacts with the Gemini AI model
to make decisions about filesystem exploration actions.
"""

import fnmatch
import os
import re
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Any
from dataclasses import dataclass

from dotenv import load_dotenv

from .llm import ChatTurn, LLMClient, LLMUsage, get_llm_client
from .models import Action, ActionType, ContextSummary, Tools
from fs_explorer_shared.fs import (
    read_file as fs_read_file,
    grep_file_content as fs_grep_file_content,
    glob_paths as fs_glob_paths,
)
from fs_explorer_shared.embeddings import EmbeddingProvider
from fs_explorer_shared.index_config import resolve_database_url
from .search import (
    IndexedQueryEngine,
    MetadataFilterParseError,
    supported_filter_syntax,
)
from fs_explorer_shared.storage import PostgresStorage

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


# =============================================================================
# Token Usage Tracking
# =============================================================================

# Gemini Flash pricing (per million tokens)
GEMINI_FLASH_INPUT_COST_PER_MILLION = 0.075
GEMINI_FLASH_OUTPUT_COST_PER_MILLION = 0.30

# Gemini's hard input-token ceiling (the "1048576" in the
# `INVALID_ARGUMENT: input token count exceeds the maximum number of
# tokens allowed 1048576` error). Used only to compute a context-usage
# percentage for the client/dashboard, not to enforce any limit here.
GEMINI_MAX_CONTEXT_TOKENS = 1_048_576

# Fraction of GEMINI_MAX_CONTEXT_TOKENS at which the agent proactively
# compacts its own mid-run chat history (see `_maybe_summarize_history`)
# instead of waiting to actually hit the ceiling and error out. Every
# `take_action()` call resends the full history, so this is checked after
# each one using that call's own prompt token count as the "current size"
# signal.
CONTEXT_SUMMARY_THRESHOLD_RATIO = float(
    os.getenv("FS_EXPLORER_CONTEXT_SUMMARY_THRESHOLD", "0.85")
)

# How many of the most recent chat turns to keep verbatim when
# summarizing — recent tool results are more likely to still be relevant
# to the very next decision than older ones.
_CONTEXT_SUMMARY_KEEP_RECENT_TURNS = 6
# The first turn (initial task framing) is always kept verbatim too, so
# the agent never loses sight of the original task after a summarization.
_CONTEXT_SUMMARY_KEEP_LEADING_TURNS = 1


@dataclass
class TokenUsage:
    """
    Track token usage and costs across the session.

    Maintains running totals of API calls, token counts, and provides
    cost estimates based on Gemini Flash pricing.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    # Track content sizes
    tool_result_chars: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0

    # Number of times this run's mid-exploration chat history was
    # compacted by `FsExplorerAgent._maybe_summarize_history`.
    context_summaries: int = 0

    def add_api_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        thinking_tokens: int = 0,
    ) -> None:
        """Record token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.thinking_tokens += thinking_tokens
        self.total_tokens += prompt_tokens + completion_tokens + thinking_tokens
        self.api_calls += 1

    def add_tool_result(self, result: str, tool_name: str) -> None:
        """Record metrics from a tool execution."""
        self.tool_result_chars += len(result)
        if tool_name == "parse_file":
            self.documents_parsed += 1
        elif tool_name == "scan_folder":
            # Count documents in scan result by counting document markers
            self.documents_scanned += result.count("│ [")
        elif tool_name == "preview_file":
            self.documents_parsed += 1

    def _calculate_cost(self) -> tuple[float, float, float]:
        """Calculate estimated costs based on Gemini Flash pricing."""
        input_cost = (
            self.prompt_tokens / 1_000_000
        ) * GEMINI_FLASH_INPUT_COST_PER_MILLION
        output_cost = (
            self.completion_tokens / 1_000_000
        ) * GEMINI_FLASH_OUTPUT_COST_PER_MILLION
        return input_cost, output_cost, input_cost + output_cost

    def context_usage_ratio(self, max_context_tokens: int) -> float:
        """Fraction of the model's context window the last prompt likely used.

        `prompt_tokens` is a running total across every call so far, not the
        size of any single request, but since each call resends the full
        chat history it also approximates "how big is the history right
        now" — good enough for a warning threshold, not for billing.
        """
        if max_context_tokens <= 0:
            return 0.0
        return self.prompt_tokens / max_context_tokens

    def summary(self) -> str:
        """Generate a formatted summary of token usage and costs."""
        input_cost, output_cost, total_cost = self._calculate_cost()

        return f"""
═══════════════════════════════════════════════════════════════
                      TOKEN USAGE SUMMARY
═══════════════════════════════════════════════════════════════
  API Calls:           {self.api_calls}
  Prompt Tokens:       {self.prompt_tokens:,}
  Completion Tokens:   {self.completion_tokens:,}
  Thinking Tokens:     {self.thinking_tokens:,}
  Total Tokens:        {self.total_tokens:,}
───────────────────────────────────────────────────────────────
  Documents Scanned:   {self.documents_scanned}
  Documents Parsed:    {self.documents_parsed}
  Tool Result Chars:   {self.tool_result_chars:,}
───────────────────────────────────────────────────────────────
  Est. Cost (Gemini Flash):
    Input:  ${input_cost:.4f}
    Output: ${output_cost:.4f}
    Total:  ${total_cost:.4f}
═══════════════════════════════════════════════════════════════
"""


# =============================================================================
# Tool Registry
# =============================================================================


@dataclass(frozen=True)
class IndexedCorpus:
    """Resolved corpus available to indexed retrieval tools."""

    root_folder: str
    corpus_id: str


@dataclass(frozen=True)
class IndexContext:
    """Execution context for indexed retrieval tools."""

    root_folders: tuple[str, ...]
    database_url: str


_INDEX_CONTEXT: IndexContext | None = None
_EMBEDDING_PROVIDER: EmbeddingProvider | None = None
_FIELD_CATALOG_SHOWN: bool = False
_ENABLE_SEMANTIC: bool = False
_ENABLE_METADATA: bool = False


def set_search_flags(
    *, enable_semantic: bool = False, enable_metadata: bool = False
) -> None:
    """Configure which indexed retrieval paths are active."""
    global _ENABLE_SEMANTIC, _ENABLE_METADATA
    _ENABLE_SEMANTIC = enable_semantic
    _ENABLE_METADATA = enable_metadata


def get_search_flags() -> tuple[bool, bool]:
    """Return (enable_semantic, enable_metadata)."""
    return _ENABLE_SEMANTIC, _ENABLE_METADATA


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    """Set the embedding provider for vector search in indexed tools."""
    global _EMBEDDING_PROVIDER
    _EMBEDDING_PROVIDER = provider


def set_index_context(
    folder: str | list[str] | tuple[str, ...],
    database_url: str | None = None,
) -> None:
    """Enable indexed tools for one or more folder corpora."""
    global _INDEX_CONTEXT, _EMBEDDING_PROVIDER
    folders = (folder,) if isinstance(folder, str) else tuple(folder)
    _INDEX_CONTEXT = IndexContext(
        root_folders=tuple(str(Path(item).resolve()) for item in folders),
        database_url=resolve_database_url(database_url),
    )
    # Auto-create embedding provider if API key available
    if _EMBEDDING_PROVIDER is None:
        try:
            _EMBEDDING_PROVIDER = EmbeddingProvider()
        except ValueError:
            pass


def clear_index_context() -> None:
    """Disable indexed tools for the current process."""
    global _INDEX_CONTEXT, _EMBEDDING_PROVIDER, _FIELD_CATALOG_SHOWN
    global _ENABLE_SEMANTIC, _ENABLE_METADATA
    _INDEX_CONTEXT = None
    _EMBEDDING_PROVIDER = None
    _FIELD_CATALOG_SHOWN = False
    _ENABLE_SEMANTIC = False
    _ENABLE_METADATA = False


def _get_index_storage_and_corpora() -> tuple[
    PostgresStorage | None, list[IndexedCorpus], str | None
]:
    if _INDEX_CONTEXT is None:
        return None, [], "Index context is not configured. Re-run with `--use-index`."

    storage = PostgresStorage(_INDEX_CONTEXT.database_url)
    corpora: list[IndexedCorpus] = []
    missing: list[str] = []
    for root_folder in _INDEX_CONTEXT.root_folders:
        corpus_id = storage.get_corpus_id(root_folder)
        if corpus_id is None:
            missing.append(root_folder)
            continue
        corpora.append(IndexedCorpus(root_folder=root_folder, corpus_id=corpus_id))

    if not corpora:
        storage.close()
        return (
            None,
            [],
            f"No index found for folders: {', '.join(_INDEX_CONTEXT.root_folders)}. "
            "Run `explore index <folder>` first.",
        )
    return storage, corpora, None


def _get_index_storage_and_corpus() -> tuple[
    PostgresStorage | None, str | None, str | None
]:
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return storage, None, error
    return storage, corpora[0].corpus_id, None


def _clean_excerpt(text: str, max_chars: int = 320) -> str:
    squashed = re.sub(r"\s+", " ", text).strip()
    if len(squashed) <= max_chars:
        return squashed
    return f"{squashed[:max_chars]}..."


def _display_path(document: dict[str, Any]) -> str:
    return str(document.get("relative_path") or document.get("absolute_path") or "")


def _display_name(document: dict[str, Any]) -> str:
    name = Path(_display_path(document)).name
    name = re.sub(r"^\d+-", "", name) or name
    name = re.sub(r"\.[a-zA-Z0-9]+$", "", name)
    name = name.replace("_x1", "(").replace("x2_", ")_").replace("x2", ")")
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name or "Indexed document"


def _all_index_documents(
    storage: PostgresStorage,
    corpora: list[IndexedCorpus],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for corpus in corpora:
        for document in storage.list_documents(
            corpus_id=corpus.corpus_id,
            include_deleted=False,
        ):
            copied = dict(document)
            copied["corpus_id"] = corpus.corpus_id
            copied["corpus_root"] = corpus.root_folder
            documents.append(copied)
    return documents


def _resolve_index_document(
    storage: PostgresStorage,
    corpora: list[IndexedCorpus],
    file_path: str,
) -> dict[str, Any] | None:
    value = str(file_path).strip()
    if not value:
        return None

    if value.startswith("doc_"):
        document = storage.get_document(doc_id=value)
        if document is not None and not document.get("is_deleted"):
            return document

    documents = _all_index_documents(storage, corpora)
    needle = value.replace("\\", "/")
    needle_path = Path(value).name
    resolved: str | None = None
    try:
        resolved = str(Path(value).expanduser().resolve())
    except Exception:
        resolved = None

    exact_matches: list[dict[str, Any]] = []
    suffix_matches: list[dict[str, Any]] = []
    basename_matches: list[dict[str, Any]] = []
    for document in documents:
        relative_path = str(document["relative_path"]).replace("\\", "/")
        absolute_path = str(document["absolute_path"])
        display_name = _display_name(document)
        basename = Path(relative_path).name

        if (
            value == document["id"]
            or needle == relative_path
            or resolved == absolute_path
        ):
            exact_matches.append(document)
            continue
        if needle and (
            relative_path.endswith(needle)
            or absolute_path.endswith(needle)
            or relative_path.endswith(f"/{needle}")
        ):
            suffix_matches.append(document)
            continue
        if needle_path in {basename, display_name, Path(absolute_path).name}:
            basename_matches.append(document)

    for matches in (exact_matches, suffix_matches, basename_matches):
        if matches:
            return matches[0]
    return None


def _chunk_locator(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    locators: list[str] = []
    for key in (
        "article_no",
        "paragraph_no",
        "clause_label",
        "subclause_label",
        "document_number",
    ):
        value = metadata.get(key)
        if value:
            locators.append(f"{key}={value}")
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        locators.append(
            "heading=" + " > ".join(str(item) for item in heading_path[-3:])
        )
    if locators:
        return " | " + "; ".join(locators)
    return ""


def _document_from_chunks(
    storage: PostgresStorage,
    document: dict[str, Any],
    *,
    max_chars: int | None = None,
) -> str:
    chunks = storage.list_document_chunks(doc_id=str(document["id"]))
    header = (
        f"=== INDEXED DOCUMENT FROM CHUNKS ===\n"
        f"doc_id: {document['id']}\n"
        f"title: {_display_name(document)}\n"
        f"content_source: core_chunks.text\n\n"
    )
    if not chunks:
        content = str(document.get("content") or "")
        body = content if max_chars is None else content[:max_chars]
        if max_chars is not None and len(content) > max_chars:
            body += (
                f"\n\n[... PREVIEW TRUNCATED. Full document has {len(content):,} "
                "characters. Use parse_file() or get_document() for full indexed content ...]"
            )
        return header + body

    lines: list[str] = [header]
    total = 0
    truncated = False
    for chunk in chunks:
        chunk_text = str(chunk["text"])
        chunk_header = (
            f"\n--- chunk {chunk['position']} "
            f"({chunk.get('chunk_type') or 'text'}, chars "
            f"{chunk['start_char']}-{chunk['end_char']})"
            f"{_chunk_locator(chunk)} ---\n"
        )
        addition = chunk_header + chunk_text.strip() + "\n"
        if max_chars is not None and total + len(addition) > max_chars:
            remaining = max(max_chars - total, 0)
            if remaining:
                lines.append(addition[:remaining])
            truncated = True
            break
        lines.append(addition)
        total += len(addition)

    if truncated:
        lines.append(
            f"\n\n[... PREVIEW TRUNCATED. Document has {len(chunks)} indexed chunks. "
            "Use parse_file() or get_document() for all chunk text ...]"
        )
    return "".join(lines)


def _indexed_preview_for_document(
    storage: PostgresStorage,
    document: dict[str, Any],
    preview_chars: int,
) -> str:
    content = _document_from_chunks(storage, document, max_chars=preview_chars)
    return content


def _index_tools_available() -> bool:
    return _INDEX_CONTEXT is not None


def describe_indexed_context() -> str:
    """Describe active indexed corpora for prompts instead of raw filesystem files."""
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return error or "Indexed context is unavailable."
    assert storage is not None
    documents = _all_index_documents(storage, corpora)
    if not documents:
        return "No indexed documents found in the selected directories."

    lines = [
        "Content is available from the index database, not raw upload files.",
        "Use the standard tools; when an index is active they read core_chunks/core_documents.",
        "",
        "INDEXED DOCUMENTS:",
    ]
    for idx, document in enumerate(documents, start=1):
        lines.append(
            f"- [{idx}] doc_id={document['id']} title={_display_name(document)}"
        )
    return "\n".join(lines)


def _not_indexed_message(file_path: str) -> str:
    """
    Fallback for preview_file/parse_file/scan_folder when a document isn't
    in the index. This service only ever reads indexed chunks — raw Docling
    parsing lives in the indexer service, not here.
    """
    return (
        f"'{file_path}' was not found in the index. This service only reads "
        "indexed documents. Run the indexer for this folder, then retry."
    )


def _indexed_read_file(file_path: str) -> str:
    if not _index_tools_available():
        return fs_read_file(file_path)
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return fs_read_file(file_path)
    assert storage is not None
    document = _resolve_index_document(storage, corpora, file_path)
    if document is None:
        return fs_read_file(file_path)
    return _document_from_chunks(storage, document)


def _indexed_preview_file(file_path: str, max_chars: int = 3000) -> str:
    if not _index_tools_available():
        return _not_indexed_message(file_path)
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return _not_indexed_message(file_path)
    assert storage is not None
    document = _resolve_index_document(storage, corpora, file_path)
    if document is None:
        return _not_indexed_message(file_path)
    return _indexed_preview_for_document(storage, document, max_chars)


def _indexed_parse_file(file_path: str) -> str:
    if not _index_tools_available():
        return _not_indexed_message(file_path)
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return _not_indexed_message(file_path)
    assert storage is not None
    document = _resolve_index_document(storage, corpora, file_path)
    if document is None:
        return _not_indexed_message(file_path)
    return _document_from_chunks(storage, document)


def _indexed_grep_file_content(file_path: str, pattern: str) -> str:
    if not _index_tools_available():
        return fs_grep_file_content(file_path, pattern)
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return fs_grep_file_content(file_path, pattern)
    assert storage is not None

    try:
        regex = re.compile(pattern=pattern, flags=re.MULTILINE | re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regex pattern {pattern!r}: {exc}"

    if file_path.strip() in {"*", "all", "."}:
        documents = _all_index_documents(storage, corpora)
    else:
        document = _resolve_index_document(storage, corpora, file_path)
        if document is None:
            return fs_grep_file_content(file_path, pattern)
        documents = [document]

    lines: list[str] = []
    for document in documents:
        for chunk in storage.list_document_chunks(doc_id=str(document["id"])):
            matches = regex.findall(str(chunk["text"]))
            if not matches:
                continue
            rendered = [
                match
                if isinstance(match, str)
                else " ".join(str(item) for item in match)
                for match in matches[:8]
            ]
            lines.append(
                f"- {_display_name(document)} doc_id={document['id']} "
                f"chunk={chunk['position']}: " + "; ".join(rendered)
            )

    if lines:
        return (
            f"MATCHES for {pattern} in indexed chunks ({file_path}):\n\n"
            + "\n".join(lines)
        )
    return "No matches found in indexed chunks"


def _document_matches_directory(document: dict[str, Any], directory: str) -> bool:
    if directory in {"", ".", "./"}:
        return True
    needle = str(directory).replace("\\", "/").rstrip("/")
    relative_path = str(document["relative_path"]).replace("\\", "/")
    absolute_path = str(document["absolute_path"]).replace("\\", "/")
    corpus_root = str(document.get("corpus_root") or "").replace("\\", "/")
    if needle in {corpus_root, absolute_path, str(Path(absolute_path).parent)}:
        return True
    return relative_path.startswith(needle + "/") or absolute_path.startswith(
        needle + "/"
    )


def _indexed_scan_folder(
    directory: str,
    max_workers: int = 4,
    preview_chars: int = 1500,
) -> str:
    if not _index_tools_available():
        return _not_indexed_message(directory)
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return _not_indexed_message(directory)
    assert storage is not None

    documents = [
        document
        for document in _all_index_documents(storage, corpora)
        if _document_matches_directory(document, directory)
    ]
    if not documents:
        return _not_indexed_message(directory)

    output = [
        "═══════════════════════════════════════════════════════════════",
        f"  INDEXED CHUNK SCAN: {directory}",
        f"  Found {len(documents)} indexed documents",
        "  Source: core_documents + core_chunks, not raw upload files",
        "═══════════════════════════════════════════════════════════════",
        "",
    ]
    for idx, document in enumerate(documents, start=1):
        preview = _indexed_preview_for_document(storage, document, preview_chars)
        preview_lines = preview.splitlines()
        output.extend(
            [
                "┌─────────────────────────────────────────────────────────────",
                f"│ [{idx}/{len(documents)}] {_display_name(document)}",
                f"│ doc_id: {document['id']}",
                f"│ title: {_display_name(document)}",
                "├─────────────────────────────────────────────────────────────",
            ]
        )
        for line in preview_lines[:18]:
            output.append(f"│ {line}")
        if len(preview_lines) > 18:
            output.append("│ ... (preview truncated)")
        output.append("└─────────────────────────────────────────────────────────────")
        output.append("")

    output.extend(
        [
            "NEXT STEPS:",
            "1. Use semantic_search(query=...) for relevant chunks.",
            "2. Use parse_file(file_path=...) or get_document(doc_id=...) for full chunk text.",
            "3. Use grep(file_path='all', pattern=...) to scan chunk text across indexed docs.",
        ]
    )
    return "\n".join(output)


def _indexed_glob_paths(directory: str, pattern: str) -> str:
    if not _index_tools_available():
        return fs_glob_paths(directory, pattern)
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return fs_glob_paths(directory, pattern)
    assert storage is not None

    matches: list[str] = []
    for document in _all_index_documents(storage, corpora):
        if not _document_matches_directory(document, directory):
            continue
        candidates = [
            str(document["relative_path"]),
            str(document["absolute_path"]),
            _display_name(document),
            Path(str(document["relative_path"])).name,
        ]
        if any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates):
            matches.append(f"{_display_name(document)} (doc_id={document['id']})")

    if matches:
        return f"MATCHES for {pattern} in indexed documents:\n\n- " + "\n- ".join(
            matches
        )
    return "No matches found in indexed documents"


def semantic_search(
    query: str,
    filters: str | None = None,
    limit: int = 5,
    as_of_date: str | None = None,
) -> str:
    """Search indexed chunks and return ranked excerpts.

    `as_of_date` (YYYY-MM-DD) restricts results to chunks whose validity
    interval covers that date — pass it whenever the user's question refers
    to a specific date/time period. Omit it (defaults to today) for
    "what's the current rule" questions.
    """
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return error
    assert storage is not None and corpora

    engine = IndexedQueryEngine(storage, embedding_provider=_EMBEDDING_PROVIDER)
    hits: list[Any] = []
    try:
        for corpus in corpora:
            hits.extend(
                engine.search(
                    corpus_id=corpus.corpus_id,
                    query=query,
                    filters=filters,
                    limit=limit,
                    enable_semantic=_ENABLE_SEMANTIC,
                    enable_metadata=_ENABLE_METADATA,
                    as_of_date=as_of_date,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        hits = hits[: max(limit, 1)]
    except MetadataFilterParseError as exc:
        return f"Invalid metadata filter: {exc}\n{supported_filter_syntax()}"
    except ValueError as exc:
        return f"Metadata filter error: {exc}"

    if not hits:
        if filters:
            return f"No indexed matches found for query={query!r} with filters={filters!r}."
        return f"No indexed matches found for query: {query!r}"

    lines = [
        "=== INDEXED SEARCH RESULTS ===",
        f"Query: {query}",
    ]
    if filters:
        lines.append(f"Filters: {filters}")
    if as_of_date:
        lines.append(f"As of date: {as_of_date}")
    lines.append("")
    for idx, hit in enumerate(hits, start=1):
        position = hit.position if hit.position is not None else "<metadata>"
        title = _display_name(
            {"relative_path": hit.relative_path, "absolute_path": hit.absolute_path}
        )
        chunk_locator = _chunk_locator({"metadata": hit.metadata})
        lines.extend(
            [
                f"[{idx}] doc_id: {hit.doc_id}",
                f"    title: {title}",
                f"    match: {hit.matched_by}",
                f"    chunk_id: {hit.chunk_id or '<metadata>'}",
                f"    chunk_path: {title}{chunk_locator}",
                f"    chunk_position: {position}",
                f"    semantic_score: {hit.semantic_score}",
                f"    metadata_score: {hit.metadata_score}",
                f"    score: {hit.score:.2f}",
                f"    excerpt: {_clean_excerpt(hit.text)}",
                "",
            ]
        )
    lines.append(
        "Use get_document(doc_id=...) to read full content for the most relevant documents."
    )

    # Include a rich field catalog on the first search so the agent can
    # construct effective metadata filters.
    global _FIELD_CATALOG_SHOWN
    if not _FIELD_CATALOG_SHOWN:
        for corpus in corpora:
            active_schema = storage.get_active_schema(corpus_id=corpus.corpus_id)
            if active_schema is None:
                continue
            schema_fields = active_schema.schema_def.get("fields")
            if isinstance(schema_fields, list) and schema_fields:
                field_names = [
                    str(f["name"])
                    for f in schema_fields
                    if isinstance(f, dict) and isinstance(f.get("name"), str)
                ]
                field_values = storage.get_metadata_field_values(
                    corpus_id=corpus.corpus_id,
                    field_names=field_names,
                )
                field_descs: list[str] = []
                for field in schema_fields:
                    if not isinstance(field, dict) or not isinstance(
                        field.get("name"), str
                    ):
                        continue
                    name = field["name"]
                    ftype = field.get("type", "string")
                    desc = field.get("description", "")
                    entry = f"{name} ({ftype})"
                    if desc:
                        entry += f": {desc}"
                    vals = field_values.get(name, [])
                    if ftype == "boolean":
                        entry += " Values: true, false"
                    elif ftype in {"integer", "number"} and vals:
                        nums = []
                        for v in vals:
                            try:
                                nums.append(float(v))
                            except (TypeError, ValueError):
                                pass
                        if nums:
                            entry += f" Range: {min(nums):.6g}-{max(nums):.6g}"
                    elif vals:
                        if "enum" in field:
                            entry += f" Values: {field['enum']}"
                        else:
                            entry += f" Values: {', '.join(repr(v) for v in vals)}"
                    elif "enum" in field:
                        entry += f" Values: {field['enum']}"
                    field_descs.append(entry)
                if field_descs:
                    lines.append("")
                    lines.append(
                        "Available filter fields for semantic_search(filters=...):"
                    )
                    for desc in field_descs:
                        lines.append(f"  - {desc}")
                _FIELD_CATALOG_SHOWN = True
                break

    return "\n".join(lines)


def get_document(doc_id: str) -> str:
    """Return full document content by id from the active index context."""
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return error
    assert storage is not None

    document = _resolve_index_document(storage, corpora, doc_id)
    if document is None:
        return f"No indexed document found for doc_id={doc_id!r}"
    if document["is_deleted"]:
        return f"Document {doc_id} is marked as deleted in the index."

    return _document_from_chunks(storage, document)


def list_indexed_documents() -> str:
    """List indexed documents for active corpora."""
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return error
    assert storage is not None and corpora

    documents = _all_index_documents(storage, corpora)
    if not documents:
        return "No indexed documents found for active corpora."

    lines = ["=== INDEXED DOCUMENTS ==="]
    for idx, document in enumerate(documents, start=1):
        lines.append(f"[{idx}] doc_id={document['id']} title={_display_name(document)}")
    lines.append("")
    lines.append(
        "Use semantic_search(...) to find relevant chunks, then get_document(doc_id=...) "
        "or parse_file(file_path=...) to read chunk text."
    )
    return "\n".join(lines)


TOOLS: dict[Tools, Callable[..., str]] = {
    "read": _indexed_read_file,
    "grep": _indexed_grep_file_content,
    "glob": _indexed_glob_paths,
    "scan_folder": _indexed_scan_folder,
    "preview_file": _indexed_preview_file,
    "parse_file": _indexed_parse_file,
    "semantic_search": semantic_search,
    "get_document": get_document,
    "list_indexed_documents": list_indexed_documents,
}


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are FsExplorer, an AI agent that answers questions about indexed documents.

## Available Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `scan_folder` | Scan indexed documents and previews from `core_chunks` | `directory` |
| `preview_file` | Quick preview of one indexed document from chunk text | `file_path` |
| `parse_file` | **DEEP READ** - full indexed chunk text for a document | `file_path` |
| `read` | Read indexed chunk text for a document | `file_path` |
| `grep` | Search regex pattern in indexed chunk text | `file_path`, `pattern` |
| `glob` | Find indexed document paths matching a pattern | `directory`, `pattern` |
| `semantic_search` | Search indexed chunks and metadata-filtered docs, then union/rank results | `query`, `filters`, `limit`, `as_of_date` |
| `get_document` | Read full indexed document by document id | `doc_id` |
| `list_indexed_documents` | List indexed documents for active corpus | none |

## Indexed Retrieval Strategy

When indexed tools are available:
1. Start with `semantic_search` to quickly find relevant documents.
2. Use `get_document` for the top candidate doc IDs, or `parse_file(file_path=...)` if you have a path.
3. Treat `parse_file`, `preview_file`, `read`, `grep`, `glob`, and `scan_folder` as chunk-backed tools. They read `core_chunks.text`/`core_documents`, not raw upload files.
4. If indexed tools report index is unavailable, only then fall back to filesystem tools.

Filter syntax for `semantic_search(filters=...)`:
- `field=value`
- `field!=value`
- `field>=number`, `field<=number`, `field>number`, `field<number`
- `field in (a, b, c)`
- `field~substring`
- combine conditions with comma or `and`

## Time-Sensitive Questions: `semantic_search(as_of_date=...)`

Regulations change over time. Every indexed chunk has a validity interval
(when it started/stopped being in force); `semantic_search` only returns
chunks whose interval covers `as_of_date` (defaults to **today** when
omitted, so ordinary questions automatically get only what's currently in
force — amended-away or not-yet-effective text is excluded without you
doing anything).

- If the user's question references a specific date or period (e.g. "2023
  yılında", "1 Ocak 2027'den itibaren", "geçen yıl", "o zamanki kural neydi"),
  extract that date, convert it to `YYYY-MM-DD`, and pass it as
  `semantic_search(query=..., as_of_date="YYYY-MM-DD")`.
- If the question is about "the current rule" / "now" / doesn't mention a
  date at all, omit `as_of_date` entirely — do not pass today's date
  explicitly, just leave it out.
- If comparing two points in time ("X, Y'ye göre nasıl değişti"), call
  `semantic_search` twice, once per `as_of_date`, and compare the results
  in your answer.

## Three-Phase Document Exploration Strategy

### PHASE 1: Indexed Parallel Scan (Use `semantic_search` or `scan_folder`)
When you encounter a folder with documents:
1. Prefer `semantic_search` for the user query.
2. Use `scan_folder` when you need an overview of all indexed documents.
3. This gives you quick previews from chunk text, even when raw upload files were removed.
3. In your **reason**, explicitly list your document categorization:
   - **RELEVANT**: Documents clearly related to the query (list them)
   - **MAYBE**: Documents that might be relevant (list them)
   - **SKIP**: Documents not relevant (list them)

### PHASE 2: Deep Dive (Use `get_document` or `parse_file`)
1. Use `parse_file` on documents marked RELEVANT
2. Use `get_document(doc_id=...)` when you have a doc_id from `semantic_search`
3. In your **reason**, explain what key information you found
3. **WATCH FOR CROSS-REFERENCES** - look for mentions like:
   - "See Exhibit A/B/C..."
   - "As stated in the [Document Name]..."
   - "Refer to [filename]..."
   - Document numbers, exhibit labels, or file names
4. In your **reason**, note any cross-references you discovered

### PHASE 3: Backtracking (Revisit if Cross-Referenced)
**CRITICAL**: If a document you're reading references another document that you SKIPPED:
1. In your **reason**, explain: "Found cross-reference to [document] - need to backtrack"
2. Use `semantic_search`, `preview_file`, `parse_file`, or `get_document` to read the referenced document from chunks
3. Continue this until all relevant cross-references are resolved

## Providing Detailed Reasoning

Your `reason` field is displayed to the user, so make it informative:
- After scanning: List which documents you're categorizing as RELEVANT/MAYBE/SKIP and why
- After parsing: Summarize key findings and any cross-references discovered
- When backtracking: Explain which reference led you back to a skipped document

## CRITICAL: Citation Requirements for Final Answers

When providing your final answer, you MUST include citations for ALL factual claims:

### Citation Format
Use inline citations in this format: `[Readable Document Title, Article/Section]`

Example:
> Yabancı taşıyıcıların Türkiye'de acente bulundurma zorunluluğu yoktur [Gümrük Genel Tebliği, Madde 54(1)].

### Citation Rules
1. **Every factual claim needs a citation** - dates, numbers, names, terms, etc.
2. **Be specific** - include section numbers, article numbers, or page references when available
3. **Use readable legal document titles** - not local filesystem paths and not raw slugified filenames
4. **Multiple sources** - if information comes from multiple documents, cite all of them
5. Never expose `/home/...`, `backend/storage/...`, `_indexes`, `_sessions`, or raw tool paths in the final answer.
6. Do not write `Source:` inside citations. Prefer `[Gümrük Genel Tebliği, Madde 54(1)]` over `[Source: gumruk_....docx, Madde 54(1)]`.

### Final Answer Structure
Your final answer should:
1. **Start with a direct answer** to the user's question
2. **Provide details** with inline citations
3. **End with a Sources section** listing only readable document titles:

```
## Sources
- Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)
- Genelge 2017/06
```

## Example Workflow

```
User asks: "What is the purchase price?"

1. scan_folder("./documents/")
   Reason: "Scanned 10 documents. Categorizing:
   - RELEVANT: purchase_agreement.pdf (mentions 'Purchase Price' in preview)
   - RELEVANT: financial_terms.pdf (contains pricing tables)
   - MAYBE: exhibits.pdf (referenced by other docs)
   - SKIP: employee_handbook.pdf, hr_policies.pdf (unrelated to pricing)"

2. parse_file("purchase_agreement.pdf")
   Reason: "Found purchase price of $50M in Section 2.1. Document references 
   'Exhibit B for price adjustments' - need to check exhibits.pdf next."

3. parse_file("exhibits.pdf")  [BACKTRACKING]
   Reason: "Backtracking to exhibits.pdf because purchase_agreement.pdf 
   referenced it for adjustment details. Found working capital adjustment 
   formula in Exhibit B."

4. STOP with final answer including citations:
   "The purchase price is $50,000,000 [Master Purchase Agreement, Section 2.1], 
   subject to working capital adjustments [Disclosure Exhibits, Exhibit B]..."
```
"""


def _build_system_prompt(enable_semantic: bool, enable_metadata: bool) -> str:
    """Build a system prompt with retrieval-path guidance appended."""
    if enable_semantic and enable_metadata:
        hint = (
            "\n\n## Retrieval: Semantic + Metadata\n"
            "An index is available. Start with `semantic_search` using optional "
            "`filters` for best results, then use chunk-backed tools for deep dives."
        )
    elif enable_semantic:
        hint = (
            "\n\n## Retrieval: Semantic Only\n"
            "An index is available. Use `semantic_search` WITHOUT the `filters` "
            "parameter for similarity search, then use chunk-backed tools for details."
        )
    elif enable_metadata:
        hint = (
            "\n\n## Retrieval: Metadata Only\n"
            "An index is available. Use `semantic_search` with the `filters=` "
            "parameter for metadata filtering, then use chunk-backed tools for details."
        )
    else:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + hint


# =============================================================================
# Agent Implementation
# =============================================================================


@dataclass
class LLMCallStats:
    """Per-call token/timing observation, for external instrumentation.

    Distinct from `TokenUsage`, which only tracks running totals for the
    life of one agent. Callers (server.py, main.py) that want per-message
    granularity — one row per LLM call, not just an end-of-run total —
    hook `FsExplorerAgent(on_llm_call=...)` to receive one of these
    per Gemini call as it happens.
    """

    purpose: str  # "action" (tool-planning step) | "final_answer"
    model: str
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    duration_ms: float


OnLLMCall = Callable[[LLMCallStats], Awaitable[None]]


class FsExplorerAgent:
    """
    AI agent for exploring filesystems, talking to the LLM via the
    provider-agnostic `LLMClient` interface (see `llm/base.py`).

    The agent maintains a conversation history with the LLM and uses
    structured JSON output to make decisions about which actions to take.

    Attributes:
        token_usage: Tracks API call statistics and costs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        temperature: float | None = None,
        on_llm_call: OnLLMCall | None = None,
    ) -> None:
        """
        Initialize the agent with an LLM client.

        Args:
            api_key: Provider API key, used only when `llm_client` is not
                     given. Service account/Vertex AI credentials can also
                     be supplied via Google environment variables.
            llm_client: A pre-built `LLMClient` to use directly (mainly for
                        tests/mocking). Takes precedence over api_key/model.
            model: Model name override, passed to `get_llm_client`.
            temperature: Sampling temperature override, passed to
                         `get_llm_client`.
            on_llm_call: Optional async callback invoked after every
                         individual Gemini call with per-call token/timing
                         stats (see `LLMCallStats`), for callers that want
                         incremental observability instead of only the
                         cumulative `token_usage` totals.

        Raises:
            ValueError: If no Google credentials are available and no
                        llm_client given.
        """
        self._llm = llm_client or get_llm_client(
            model=model, temperature=temperature, api_key=api_key
        )
        self._chat_history: list[ChatTurn] = []
        self.token_usage = TokenUsage()
        self._on_llm_call = on_llm_call

    async def _report_llm_call(self, purpose: str, usage: "LLMUsage") -> None:
        if self._on_llm_call is None:
            return
        await self._on_llm_call(
            LLMCallStats(
                purpose=purpose,
                model=getattr(self._llm, "model", "unknown"),
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                thinking_tokens=usage.thinking_tokens,
                duration_ms=usage.duration_ms,
            )
        )

    def configure_task(self, task: str) -> None:
        """
        Add a task message to the conversation history.

        Args:
            task: The task or context to add to the conversation.
        """
        self._chat_history.append(ChatTurn(role="user", text=task))

    async def take_action(self) -> tuple[Action, ActionType] | None:
        """
        Request the next action from the LLM.

        Sends the current conversation history and receives a structured
        JSON response indicating the next action to take.

        Returns:
            A tuple of (Action, ActionType) if successful, None otherwise.
        """
        action, usage = await self._llm.generate_structured(
            self._chat_history,
            _build_system_prompt(_ENABLE_SEMANTIC, _ENABLE_METADATA),
            Action,
        )
        self.token_usage.add_api_call(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
        )
        await self._report_llm_call("action", usage)
        self._chat_history.append(ChatTurn(role="model", text=action.model_dump_json()))
        await self._maybe_summarize_history(usage.input_tokens)
        return action, action.to_action_type()

    async def _maybe_summarize_history(self, last_prompt_tokens: int) -> None:
        """
        Compact the chat history once it nears the model's context window.

        `last_prompt_tokens` is the size (in tokens) of the request that was
        just sent — since every call resends the full history, that's also
        a good proxy for "how big is `_chat_history` right now". Above
        `CONTEXT_SUMMARY_THRESHOLD_RATIO` of the model's ceiling, the middle
        of the history (everything except the original task framing and the
        most recent turns) is replaced with a single compact summary turn,
        generated by one extra LLM call over just that middle slice.
        """
        if (
            last_prompt_tokens / GEMINI_MAX_CONTEXT_TOKENS
            < CONTEXT_SUMMARY_THRESHOLD_RATIO
        ):
            return

        keep = _CONTEXT_SUMMARY_KEEP_LEADING_TURNS + _CONTEXT_SUMMARY_KEEP_RECENT_TURNS
        if len(self._chat_history) <= keep:
            return  # Nothing worth compacting yet.

        leading = self._chat_history[:_CONTEXT_SUMMARY_KEEP_LEADING_TURNS]
        recent = self._chat_history[-_CONTEXT_SUMMARY_KEEP_RECENT_TURNS:]
        middle = self._chat_history[
            _CONTEXT_SUMMARY_KEEP_LEADING_TURNS:-_CONTEXT_SUMMARY_KEEP_RECENT_TURNS
        ]
        if not middle:
            return

        summary_prompt = (
            "Summarize the exploration steps and tool results below into a "
            "compact paragraph. Preserve concrete facts: document names/paths, "
            "article/section numbers, figures, and findings relevant to the "
            "task. Omit conversational filler and raw tool-call formatting. "
            "Do not answer the task yourself — only summarize what has been "
            "discovered so far."
        )
        try:
            result, usage = await self._llm.generate_structured(
                [*middle, ChatTurn(role="user", text=summary_prompt)],
                "You are compacting an AI agent's exploration transcript to "
                "free up context window space. Be faithful and concise.",
                ContextSummary,
            )
        except Exception:
            # Summarization is a best-effort space-saving measure — if it
            # fails, carry on with the untouched (larger) history rather
            # than losing the run over it.
            return

        self.token_usage.add_api_call(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
        )
        self.token_usage.context_summaries += 1
        await self._report_llm_call("context_summary", usage)

        self._chat_history = [
            *leading,
            ChatTurn(
                role="user",
                text=(
                    "Summary of earlier exploration steps (compacted to save "
                    f"context space):\n\n{result.summary}"
                ),
            ),
            *recent,
        ]

    async def stream_final_answer(
        self,
        fallback_answer: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the final user-facing answer as plain text.

        Tool planning uses structured JSON responses. The final answer is
        generated separately so the WebSocket UI can render it incrementally.
        If the LLM client does not support streaming (or errors before
        yielding anything), falls back to yielding `fallback_answer` once.
        """
        prompt = (
            "Write the final answer for the user now. Use the evidence and tool "
            "results already gathered in this conversation. Keep all factual "
            "claims cited with the required [Readable Document Title, Article/Section] "
            "format. Do not use 'Source:' labels, local paths, backend/storage paths, "
            "or raw slugified filenames in citations. Include a final '## Sources' "
            "section with readable document titles only. Return plain text only."
        )
        stream_history = [*self._chat_history, ChatTurn(role="user", text=prompt)]
        system_prompt = _build_system_prompt(_ENABLE_SEMANTIC, _ENABLE_METADATA)

        chunks: list[str] = []
        try:
            async for text in self._llm.stream_text(stream_history, system_prompt):
                chunks.append(text)
                yield text
        except Exception:
            pass

        if not chunks:
            if fallback_answer:
                yield fallback_answer
            return

        final_text = "".join(chunks)
        self._chat_history.append(ChatTurn(role="model", text=final_text))

        usage = self._llm.last_stream_usage()
        if usage:
            self.token_usage.add_api_call(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                thinking_tokens=usage.thinking_tokens,
            )
            await self._report_llm_call("final_answer", usage)

    def call_tool(self, tool_name: Tools, tool_input: dict[str, Any]) -> None:
        """
        Execute a tool and add the result to the conversation history.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dictionary of arguments to pass to the tool.
        """
        try:
            result = TOOLS[tool_name](**tool_input)
        except Exception as e:
            result = (
                f"An error occurred while calling tool {tool_name} "
                f"with {tool_input}: {e}"
            )

        # Track tool result sizes
        self.token_usage.add_tool_result(result, tool_name)

        self._chat_history.append(
            ChatTurn(role="user", text=f"Tool result for {tool_name}:\n\n{result}")
        )

    def reset(self) -> None:
        """Reset the agent's conversation history and token tracking."""
        self._chat_history.clear()
        self.token_usage = TokenUsage()
