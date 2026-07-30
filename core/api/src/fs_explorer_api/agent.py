"""
FsExplorer Agent for filesystem exploration using Google Gemini.

This module contains the agent that interacts with the Gemini AI model
to make decisions about filesystem exploration actions.
"""

import contextvars
import asyncio
import fnmatch
import html
import logging
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Any, Literal, Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

from .llm import ChatTurn, LLMClient, LLMUsage, get_llm_client
from .llm.profile import LLMProfile, LLMRole, load_llm_profile
from .multi_agent import (
    MultiAgentResearchOrchestrator,
    MultiAgentResearchResult,
    ResearchProgress,
)
from .orchestration_models import ProblemType
from .orchestration_prompts import (
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT,
)
from .models import (
    Action,
    ActionType,
    CompletenessCheck,
    ContextSummary,
    StopAction,
    Tools,
)
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
    SearchHit,
    supported_filter_syntax,
)
from .search.ranker import RankedDocument
from fs_explorer_shared.storage import PostgresStorage

logger = logging.getLogger(__name__)

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
# instead of waiting to actually hit the ceiling and error out. Kept as a
# hard backstop against ever actually hitting the ceiling — the *cost*
# lever is `CONTEXT_SUMMARY_MAX_TOKENS` below, which triggers far earlier.
CONTEXT_SUMMARY_THRESHOLD_RATIO = float(
    os.getenv("FS_EXPLORER_CONTEXT_SUMMARY_THRESHOLD", "0.85")
)

# Absolute-token trigger for the same compaction, checked *in addition to*
# the ratio above (whichever fires first wins). A 20+-step run was
# observed summing to ~900K billed tokens even after every individual tool
# result was capped (see _DEEP_READ_MAX_CHARS and the whole-chunk search
# result limits below)
# — because every call resends the *entire* history, a long run's cost is
# dominated by how large that history is allowed to grow, not by any one
# call. The 0.85-of-1M ratio above only ever prevents a hard crash; it
# does nothing for cost until history is already ~900K tokens. This much
# lower absolute cap keeps the resent history — and therefore the
# per-call and cumulative cost — bounded throughout a long run instead of
# only in its final stretch. This is the *real* lever for total run cost:
# the agent is allowed to take as many genuine research steps as it needs
# (see _MAX_STEPS below) — what must not scale with step count is the size
# of the history resent on every single one of those steps, which is
# exactly what this cap (re-triggering every time the sawtooth refills)
# bounds. Lowered from 8000 on the observation that even 8-10 genuine
# (non-duplicate) steps between compactions was still enough to make a
# single run's *input* tokens dominate its cost.
CONTEXT_SUMMARY_MAX_TOKENS = int(
    os.getenv("FS_EXPLORER_CONTEXT_SUMMARY_MAX_TOKENS", "4000")
)

# Recent history is retained by token budget rather than a fixed count. Always
# keep the final instruction/action pair, then add at most one more turn if it
# fits. A large raw tool result is therefore summarized after the model has
# extracted its findings instead of surviving several more paid requests.
_CONTEXT_RECENT_MAX_TOKENS = int(
    os.getenv("FS_EXPLORER_CONTEXT_RECENT_MAX_TOKENS", "1000")
)
_CONTEXT_RECENT_MIN_TURNS = 2
_CONTEXT_RECENT_MAX_TURNS = 3
# The first turn (initial task framing) is always kept verbatim too, so
# the agent never loses sight of the original task after a summarization.
_CONTEXT_SUMMARY_KEEP_LEADING_TURNS = 1


def _estimated_turn_tokens(turn: ChatTurn) -> int:
    return max(1, (len(turn.text) + 3) // 4)


def _token_budgeted_recent_turns(history: list[ChatTurn]) -> list[ChatTurn]:
    selected: list[ChatTurn] = []
    used = 0
    for turn in reversed(history[_CONTEXT_SUMMARY_KEEP_LEADING_TURNS:]):
        turn_tokens = _estimated_turn_tokens(turn)
        must_keep = len(selected) < _CONTEXT_RECENT_MIN_TURNS
        if not must_keep and (
            len(selected) >= _CONTEXT_RECENT_MAX_TURNS
            or used + turn_tokens > _CONTEXT_RECENT_MAX_TOKENS
        ):
            break
        selected.append(turn)
        used += turn_tokens
    return list(reversed(selected))


# Safety-net ceiling on total agent decision-points (take_action() calls)
# in a single run — NOT a normal operating limit. A genuinely hard
# multi-document question can legitimately need many real (non-duplicate)
# research steps, and cutting that off early just produces an incomplete
# answer for no token savings worth mentioning (see CONTEXT_SUMMARY_MAX_TOKENS
# above for the actual cost control — that bounds the size of *each* step,
# not how many steps are allowed). This only exists to stop a run that's
# escaped the near-duplicate-call guard below (or hit some other bug) from
# looping indefinitely and never terminating; it should essentially never
# fire in practice. Previously defaulted to 10, which was cutting off
# legitimate multi-step research runs before they reached a real answer.
_MAX_STEPS = int(os.getenv("FS_EXPLORER_MAX_STEPS", "60"))

# Cap on how many times a proposed StopAction can be rejected by the
# pre-answer completeness check (`_verify_answer_completeness`) before a
# run just accepts whatever the model proposes instead of asking it to
# keep researching. Without this, a question that is genuinely
# unanswerable from the indexed corpus could cycle "propose stop → check
# says incomplete → propose stop again" indefinitely — each cycle costs
# two extra LLM calls (the rejected action + the check itself) on top of
# the eventual _MAX_STEPS safety net, so this bounds that specific cost
# far below the full step budget rather than relying on _MAX_STEPS alone.
_MAX_STOP_REJECTIONS = int(os.getenv("FS_EXPLORER_MAX_STOP_REJECTIONS", "2"))
_FORCED_STOP_FALLBACK_ANSWER = (
    "Bu soruyu yanıtlamak için ayrılan araştırma adımı bütçesi doldu ve "
    "kesin bir cevaba ulaşılamadı. Şu ana kadar bulunanlara dayanarak "
    "elimden gelen en iyi özeti vermeye çalışacağım."
)

# Hard cap on how much raw chunk text a single "deep read" call (read,
# parse_file, get_document) can inject into chat history in one go. Real
# indexed documents are not all small: some in production run well past a
# million characters (a full treaty text with annexes, e.g.), and these
# tools previously had no cap at all — a single call on one of those could
# single-handedly inject 300K+ tokens, then get rebilled on *every*
# subsequent step since the whole history is resent each call. `grep`
# already searches a document's full, uncapped chunk set server-side and
# returns only matching excerpts, so capping this doesn't lose the ability
# to search a huge document — it only stops one call from being able to
# dump the entire thing into memory at once.
_DEEP_READ_MAX_CHARS = int(os.getenv("FS_EXPLORER_DEEP_READ_MAX_CHARS", "15000"))
_DEEP_READ_TRUNCATION_HINT = (
    "This is already the full-content tool and will not return more via a "
    "repeat call. Use grep(file_path=..., pattern=...) to search this "
    "document's complete text for specific terms, or semantic_search for "
    "a different angle."
)


def _normalize_indexed_text(value: str) -> str:
    """Decode document-export entities before they become model context."""
    return html.unescape(value)


# --- Near-duplicate tool-call guard ---
#
# Real runs have been observed re-issuing 10+ near-identical semantic_search
# queries in a row (same topic, slightly reworded each time) when the
# corpus doesn't have a clean answer — each one resent the *entire* history
# plus a full new search result, multiplying cost for zero new information.
# This tracks each tool's identifying argument (query/pattern/file_path)
# across the last few calls and short-circuits an obvious repeat with a
# tiny canned message instead of re-running the real (expensive) tool —
# cutting both the wasted DB/embedding work and, more importantly, the
# runaway step count that was the actual dominant driver of total cost.
_DUPLICATE_CALL_LOOKBACK = 5
_DUPLICATE_QUERY_SIMILARITY_THRESHOLD = 0.6
_FUZZY_DEDUP_GROUPS = {"semantic_search", "grep", "glob"}

# parse_file/get_document/read all resolve to the exact same content
# (`_document_from_chunks` on the same document) — grouped together so
# fetching the same document through a different tool *name* still counts
# as a repeat, instead of each name getting its own "first one's free".
_DEEP_READ_GROUP = "deep_read"
_TOOL_DEDUP_GROUPS: dict[str, str] = {
    "semantic_search": "semantic_search",
    "grep": "grep",
    "glob": "glob",
    "parse_file": _DEEP_READ_GROUP,
    "get_document": _DEEP_READ_GROUP,
    "read": _DEEP_READ_GROUP,
    "preview_file": "preview_file",
    "scan_folder": "scan_folder",
}


def _dedup_key(tool_name: str, tool_input: dict[str, Any]) -> tuple[str, str] | None:
    """Extract (group, identifying-argument) for a tool call, so repeats —
    including the same document fetched via a different tool name — can be
    detected regardless of other arguments."""
    group = _TOOL_DEDUP_GROUPS.get(tool_name)
    if group is None:
        return None
    if tool_name == "semantic_search":
        value = tool_input.get("query", "")
    elif tool_name in {"grep", "glob"}:
        value = tool_input.get("pattern", "")
    elif tool_name in {
        "parse_file",
        "preview_file",
        "get_document",
        "read",
    }:
        value = (
            tool_input.get("file_path")
            or tool_input.get("doc_id")
            or tool_input.get("chunk_id")
            or ""
        )
    elif tool_name == "scan_folder":
        value = tool_input.get("directory", "")
    else:
        value = ""
    value = str(value).strip()
    return (group, value) if value else None


def _query_token_overlap(a: str, b: str) -> float:
    """Jaccard-style token overlap, used only as a cheap "is this basically
    the same search again" signal — not a real semantic similarity."""
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / max(len(tokens_a), len(tokens_b))


# Tools whose rendered output actually contains indexed chunk text — the
# ones "how many chunks/tokens went to the LLM" should be counted against,
# distinct from `tool_result_chars` (which sums every tool's output,
# including non-retrieval ones like glob/scan_folder). `preview_file` is
# deliberately excluded even though it shares `_document_from_chunks` with
# `get_document`/`parse_file`/`read` — it's a fixed, cheap, truncated peek
# (`max_chars=3000` default), not representative of real retrieval breadth.
CHUNK_BEARING_TOOLS = frozenset(
    {"semantic_search", "grep", "get_document", "parse_file", "read"}
)
_SEMANTIC_SEARCH_HIT_RE = re.compile(r"^\[\d+\] doc_id: ", re.MULTILINE)
_CHUNK_HEADER_RE = re.compile(r"^--- chunk ", re.MULTILINE)


def _count_chunk_markers(tool_name: str, result: str) -> int:
    """Count chunks in a chunk-bearing tool's rendered output by counting
    the same small structured markers the rendering itself already emits
    (`"[idx] doc_id:"` for `semantic_search`, `"--- chunk "` for the
    document/context readers) — mirrors `exploration_trace.py`'s
    `extract_cited_sources()`, which already recovers structured data by
    regex-parsing already-rendered text, and `add_tool_result`'s own
    existing `result.count("│ [")` for `scan_folder` below, rather than
    plumbing a new return type through every one of the `TOOLS` functions."""
    if tool_name == "semantic_search":
        return len(_SEMANTIC_SEARCH_HIT_RE.findall(result))
    if tool_name in {"grep", "get_document", "parse_file", "read"}:
        return len(_CHUNK_HEADER_RE.findall(result))
    return 0


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

    # Retrieval-specific subset of `tool_result_chars` below — how many
    # indexed chunks were actually sent to the LLM, and how many chars of
    # that were chunk text specifically (as opposed to every tool's output
    # combined). See `CHUNK_BEARING_TOOLS`/`_count_chunk_markers` above.
    retrieval_chunks: int = 0
    retrieval_chars: int = 0

    # `prompt_tokens` above is a running SUM across every call so far — a
    # correct "tokens billed this turn" figure (every call really is
    # billed for the full history it resends), but NOT the current size of
    # the chat history, since it adds up an increasing sequence rather
    # than reporting its last value. `last_prompt_tokens` is that last
    # value, and is what context-window-fill checks must use instead.
    last_prompt_tokens: int = 0

    # Track content sizes
    tool_result_chars: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0

    # Number of times this run's mid-exploration chat history was
    # compacted by `FsExplorerAgent._maybe_summarize_history`.
    context_summaries: int = 0
    provider_billed_cost_usd: Decimal = Decimal("0")
    provider_cost_calls: int = 0

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
        self.last_prompt_tokens = prompt_tokens

    def add_tool_result(self, result: str, tool_name: str) -> tuple[int, int]:
        """Record metrics from a tool execution.

        Returns `(chunk_count, chars)` for *this call specifically* — `(0, 0)`
        for non-chunk-bearing tools — so callers wanting per-call granularity
        (e.g. a retrieval-stats hook) don't need to duplicate
        `_count_chunk_markers` themselves.
        """
        self.tool_result_chars += len(result)
        if tool_name == "parse_file":
            self.documents_parsed += 1
        elif tool_name == "scan_folder":
            # Count documents in scan result by counting document markers
            self.documents_scanned += result.count("│ [")
        elif tool_name == "preview_file":
            self.documents_parsed += 1
        if tool_name not in CHUNK_BEARING_TOOLS:
            return 0, 0
        chunk_count = _count_chunk_markers(tool_name, result)
        chars = len(result)
        self.retrieval_chunks += chunk_count
        self.retrieval_chars += chars
        return chunk_count, chars

    def retrieval_estimated_tokens(self) -> int:
        """Rough chars/4 estimate for `retrieval_chars` — same heuristic
        `_estimated_turn_tokens` already uses elsewhere in this file, so the
        two are directly comparable."""
        return (self.retrieval_chars + 3) // 4

    def _calculate_cost(self) -> tuple[float, float, float]:
        """Calculate estimated costs based on Gemini Flash pricing."""
        input_cost = (
            self.prompt_tokens / 1_000_000
        ) * GEMINI_FLASH_INPUT_COST_PER_MILLION
        output_cost = (
            self.completion_tokens / 1_000_000
        ) * GEMINI_FLASH_OUTPUT_COST_PER_MILLION
        return input_cost, output_cost, input_cost + output_cost

    def add_provider_cost(self, billed_cost_usd: Decimal | None) -> None:
        """Accumulate authoritative per-call provider billing when available."""

        if billed_cost_usd is None:
            return
        self.provider_billed_cost_usd += billed_cost_usd
        self.provider_cost_calls += 1

    def total_cost_usd(self) -> float:
        """Prefer provider billing over the legacy single-model estimate."""

        if self.provider_cost_calls:
            return float(self.provider_billed_cost_usd)
        return self._calculate_cost()[2]

    def context_usage_ratio(self, max_context_tokens: int) -> float:
        """Fraction of the model's context window the last call actually used.

        Uses `last_prompt_tokens` (the size of the most recent request), not
        the cumulative `prompt_tokens` total — the latter sums every call
        made so far, so for a multi-step run it overshoots the model's
        actual context window usage by a wide margin (e.g. a 7-step run
        whose history grows from ~20K to ~110K tokens sums to nearly
        400K, even though no single request ever exceeded ~110K).
        """
        if max_context_tokens <= 0:
            return 0.0
        return self.last_prompt_tokens / max_context_tokens

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


@dataclass(frozen=True)
class PlannedSearchResult:
    """Structured result of one stateless parallel indexed search."""

    query: str
    hits: list[SearchHit]
    error: str | None = None


# `ContextVar`, not a plain module global: `core-api` handles multiple
# chats concurrently in one process, and a plain `global` here was a real
# cross-chat data-leak bug
# — request A's `set_index_context()` and request B's `clear_index_context()`
# raced on the same variable, so a slow-running chat could have its index
# context silently wiped out (tool calls start erroring "not configured")
# or replaced by a *different* chat's corpus mid-run out from under it,
# purely based on request timing. A `ContextVar` gives each request's task
# (and the workflow's internal step tasks spawned from it, which copy the
# context at creation time) its own isolated value instead of one shared
# by the whole process. This mirrors the fix already applied to per-request
# model/temperature/hook config in `workflow.py`'s `new_workflow()` — see
# that function's docstring for the same class of bug.
_INDEX_CONTEXT_VAR: contextvars.ContextVar[IndexContext | None] = (
    contextvars.ContextVar("_INDEX_CONTEXT", default=None)
)
_EMBEDDING_PROVIDER_VAR: contextvars.ContextVar[EmbeddingProvider | None] = (
    contextvars.ContextVar("_EMBEDDING_PROVIDER", default=None)
)
_FIELD_CATALOG_SHOWN_VAR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_FIELD_CATALOG_SHOWN", default=False
)
_ENABLE_SEMANTIC_VAR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ENABLE_SEMANTIC", default=False
)
_ENABLE_METADATA_VAR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ENABLE_METADATA", default=False
)


def set_search_flags(
    *, enable_semantic: bool = False, enable_metadata: bool = False
) -> None:
    """Configure which indexed retrieval paths are active for this request."""
    _ENABLE_SEMANTIC_VAR.set(enable_semantic)
    _ENABLE_METADATA_VAR.set(enable_metadata)


def get_search_flags() -> tuple[bool, bool]:
    """Return (enable_semantic, enable_metadata) for this request."""
    return _ENABLE_SEMANTIC_VAR.get(), _ENABLE_METADATA_VAR.get()


# User-selectable per-message retrieval breadth. Unlike `limit` being a mere
# suggestion the model could previously override to any value, this is a
# hard ceiling — see `semantic_search`'s `effective_limit` below. "low" is
# deliberately tuned to match the old (pre-effort) unconditional behavior
# (`limit=5`, `overfetch_multiplier=4`) rather than cutting below it —
# retrieving fewer/shallower chunks doesn't save tokens if the model just
# compensates with extra `semantic_search`/`get_document` round trips, each
# of which resends the entire accumulated history plus (for `get_document`)
# up to 15,000 chars of chunk text. So "low" is the safe floor (never worse
# than before this feature existed) and "medium"/"high" are opt-in increases
# in depth/cost, not "low" being an opt-in decrease.
EffortLevel = Literal["low", "medium", "high"]
DEFAULT_EFFORT: EffortLevel = "low"
# Keyed by plain `str`, not `EffortLevel` — looked up with `_EFFORT_VAR.get()`,
# which is a `str` since it's fed by arbitrary client-supplied WS input
# (validated/defaulted in `set_effort`, not narrowed at the type level).
_EFFORT_LEVELS: dict[str, dict[str, int]] = {
    "low": {"default_limit": 5, "max_limit": 6, "overfetch_multiplier": 4},
    "medium": {"default_limit": 8, "max_limit": 10, "overfetch_multiplier": 5},
    "high": {"default_limit": 12, "max_limit": 20, "overfetch_multiplier": 6},
}
_EFFORT_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_EFFORT", default=DEFAULT_EFFORT
)


def set_effort(effort: str) -> None:
    """Configure retrieval breadth for this request; unknown values fall
    back to `DEFAULT_EFFORT` rather than erroring, since this is fed by
    client-supplied input."""
    _EFFORT_VAR.set(effort if effort in _EFFORT_LEVELS else DEFAULT_EFFORT)


def get_effort() -> str:
    return _EFFORT_VAR.get()


def _effort_config() -> dict[str, int]:
    return _EFFORT_LEVELS[_EFFORT_VAR.get()]


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    """Set the embedding provider for vector search in indexed tools."""
    _EMBEDDING_PROVIDER_VAR.set(provider)


def set_index_context(
    folder: str | list[str] | tuple[str, ...],
    database_url: str | None = None,
) -> None:
    """Enable indexed tools for one or more folder corpora, for this request."""
    folders = (folder,) if isinstance(folder, str) else tuple(folder)
    _INDEX_CONTEXT_VAR.set(
        IndexContext(
            root_folders=tuple(str(Path(item).resolve()) for item in folders),
            database_url=resolve_database_url(database_url),
        )
    )
    # Auto-create embedding provider if API key available
    if _EMBEDDING_PROVIDER_VAR.get() is None:
        try:
            _EMBEDDING_PROVIDER_VAR.set(EmbeddingProvider())
        except ValueError:
            pass


def clear_index_context() -> None:
    """Disable indexed tools for this request."""
    _INDEX_CONTEXT_VAR.set(None)
    _EMBEDDING_PROVIDER_VAR.set(None)
    _FIELD_CATALOG_SHOWN_VAR.set(False)
    _ENABLE_SEMANTIC_VAR.set(False)
    _ENABLE_METADATA_VAR.set(False)


def _get_index_storage_and_corpora() -> tuple[
    PostgresStorage | None, list[IndexedCorpus], str | None
]:
    index_context = _INDEX_CONTEXT_VAR.get()
    if index_context is None:
        return None, [], "Index context is not configured. Re-run with `--use-index`."

    storage = PostgresStorage(index_context.database_url)
    corpora: list[IndexedCorpus] = []
    missing: list[str] = []
    for root_folder in index_context.root_folders:
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
            f"No index found for folders: {', '.join(index_context.root_folders)}. "
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
    truncation_hint: str = (
        "Use parse_file() or get_document() for more of this document's text."
    ),
) -> str:
    chunks = storage.list_document_chunks(doc_id=str(document["id"]))
    header = (
        f"=== INDEXED DOCUMENT FROM CHUNKS ===\n"
        f"doc_id: {document['id']}\n"
        f"title: {_display_name(document)}\n"
        f"content_source: core_chunks.text\n\n"
    )
    if not chunks:
        content = _normalize_indexed_text(str(document.get("content") or ""))
        body = content if max_chars is None else content[:max_chars]
        if max_chars is not None and len(content) > max_chars:
            body += (
                f"\n\n[... TRUNCATED. Full document has {len(content):,} "
                f"characters. {truncation_hint} ...]"
            )
        return header + body

    lines: list[str] = [header]
    total = 0
    truncated = False
    for chunk in chunks:
        chunk_text = _normalize_indexed_text(str(chunk["text"]))
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
        total_chars = sum(len(str(chunk["text"])) for chunk in chunks)
        lines.append(
            f"\n\n[... TRUNCATED. Document has {len(chunks)} indexed chunks "
            f"({total_chars:,} characters total). {truncation_hint} ...]"
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
    return _INDEX_CONTEXT_VAR.get() is not None


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
    return _document_from_chunks(
        storage,
        document,
        max_chars=_DEEP_READ_MAX_CHARS,
        truncation_hint=_DEEP_READ_TRUNCATION_HINT,
    )


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
    return _document_from_chunks(
        storage,
        document,
        max_chars=_DEEP_READ_MAX_CHARS,
        truncation_hint=_DEEP_READ_TRUNCATION_HINT,
    )


def _render_full_chunk_result(
    *,
    index: int,
    doc_id: str,
    title: str,
    matched_by: str,
    chunk_id: str | None,
    chunk_position: int | None,
    chunk_type: str | None,
    metadata: dict[str, Any],
    score: float,
    text: str,
    semantic_score: float | None = None,
    metadata_score: int | None = None,
) -> list[str]:
    """Render one selected retrieval hit with its complete chunk text."""
    locator = _chunk_locator({"metadata": metadata})
    position = chunk_position if chunk_position is not None else "<unknown>"
    lines = [
        f"[{index}] doc_id: {doc_id}",
        f"    title: {title}",
        f"    match: {matched_by}",
        f"    chunk_id: {chunk_id or '<unknown>'}",
        f"    chunk_path: {title}{locator}",
        f"    chunk_position: {position}",
    ]
    if semantic_score is not None:
        lines.append(f"    semantic_score: {semantic_score}")
    if metadata_score is not None:
        lines.append(f"    metadata_score: {metadata_score}")
    lines.extend(
        [
            f"    score: {score:.2f}",
            "",
            f"--- chunk {position} ({chunk_type or 'text'}){locator} ---",
            _normalize_indexed_text(text).strip(),
            "",
        ]
    )
    return lines


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

    # Grep is a chunk retriever, not merely a matched-substring renderer:
    # send every matching whole chunk to the same hosted cross-encoder used
    # by semantic_search, and deliver every selected chunk to the LLM
    # without truncation.
    candidates: list[RankedDocument] = []
    matched_documents: set[str] = set()
    for document in documents:
        for chunk in storage.list_document_chunks(doc_id=str(document["id"])):
            chunk_text = _normalize_indexed_text(str(chunk["text"]))
            matches = regex.findall(chunk_text)
            if not matches:
                continue
            matched_documents.add(str(document["id"]))
            raw_metadata = chunk.get("metadata")
            chunk_metadata: dict[str, Any] = (
                dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            )
            candidates.append(
                RankedDocument(
                    doc_id=str(document["id"]),
                    relative_path=str(document["relative_path"]),
                    absolute_path=str(document["absolute_path"]),
                    position=int(chunk["position"]),
                    text=chunk_text,
                    semantic_score=float(len(matches)),
                    metadata_score=0,
                    chunk_id=str(chunk["id"]),
                    chunk_type=str(chunk.get("chunk_type") or "text"),
                    metadata=chunk_metadata,
                )
            )

    if not candidates:
        return "No matches found in indexed chunks"

    candidates.sort(
        key=lambda candidate: (
            -candidate.semantic_score,
            candidate.relative_path,
            candidate.position or 0,
        )
    )
    ranked = IndexedQueryEngine.rank_candidates(
        query=pattern,
        documents=candidates,
        limit=_effort_config()["default_limit"],
    )

    lines = [
        "=== INDEXED GREP RESULTS (FULL CHUNKS) ===",
        f"Pattern: {pattern}",
        f"Searched: {file_path}",
        f"Matched documents: {len(matched_documents)}",
        "",
    ]
    for idx, (candidate, score) in enumerate(ranked, start=1):
        lines.extend(
            _render_full_chunk_result(
                index=idx,
                doc_id=candidate.doc_id,
                title=_display_name(
                    {
                        "relative_path": candidate.relative_path,
                        "absolute_path": candidate.absolute_path,
                    }
                ),
                matched_by="grep",
                chunk_id=candidate.chunk_id,
                chunk_position=candidate.position,
                chunk_type=candidate.chunk_type,
                metadata=candidate.metadata,
                score=score,
                text=candidate.text,
            )
        )
    lines.append(
        "The selected grep hits above already include complete chunk text. "
        "Use the evidence directly; do not issue a deep-read call merely to reread it."
    )
    return "\n".join(lines)


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


def _run_planned_index_search(
    *,
    query: str,
    filters: str | None = None,
    as_of_date: str | None = None,
) -> PlannedSearchResult:
    """Run one independent search and return structured hits, not rendered text."""
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return PlannedSearchResult(query=query, hits=[], error=error)
    assert storage is not None and corpora

    effort_config = _effort_config()
    engine = IndexedQueryEngine(
        storage, embedding_provider=_EMBEDDING_PROVIDER_VAR.get()
    )
    hits: list[SearchHit] = []
    try:
        for corpus in corpora:
            try:
                corpus_hits = engine.search(
                    corpus_id=corpus.corpus_id,
                    query=query,
                    filters=filters,
                    limit=effort_config["default_limit"],
                    enable_semantic=_ENABLE_SEMANTIC_VAR.get(),
                    enable_metadata=_ENABLE_METADATA_VAR.get(),
                    as_of_date=as_of_date,
                    overfetch_multiplier=effort_config["overfetch_multiplier"],
                )
            except MetadataFilterParseError:
                # Corpus-specific unknown fields can only be detected here.
                # A metadata filter is an optional narrowing hint, so retry
                # the semantic query without it instead of returning zero
                # evidence and triggering a futile agent retry loop.
                corpus_hits = engine.search(
                    corpus_id=corpus.corpus_id,
                    query=query,
                    filters=None,
                    limit=effort_config["default_limit"],
                    enable_semantic=_ENABLE_SEMANTIC_VAR.get(),
                    enable_metadata=False,
                    as_of_date=as_of_date,
                    overfetch_multiplier=effort_config["overfetch_multiplier"],
                )
            hits.extend(corpus_hits)
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return PlannedSearchResult(
            query=query,
            hits=hits[: effort_config["default_limit"]],
        )
    except (MetadataFilterParseError, ValueError) as exc:
        return PlannedSearchResult(query=query, hits=[], error=str(exc))
    finally:
        storage.close()


def semantic_search(
    query: str,
    filters: str | None = None,
    limit: int | None = None,
    as_of_date: str | None = None,
) -> str:
    """Search indexed chunks, rerank them, and return full selected chunks.

    `as_of_date` (YYYY-MM-DD) restricts results to chunks whose validity
    interval covers that date — pass it whenever the user's question refers
    to a specific date/time period. Omit it (defaults to today) for
    "what's the current rule" questions.

    `limit` is capped by the user's selected effort level rather than honored
    verbatim — the effort level is what actually controls retrieval breadth.
    """
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return error
    assert storage is not None and corpora

    effort_config = _effort_config()
    effective_limit = (
        min(limit, effort_config["max_limit"])
        if limit
        else effort_config["default_limit"]
    )

    engine = IndexedQueryEngine(
        storage, embedding_provider=_EMBEDDING_PROVIDER_VAR.get()
    )
    hits: list[Any] = []
    try:
        for corpus in corpora:
            hits.extend(
                engine.search(
                    corpus_id=corpus.corpus_id,
                    query=query,
                    filters=filters,
                    limit=effective_limit,
                    enable_semantic=_ENABLE_SEMANTIC_VAR.get(),
                    enable_metadata=_ENABLE_METADATA_VAR.get(),
                    as_of_date=as_of_date,
                    overfetch_multiplier=effort_config["overfetch_multiplier"],
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        hits = hits[:effective_limit]
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
        title = _display_name(
            {"relative_path": hit.relative_path, "absolute_path": hit.absolute_path}
        )
        lines.extend(
            _render_full_chunk_result(
                index=idx,
                doc_id=hit.doc_id,
                title=title,
                matched_by=hit.matched_by,
                chunk_id=hit.chunk_id,
                chunk_position=hit.position,
                chunk_type=hit.chunk_type,
                metadata=hit.metadata,
                score=hit.score,
                text=hit.text,
                semantic_score=hit.semantic_score,
                metadata_score=hit.metadata_score,
            )
        )
    lines.append(
        "Every selected result above already includes its complete chunk text. "
        "Use that evidence directly; do not call get_document merely to reread "
        "a returned hit. Deep-read a document only for a specific unresolved "
        "cross-reference or document-wide question."
    )

    # Include a rich field catalog on the first search so the agent can
    # construct effective metadata filters.
    if not _FIELD_CATALOG_SHOWN_VAR.get():
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
                _FIELD_CATALOG_SHOWN_VAR.set(True)
                break

    return "\n".join(lines)


def get_document(doc_id: str) -> str:
    """Return document content by id from the active index context."""
    storage, corpora, error = _get_index_storage_and_corpora()
    if error:
        return error
    assert storage is not None

    document = _resolve_index_document(storage, corpora, doc_id)
    if document is None:
        return f"No indexed document found for doc_id={doc_id!r}"
    if document["is_deleted"]:
        return f"Document {doc_id} is marked as deleted in the index."

    return _document_from_chunks(
        storage,
        document,
        max_chars=_DEEP_READ_MAX_CHARS,
        truncation_hint=_DEEP_READ_TRUNCATION_HINT,
    )


def _chunk_context_from_storage(
    storage: PostgresStorage,
    chunk_id: str,
    *,
    before: int = 1,
    after: int = 1,
) -> str:
    target = storage.get_chunk(chunk_id=chunk_id)
    if target is None:
        return f"No indexed chunk found for chunk_id={chunk_id!r}"

    doc_id = str(target.get("document_id") or target.get("doc_id") or "")
    chunks = storage.list_document_chunks(doc_id=doc_id)
    target_index = next(
        (index for index, chunk in enumerate(chunks) if str(chunk["id"]) == chunk_id),
        None,
    )
    if target_index is None:
        return f"No indexed chunk found for chunk_id={chunk_id!r}"

    bounded_before = min(max(int(before), 0), 3)
    bounded_after = min(max(int(after), 0), 3)
    selected = chunks[
        max(0, target_index - bounded_before) : target_index + bounded_after + 1
    ]
    lines = [
        f"=== RELEVANT CHUNK CONTEXT ===\nchunk_id: {chunk_id}\ndoc_id: {doc_id}\n"
    ]
    for chunk in selected:
        lines.append(
            f"\n--- chunk {chunk['position']}{_chunk_locator(chunk)} ---\n"
            f"{_normalize_indexed_text(str(chunk['text'])).strip()}\n"
        )
    return "".join(lines)


def get_chunk_context(chunk_id: str, before: int = 1, after: int = 1) -> str:
    """Return a semantic hit plus a small bounded neighboring chunk window."""
    storage, _corpora, error = _get_index_storage_and_corpora()
    if error:
        return error
    assert storage is not None
    return _chunk_context_from_storage(storage, chunk_id, before=before, after=after)


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
        "Use semantic_search(...) to retrieve relevant full chunks. Deep-read a "
        "document only for a specific unresolved cross-reference or document-wide question."
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

_DETAILED_SYSTEM_PROMPT = """
You are FsExplorer, an AI agent that answers questions about indexed documents.

## Output Encoding

Write answers as plain Unicode UTF-8 text. Never emit HTML/XML entities; always
write actual Unicode characters, even if an indexed source contains entities.
Do not use HTML markup for answer formatting.

## Available Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `scan_folder` | Scan indexed documents and previews from `core_chunks` | `directory` |
| `preview_file` | Quick preview of one indexed document from chunk text | `file_path` |
| `parse_file` | **DEEP READ** - full indexed chunk text for a document | `file_path` |
| `read` | Read indexed chunk text for a document | `file_path` |
| `grep` | Search and rerank regex-matching full chunks | `file_path`, `pattern` |
| `glob` | Find indexed document paths matching a pattern | `directory`, `pattern` |
| `semantic_search` | Search/rerank indexed candidates and return complete selected chunks | `query`, `filters`, `limit`, `as_of_date` |
| `get_document` | Read full indexed document by document id | `doc_id` |
| `list_indexed_documents` | List indexed documents for active corpus | none |

## Indexed Retrieval Strategy

When indexed tools are available:
1. Start with `semantic_search`; every selected result already contains its complete chunk text.
2. Use those chunks directly. Do not call `get_document`, `parse_file`, or `read` merely to reread a returned hit.
3. Treat `parse_file`, `preview_file`, `read`, `grep`, `glob`, and `scan_folder` as chunk-backed tools. They read `core_chunks.text`/`core_documents`, not raw upload files.
4. Deep-read only for a specific unresolved cross-reference or a genuinely document-wide question.
5. If indexed tools report index is unavailable, only then fall back to filesystem tools.

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

### PHASE 2: Evidence Check (Usually no second read)
1. Read the complete selected chunks already returned by `semantic_search` or `grep`
2. Use `get_document`/`parse_file` only when a specific unresolved cross-reference or document-wide structure requires text outside those chunks
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

SYSTEM_PROMPT = """You are an indexed-document research planner.

Choose the next action using the response schema. Keep `reason` concise but
record concrete findings and document/article locators so later synthesis can
cite them accurately.

Tools: semantic_search and grep return complete reranked chunks; get_document
reads a selected full document only for unresolved cross-references or genuinely
document-wide questions; scan_folder lists the corpus;
preview_file/parse_file/read inspect indexed text; glob lists matching paths;
list_indexed_documents lists document IDs.

Strategy:
1. Start with semantic_search. Use `as_of_date=YYYY-MM-DD` when the question
   names a historical date; otherwise omit it for current rules.
   Put two or three independent searches/reads in one ToolBatchAction.
2. Search results already contain complete selected chunks. Use them directly;
   never call get_document/parse_file/read merely to reread a returned hit.
   Deep-read only when evidence outside those chunks is specifically required.
3. Use a genuinely different query/tool after a miss; do not repeat searches.
4. Stop as soon as the gathered evidence directly answers the question. Do not
   run additional searches or reads purely to re-confirm or gather more
   support for an answer you already have — only continue if you found a
   genuine gap, contradiction, or an unresolved cross-reference, not to
   double-check something already well-supported. Retain readable document
   titles and article/section locators for citations before stopping; the
   final synthesis step (not this one) is where the answer gets written out
   in full, so stopping early here does not mean a shorter final answer.
5. Never expose internal paths or storage identifiers in the final answer.
"""

FINAL_SYSTEM_PROMPT = """Write the final answer from the gathered evidence.

Start with a direct answer. Cite every factual claim inline as
`[Readable Document Title, Article/Section]`. Never cite local paths, storage
paths, raw filenames, or `Source:` labels. End with `## Sources` containing
readable document titles only. Use plain Unicode UTF-8 text, never HTML/XML
entities or HTML markup. Do not invent facts or citations absent from the
evidence.
"""

COMPLETENESS_CHECK_PROMPT = """Review the exploration transcript above, which
ends with a proposed STOP action and its final_result. Do not answer the
user's question yourself — only judge whether the proposed final_result is
actually ready to send.

Set `can_answer_fully=true` only when the main rule and every material part
of the original question are supported by evidence already gathered in the
transcript, with no unresolved gap. Set it to `false` when any of these hold:
- a cross-reference the transcript itself flagged (e.g. "see Exhibit B",
  "refer to Genelge X") was never actually followed up and read,
- an exception, condition, procedural detail, or date qualifier relevant to
  the question was not checked,
- the evidence conflicts or is ambiguous and that was never reconciled,
- the proposed final_result would have to guess, generalize, or state
  something not actually present in the retrieved evidence.

When `can_answer_fully` is false, name the single most important missing
piece of evidence in `missing_information` so the next research step has a
concrete target. Do not flag a stop as incomplete merely to double-check an
already well-supported point.
"""


def _build_system_prompt(enable_semantic: bool, enable_metadata: bool) -> str:
    """Build a system prompt with retrieval-path guidance appended."""
    if enable_semantic and enable_metadata:
        hint = (
            "\n\n## Retrieval: Semantic + Metadata\n"
            "An index is available. Start with `semantic_search` using optional "
            "`filters`; selected results already include complete reranked chunks."
        )
    elif enable_semantic:
        hint = (
            "\n\n## Retrieval: Semantic Only\n"
            "An index is available. Use `semantic_search` WITHOUT the `filters` "
            "parameter; selected results already include complete reranked chunks."
        )
    elif enable_metadata:
        hint = (
            "\n\n## Retrieval: Metadata Only\n"
            "An index is available. Use `semantic_search` with the `filters=` "
            "parameter for metadata filtering; matching documents are hydrated "
            "and reranked as complete chunks."
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
    per provider call as it happens.
    """

    purpose: str  # "action" (tool-planning step) | "final_answer"
    model: str
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    duration_ms: float
    step: int = 0
    provider: str = "gemini"
    generation_id: str | None = None
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    billed_cost_usd: str | None = None
    upstream_cost_usd: str | None = None
    cost_source: str | None = None
    agent_role: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    sequence: int | None = None


OnLLMCall = Callable[[LLMCallStats], Awaitable[None]]


@dataclass
class RetrievalStats:
    """Per-call chunk/token observation, for external instrumentation —
    the retrieval-specific analog of `LLMCallStats` above, firing once per
    chunk-bearing tool call (see `CHUNK_BEARING_TOOLS`) rather than once
    per LLM call. Plain sync callback, deliberately not `Awaitable` like
    `OnLLMCall` — `call_tool()` is a sync method; a sync callback lets both
    it and the async `call_tools()` batch path invoke it directly without
    making `call_tool` itself `async def`, which would ripple into its
    `workflow.py` call sites.
    """

    step: int
    tool_name: str
    chunk_count: int
    chars: int
    estimated_tokens: int
    task_id: str | None = None
    agent_id: str | None = None
    sequence: int | None = None


OnRetrieval = Callable[[RetrievalStats], None]


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
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        on_llm_call: OnLLMCall | None = None,
        on_retrieval: OnRetrieval | None = None,
        llm_profile: LLMProfile | None = None,
        role_clients: Mapping[LLMRole, LLMClient] | None = None,
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
                         individual provider call with per-call token/timing
                         stats (see `LLMCallStats`), for callers that want
                         incremental observability instead of only the
                         cumulative `token_usage` totals.
            on_retrieval: Optional sync callback invoked after every
                          chunk-bearing tool call with per-call chunk/token
                          stats (see `RetrievalStats`), for callers that
                          want incremental "how many chunks/tokens went to
                          the LLM" observability instead of only the
                          cumulative `token_usage.retrieval_*` totals.

        Raises:
            ValueError: If no Google credentials are available and no
                        llm_client given.
        """
        self._llm = llm_client or get_llm_client(
            provider=provider, model=model, temperature=temperature, api_key=api_key
        )
        self._chat_history: list[ChatTurn] = []
        self.token_usage = TokenUsage()
        self._on_llm_call = on_llm_call
        self._on_retrieval = on_retrieval
        self._llm_profile = llm_profile
        self._role_clients: dict[LLMRole, LLMClient] = dict(role_clients or {})
        self._multi_agent_orchestrator: MultiAgentResearchOrchestrator | None = None
        self._multi_agent_result: MultiAgentResearchResult | None = None
        self._multi_agent_final_llm: LLMClient | None = None
        self._multi_agent_research_active = False
        # Final synthesis is a per-run single-flight operation.  The provider
        # stream is drained by a background task and its chunks are retained on
        # this agent, so a dropped WebSocket can replay the same answer instead
        # of paying for another final-model call.  The agent itself is owned by
        # the run registry, whose bounded TTL controls the cache lifetime.
        self._final_answer_task: asyncio.Task[None] | None = None
        self._final_answer_chunks: list[str] = []
        self._final_answer_complete = False
        self._final_answer_error: Exception | None = None
        self._final_answer_condition = asyncio.Condition()
        self._step_count = 0
        # Instance-level, not just the module constant — grant_more_steps()
        # (called on resume) raises this so a resumed run gets genuine extra
        # room instead of immediately re-hitting the same lifetime ceiling.
        self._max_steps = _MAX_STEPS
        # (tool_name, dedup_key) for the last `_DUPLICATE_CALL_LOOKBACK`
        # real tool calls — see `_dedup_key`/`_is_near_duplicate_call`.
        self._recent_tool_calls: list[tuple[str, str]] = []
        self._prepared_indexed_evidence = False
        # Set when take_action() hits the _MAX_STEPS safety net rather than
        # the model choosing to stop on its own — the run technically
        # "completed" (no error), but the answer may be an unsatisfying
        # apology rather than a real conclusion. Callers (server.py) use
        # this to flag the run as resumable even without an error/cancel.
        self._forced_stop = False
        # How many proposed StopActions the pre-answer completeness check
        # has rejected so far this run — see _verify_answer_completeness
        # and _MAX_STOP_REJECTIONS.
        self._stop_rejections = 0

    @property
    def step_count(self) -> int:
        """Decision-points (`take_action()` calls) made so far this run."""
        return self._step_count

    @property
    def forced_stop(self) -> bool:
        """Whether the run ended by hitting _MAX_STEPS, not a real StopAction."""
        return self._forced_stop

    @property
    def prepared_indexed_evidence(self) -> bool:
        """Whether the multi-agent research result is ready for final synthesis."""
        return self._prepared_indexed_evidence

    @property
    def multi_agent_research_active(self) -> bool:
        """Whether a resumable multi-agent research plan is in progress."""

        return self._multi_agent_research_active

    @property
    def final_answer_started(self) -> bool:
        """Whether this run has entered its single-flight final synthesis."""

        return self._final_answer_task is not None

    @property
    def multi_agent_incomplete(self) -> bool:
        """Whether a required task remained partial or failed."""

        return bool(self._multi_agent_result and self._multi_agent_result.incomplete)

    @property
    def multi_agent_unresolved_information(self) -> list[str]:
        """Return bounded, user-readable evidence gaps and material unknowns."""

        result = self._multi_agent_result
        if result is None:
            return []
        return list(getattr(result, "unresolved_information", ()))

    @property
    def benchmark_plan_trace(self) -> dict[str, object] | None:
        """Return a bounded JSON-safe orchestration trace for offline scoring."""

        result = self._multi_agent_result
        if result is None:
            return None
        task_artifacts = [
            {
                "task_id": artifact.task_id,
                "status": artifact.status.value,
                "covered_requirement_ids": list(artifact.covered_requirement_ids),
                "uncovered_requirement_ids": list(artifact.uncovered_requirement_ids),
                "claim_ids": [claim.claim_id for claim in artifact.claims],
                "application_findings": [
                    {
                        "conclusion_id": finding.conclusion_id,
                        "requirement_ids": list(finding.requirement_ids),
                        "fact_ids": list(finding.fact_ids),
                        "branch_ids": list(finding.branch_ids),
                        "supporting_claim_ids": list(finding.supporting_claim_ids),
                        "dependency_refs": [
                            reference.model_dump(mode="json")
                            for reference in finding.dependency_refs
                        ],
                        "confidence": finding.confidence.value,
                        "limitations": list(finding.limitations),
                    }
                    for finding in artifact.application_findings
                ],
                "gaps": list(artifact.gaps),
            }
            for artifact in result.task_artifacts
        ]
        return {
            "schema_version": 1,
            "contract_version": result.plan.version,
            "plan": result.plan.model_dump(mode="json"),
            "task_artifacts": task_artifacts,
            "incomplete": result.incomplete,
        }

    @property
    def final_model(self) -> str | None:
        """Model that will write the user-visible answer."""

        client = (
            self._multi_agent_final_llm
            if self._multi_agent_result is not None
            else self._llm
        )
        return getattr(client, "model", None)

    def evidence_sources_for_answer(self, final_answer: str) -> list[dict[str, object]]:
        """Return only chunks cited by their exact readable title and locator."""

        if self._multi_agent_result is None:
            return []
        normalized_answer = " ".join(html.unescape(final_answer).casefold().split())
        selected: list[dict[str, object]] = []
        seen: set[str] = set()
        for source in self._multi_agent_result.evidence_sources:
            title = str(source.get("title") or "").strip()
            locator = str(source.get("locator") or "").strip()
            chunk_id = str(source.get("chunk_id") or "").strip()
            citation = " ".join(
                html.unescape(f"[{title}, {locator}]").casefold().split()
            )
            if (
                not title
                or not locator
                or citation not in normalized_answer
                or not chunk_id
                or chunk_id in seen
            ):
                continue
            seen.add(chunk_id)
            selected.append(dict(source))
        return selected

    async def _report_llm_call(
        self,
        purpose: str,
        usage: "LLMUsage",
        *,
        client: LLMClient | None = None,
        agent_role: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        sequence: int | None = None,
    ) -> None:
        self.token_usage.add_provider_cost(usage.billed_cost_usd)
        if self._on_llm_call is None:
            return
        resolved_client = client or self._llm
        provider = getattr(resolved_client, "provider", None)
        if not isinstance(provider, str):
            provider = (
                "openrouter"
                if resolved_client.__class__.__name__ == "OpenRouterLLMClient"
                else "gemini"
            )
        await self._on_llm_call(
            LLMCallStats(
                purpose=purpose,
                model=getattr(resolved_client, "model", "unknown"),
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                thinking_tokens=usage.thinking_tokens,
                duration_ms=usage.duration_ms,
                step=sequence if sequence is not None else self._step_count,
                provider=provider,
                generation_id=usage.generation_id,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                billed_cost_usd=(
                    str(usage.billed_cost_usd)
                    if usage.billed_cost_usd is not None
                    else None
                ),
                upstream_cost_usd=(
                    str(usage.upstream_cost_usd)
                    if usage.upstream_cost_usd is not None
                    else None
                ),
                cost_source=usage.cost_source,
                agent_role=agent_role,
                task_id=task_id,
                agent_id=agent_id,
                sequence=sequence,
            )
        )

    def _report_retrieval(self, tool_name: str, chunk_count: int, chars: int) -> None:
        # Fires for every chunk-bearing call, even a genuine chunk_count=0
        # (e.g. a get_document on a doc_id with no indexed chunks) — not
        # just when chunks were actually found. Ordinary tool execution emits
        # one event per chunk-bearing call. Adaptive stateless retrieval calls
        # this once per parallel round with the bounded digest/full evidence
        # actually exposed to its coordinator/final synthesis.
        if self._on_retrieval is None or tool_name not in CHUNK_BEARING_TOOLS:
            return
        self._on_retrieval(
            RetrievalStats(
                step=self._step_count,
                tool_name=tool_name,
                chunk_count=chunk_count,
                chars=chars,
                estimated_tokens=(chars + 3) // 4,
            )
        )

    def _ensure_multi_agent_clients(self) -> dict[LLMRole, LLMClient]:
        """Create one immutable role-client set lazily for this run."""

        profile = self._llm_profile or load_llm_profile()
        for role in ("planner", "task", "worker", "final"):
            if role not in self._role_clients:
                self._role_clients[role] = get_llm_client(
                    role=role,
                    profile=profile,
                )
        return self._role_clients

    async def _record_multi_agent_llm_usage(
        self,
        role: str,
        purpose: str,
        client: LLMClient,
        usage: LLMUsage,
        task_id: str | None,
        agent_id: str | None,
        sequence: int,
    ) -> None:
        """Fold one isolated role call into legacy aggregate telemetry."""

        self._step_count += 1
        self.token_usage.add_api_call(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
        )
        await self._report_llm_call(
            purpose,
            usage,
            client=client,
            agent_role=role,
            task_id=task_id,
            agent_id=agent_id,
            sequence=sequence,
        )

    def _record_multi_agent_retrieval(
        self,
        chunk_count: int,
        chars: int,
        task_id: str | None,
        agent_id: str | None,
        sequence: int,
    ) -> None:
        """Record the bounded evidence actually exposed to one worker."""

        self.token_usage.tool_result_chars += chars
        self.token_usage.retrieval_chunks += chunk_count
        self.token_usage.retrieval_chars += chars
        if self._on_retrieval is not None:
            self._on_retrieval(
                RetrievalStats(
                    step=sequence,
                    tool_name="semantic_search",
                    chunk_count=chunk_count,
                    chars=chars,
                    estimated_tokens=(chars + 3) // 4,
                    task_id=task_id,
                    agent_id=agent_id,
                    sequence=sequence,
                )
            )

    async def prepare_multi_agent_indexed(
        self,
        task: str,
        *,
        on_progress: Callable[[ResearchProgress], None] | None = None,
    ) -> MultiAgentResearchResult:
        """Run or resume the context-isolated multi-agent indexed workflow."""

        if self._multi_agent_result is not None:
            return self._multi_agent_result

        clients = self._ensure_multi_agent_clients()
        self._multi_agent_final_llm = clients["final"]
        if self._multi_agent_orchestrator is None:
            self._multi_agent_orchestrator = MultiAgentResearchOrchestrator(
                planner_llm=clients["planner"],
                task_llm=clients["task"],
                worker_llm=clients["worker"],
                search_runner=_run_planned_index_search,
                on_llm_usage=self._record_multi_agent_llm_usage,
                on_retrieval=self._record_multi_agent_retrieval,
                on_progress=on_progress,
            )
        else:
            self._multi_agent_orchestrator.set_progress_observer(on_progress)

        self._multi_agent_research_active = True
        try:
            result = await self._multi_agent_orchestrator.run(task)
        except BaseException:
            # Keep the orchestrator and completed artifacts on the agent so an
            # interrupted WebSocket run can continue without replaying them.
            raise
        else:
            self._multi_agent_result = result
            self._multi_agent_research_active = False
            self._prepared_indexed_evidence = True
            self._chat_history = [
                ChatTurn(role="user", text=result.final_context),
            ]
            return result

    def grant_more_steps(self, extra: int | None = None) -> None:
        """Extend this agent's step budget — call when resuming a run.

        `_max_steps` is a lifetime ceiling on `_step_count`, which keeps
        growing across a resume (it's the same agent instance). Without
        this, resuming a run that hit the step budget would immediately
        re-trigger the exact same forced stop on its very next
        `take_action()` call, making "Continue" a no-op. Also clears
        `forced_stop`, since the run is no longer sitting at that state.
        """
        self._max_steps += extra if extra is not None else _MAX_STEPS
        self._forced_stop = False

    def configure_task(self, task: str) -> None:
        """
        Add a task message to the conversation history.

        Args:
            task: The task or context to add to the conversation.
        """
        self._chat_history.append(ChatTurn(role="user", text=task))

    def set_llm_call_hook(self, on_llm_call: OnLLMCall | None) -> None:
        """Rebind the per-call observability hook to a new connection.

        Used when resuming an interrupted run: the agent instance is reused
        as-is (its `_chat_history`/`_step_count` are exactly what makes
        resuming meaningful), but its original `on_llm_call` closure was
        bound to the WebSocket connection that got interrupted — calling it
        after that connection is gone would send into a dead socket. The
        resumed connection rebinds this to its own closure before driving
        the agent any further.
        """
        self._on_llm_call = on_llm_call

    def set_retrieval_hook(self, on_retrieval: OnRetrieval | None) -> None:
        """Rebind the per-call retrieval-stats hook to a new connection.

        Same reason as `set_llm_call_hook()` — a resumed run reuses this
        agent instance as-is, but the original hook closure was bound to
        the interrupted connection.
        """
        self._on_retrieval = on_retrieval

    async def take_action(self) -> tuple[Action, ActionType] | None:
        """
        Request the next action from the LLM.

        Sends the current conversation history and receives a structured
        JSON response indicating the next action to take. When that action
        is a `StopAction`, it is not handed back as-is: `_verify_answer_completeness`
        runs one extra, cheap check asking whether the gathered evidence
        actually supports a complete answer. If the check says no, the
        proposed stop is rejected — a note describing the gap is appended to
        history and this loops back for another real action instead of
        letting an under-researched answer reach final synthesis. Bounded by
        `_MAX_STOP_REJECTIONS` so a question that turns out to be genuinely
        unanswerable from the corpus still terminates rather than looping
        forever.

        Returns:
            A tuple of (Action, ActionType) if successful, None otherwise.
        """
        while True:
            self._step_count += 1
            if self._step_count > self._max_steps:
                # Force a stop without spending another LLM call on it — a run
                # that's hit the step budget doesn't need the model's opinion on
                # whether to keep going, it needs to wrap up. `final_result`
                # normally only has to satisfy the Action schema, since
                # `stream_final_answer()` (called next, with the full
                # accumulated history) is what actually composes the real
                # answer — but it's also used verbatim as that call's fallback
                # if streaming itself fails, so it must never be blank: an
                # empty string there plus a failed stream previously meant the
                # user got a silently empty response.
                self._forced_stop = True
                action = Action(
                    action=StopAction(final_result=_FORCED_STOP_FALLBACK_ANSWER),
                    reason=(
                        f"Reached the step budget ({self._max_steps} actions) without "
                        "a conclusive path through further tool calls — "
                        "stopping to compose the best answer from what has "
                        "been gathered so far."
                    ),
                )
                self._chat_history.append(
                    ChatTurn(role="model", text=action.model_dump_json())
                )
                return action, action.to_action_type()

            action, usage = await self._llm.generate_structured(
                self._chat_history,
                _build_system_prompt(*get_search_flags()),
                Action,
                thinking_level="low",
            )
            self.token_usage.add_api_call(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                thinking_tokens=usage.thinking_tokens,
            )
            await self._report_llm_call("action", usage)
            self._chat_history.append(
                ChatTurn(role="model", text=action.model_dump_json())
            )
            await self._maybe_summarize_history(usage.input_tokens)

            action_type = action.to_action_type()
            if action_type != "stop" or self._stop_rejections >= _MAX_STOP_REJECTIONS:
                return action, action_type

            check = await self._verify_answer_completeness()
            if check is None or check.can_answer_fully:
                return action, action_type

            self._stop_rejections += 1
            self._chat_history.append(
                ChatTurn(
                    role="user",
                    text=(
                        "Your proposed stop was rejected by a completeness "
                        f"check: {check.missing_information}\n\n"
                        "Do not answer yet — take another action to resolve "
                        "this gap, then stop again once it is addressed."
                    ),
                )
            )

    async def _verify_answer_completeness(self) -> CompletenessCheck | None:
        """Ask one extra, cheap follow-up call whether a just-proposed
        StopAction is actually ready to answer, before treating it as terminal.

        Best-effort: any failure here falls open (returns `None`, which the
        caller treats the same as "accept the stop") rather than blocking a
        run on a secondary check that isn't itself essential to producing an
        answer.
        """
        self._step_count += 1
        try:
            check, usage = await self._llm.generate_structured(
                self._chat_history,
                COMPLETENESS_CHECK_PROMPT,
                CompletenessCheck,
                thinking_level="low",
            )
        except Exception:
            logger.exception(
                "_verify_answer_completeness: completeness check failed; "
                "accepting the proposed stop as-is"
            )
            return None
        self.token_usage.add_api_call(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
        )
        await self._report_llm_call("completeness_check", usage)
        return check

    async def _maybe_summarize_history(self, last_prompt_tokens: int) -> None:
        """
        Compact the chat history once it nears the model's context window.

        `last_prompt_tokens` is the size (in tokens) of the request that was
        just sent — since every call resends the full history, that's also
        a good proxy for "how big is `_chat_history` right now". Once that
        crosses `CONTEXT_SUMMARY_MAX_TOKENS` (the cost-driven trigger) or
        `CONTEXT_SUMMARY_THRESHOLD_RATIO` of the model's ceiling (the
        crash-prevention backstop) — whichever comes first — the middle of
        the history (everything except the original task framing and the
        most recent turns) is replaced with a single compact summary turn,
        generated by one extra LLM call over just that middle slice.
        """
        ratio_trigger = (
            last_prompt_tokens / GEMINI_MAX_CONTEXT_TOKENS
            >= CONTEXT_SUMMARY_THRESHOLD_RATIO
        )
        absolute_trigger = last_prompt_tokens >= CONTEXT_SUMMARY_MAX_TOKENS
        if not (ratio_trigger or absolute_trigger):
            return

        leading = self._chat_history[:_CONTEXT_SUMMARY_KEEP_LEADING_TURNS]
        recent = _token_budgeted_recent_turns(self._chat_history)
        middle_end = len(self._chat_history) - len(recent)
        middle = self._chat_history[_CONTEXT_SUMMARY_KEEP_LEADING_TURNS:middle_end]
        if not middle:
            return
        if len(middle) == 1 and middle[0].text.startswith(
            "Summary of earlier exploration steps"
        ):
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
                thinking_level="low",
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

    def start_final_answer(self, fallback_answer: str | None = None) -> None:
        """Start final synthesis once, independently of its WebSocket consumer.

        Keeping the producer in its own task is intentional: cancellation of a
        disconnected response iterator must not cancel the paid provider call.
        A resumed connection consumes the retained chunks from the beginning
        and then follows the same in-flight task to completion.
        """
        if self._final_answer_task is None:
            self._final_answer_task = asyncio.create_task(
                self._run_final_answer_producer(fallback_answer),
                name="fs-explorer-final-answer",
            )

    def clear_final_answer(self) -> None:
        """Discard final replay state before deliberately continuing a run."""
        if self._final_answer_task is not None and not self._final_answer_task.done():
            self._final_answer_task.cancel()
        self._final_answer_task = None
        self._final_answer_chunks.clear()
        self._final_answer_complete = False
        self._final_answer_error = None

    async def _publish_final_chunk(self, text: str) -> None:
        async with self._final_answer_condition:
            self._final_answer_chunks.append(text)
            self._final_answer_condition.notify_all()

    async def _run_final_answer_producer(self, fallback_answer: str | None) -> None:
        """Always release replay consumers and preserve provider failures."""
        try:
            await self._produce_final_answer(fallback_answer)
        except Exception as exc:
            logger.exception("stream_final_answer: final-answer producer failed")
            self._final_answer_error = exc
        finally:
            # `reset()` can cancel an old producer and immediately clear the
            # state for reuse.  Do not let that old task mark the new state as
            # complete when its cancellation is delivered on the next loop.
            if asyncio.current_task() is self._final_answer_task:
                async with self._final_answer_condition:
                    self._final_answer_complete = True
                    self._final_answer_condition.notify_all()

    async def _produce_final_answer(self, fallback_answer: str | None) -> None:
        """Drain exactly one final-model stream into the replayable run cache."""
        is_multi_agent = (
            self._multi_agent_result is not None
            and self._multi_agent_final_llm is not None
        )
        is_scenario = bool(
            is_multi_agent
            and self._multi_agent_result is not None
            and self._multi_agent_result.plan.scenario is not None
            and self._multi_agent_result.plan.problem_type
            in {
                ProblemType.SCENARIO_APPLICATION,
                ProblemType.MIXED,
            }
        )
        prompt = (
            (
                "Write the scenario-specific final answer from the validated "
                "application findings and verified evidence above. Preserve "
                "every material unknown and conditional branch."
                if is_scenario
                else "Synthesize the final answer from the verified task "
                "artifacts and evidence above. Preserve every material limitation."
            )
            if is_multi_agent
            else "Using the evidence above, write the final answer now."
        )
        stream_history = [*self._chat_history, ChatTurn(role="user", text=prompt)]
        system_prompt = (
            SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT
            if is_scenario
            else FINAL_SYNTHESIS_SYSTEM_PROMPT
            if is_multi_agent
            else FINAL_SYSTEM_PROMPT
        )
        client = self._multi_agent_final_llm if is_multi_agent else self._llm
        assert client is not None

        async for text in client.stream_text(
            stream_history,
            system_prompt,
            thinking_level=None if is_multi_agent else "high",
        ):
            await self._publish_final_chunk(text)

        if not self._final_answer_chunks:
            if is_multi_agent:
                raise RuntimeError(
                    "The final model returned no answer content for the "
                    "completed research workflow."
                )
            # Retain legacy non-indexed behavior; the indexed multi-agent path
            # above never substitutes a deterministic/direct fallback.
            await self._publish_final_chunk(
                fallback_answer
                or (
                    "Üzgünüm, bu soru için şu anda bir cevap oluşturamadım. "
                    "Lütfen tekrar deneyin."
                )
            )
        else:
            final_text = "".join(self._final_answer_chunks)
            self._chat_history.append(ChatTurn(role="model", text=final_text))

            usage = client.last_stream_usage()
            if usage:
                self.token_usage.add_api_call(
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.output_tokens,
                    thinking_tokens=usage.thinking_tokens,
                )
                if is_multi_agent:
                    self._step_count += 1
                await self._report_llm_call(
                    "final_answer",
                    usage,
                    client=client,
                    agent_role="final" if is_multi_agent else None,
                    agent_id="final-synthesizer" if is_multi_agent else None,
                )

    async def stream_final_answer(
        self,
        fallback_answer: str | None = None,
    ) -> AsyncIterator[str]:
        """Replay and follow this run's single-flight final answer stream."""
        self.start_final_answer(fallback_answer)

        next_chunk = 0
        while True:
            async with self._final_answer_condition:
                await self._final_answer_condition.wait_for(
                    lambda: (
                        next_chunk < len(self._final_answer_chunks)
                        or self._final_answer_complete
                    )
                )
                available = self._final_answer_chunks[next_chunk:]
                is_complete = self._final_answer_complete
                final_error = self._final_answer_error

            for chunk in available:
                next_chunk += 1
                yield chunk

            if is_complete and next_chunk == len(self._final_answer_chunks):
                if final_error is not None:
                    raise final_error
                return

    def call_tool(self, tool_name: Tools, tool_input: dict[str, Any]) -> None:
        """
        Execute a tool and add the result to the conversation history.

        Skips the actual (potentially expensive) tool call and short-circuits
        with a small canned message if this looks like a near-duplicate of a
        recent call — see the near-duplicate guard constants near the top of
        this module for why.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dictionary of arguments to pass to the tool.
        """
        dedup = _dedup_key(tool_name, tool_input)
        if dedup is not None and self._is_near_duplicate_call(*dedup):
            result = (
                "SKIPPED: this looks like a near-duplicate of a search you "
                "already ran in the last few steps — repeating it (even via "
                "a different tool name for the same document) will not "
                "surface new information. Try a genuinely different tool or "
                "query (a different search angle, a specific article/"
                "document number, an exact phrase for grep), or if you "
                "already have enough information, STOP now and give your "
                "final answer."
            )
        else:
            try:
                result = TOOLS[tool_name](**tool_input)
            except Exception as e:
                result = (
                    f"An error occurred while calling tool {tool_name} "
                    f"with {tool_input}: {e}"
                )
            if dedup is not None:
                self._recent_tool_calls.append(dedup)
                self._recent_tool_calls = self._recent_tool_calls[
                    -_DUPLICATE_CALL_LOOKBACK:
                ]

        # Track tool result sizes
        chunk_count, chars = self.token_usage.add_tool_result(result, tool_name)
        self._report_retrieval(tool_name, chunk_count, chars)

        self._chat_history.append(
            ChatTurn(role="user", text=f"Tool result for {tool_name}:\n\n{result}")
        )

    async def call_tools(self, calls: list[tuple[Tools, dict[str, Any]]]) -> None:
        """Execute a bounded independent batch and append one combined turn."""
        if not 2 <= len(calls) <= 3:
            raise ValueError("A tool batch must contain two or three calls.")

        results: list[str | None] = [None] * len(calls)
        pending: list[tuple[int, Tools, dict[str, Any]]] = []
        for index, (tool_name, tool_input) in enumerate(calls):
            dedup = _dedup_key(tool_name, tool_input)
            if dedup is not None and self._is_near_duplicate_call(*dedup):
                results[index] = "SKIPPED: duplicate of a recent or batched tool call."
                continue
            if dedup is not None:
                self._recent_tool_calls.append(dedup)
                self._recent_tool_calls = self._recent_tool_calls[
                    -_DUPLICATE_CALL_LOOKBACK:
                ]
            pending.append((index, tool_name, tool_input))

        async def execute(tool_name: Tools, tool_input: dict[str, Any]) -> str:
            try:
                return await asyncio.to_thread(TOOLS[tool_name], **tool_input)
            except Exception as exc:
                return (
                    f"An error occurred while calling tool {tool_name} "
                    f"with {tool_input}: {exc}"
                )

        completed = await asyncio.gather(
            *(execute(tool_name, tool_input) for _, tool_name, tool_input in pending)
        )
        for (index, _tool_name, _tool_input), result in zip(pending, completed):
            results[index] = result

        sections: list[str] = ["Batch tool results:"]
        for (tool_name, tool_input), result in zip(calls, results):
            resolved = result or "No result"
            chunk_count, chars = self.token_usage.add_tool_result(resolved, tool_name)
            self._report_retrieval(tool_name, chunk_count, chars)
            sections.append(f"\n## {tool_name} {tool_input}\n{resolved}")
        self._chat_history.append(ChatTurn(role="user", text="\n".join(sections)))

    def _is_near_duplicate_call(self, group: str, key: str) -> bool:
        fuzzy = group in _FUZZY_DEDUP_GROUPS
        for prior_group, prior_key in self._recent_tool_calls:
            if prior_group != group:
                continue
            if fuzzy:
                if (
                    _query_token_overlap(key, prior_key)
                    >= _DUPLICATE_QUERY_SIMILARITY_THRESHOLD
                ):
                    return True
            elif key.strip().lower() == prior_key.strip().lower():
                return True
        return False

    def reset(self) -> None:
        """Reset the agent's conversation history and token tracking."""
        self.clear_final_answer()
        self._chat_history.clear()
        self.token_usage = TokenUsage()
        self._step_count = 0
        self._max_steps = _MAX_STEPS
        self._recent_tool_calls.clear()
        self._prepared_indexed_evidence = False
        self._multi_agent_orchestrator = None
        self._multi_agent_result = None
        self._multi_agent_final_llm = None
        self._multi_agent_research_active = False
        self._forced_stop = False
        self._stop_rejections = 0
