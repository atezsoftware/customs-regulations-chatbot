"""Request-bound operational limits for Cropped research, never legal acceptance."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import unicodedata
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
)
from onyx.regulatory.heading_path import (
    RegulatoryProvisionReference,
    extract_single_regulatory_provision_reference,
)
from onyx.regulatory.source_identity import named_law_number

_POLICY_ARTIFACT = "exp-cropped-rerank-audit-20261012/cohort/cohort-table.json"
_POLICY_ARTIFACT_SHA256 = (
    "4e3a537c4c872fe2412d4f49b9746ed0c8098fa082203c91a285fd2392ad349b"
)
_MAX_ORIGINALS = 2000
_MAX_CALLS = 4096
_MAX_SEARCHES = 1024
_MAX_DELEGATED_OWNERS = 64
_HASH = r"^[a-f0-9]{64}$"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class CroppedResearchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    research_model_calls: int = Field(default=8, ge=1, le=128)
    searches: int = Field(default=2, ge=1, le=128)
    original_records: int = Field(default=224, ge=1, le=_MAX_ORIGINALS)
    critical_reads: int = Field(default=16, ge=1, le=64)
    enforce_observed_limits: bool = False
    observed_model: str = "gpt-6.1-sol"
    observed_assistant_message_ids: tuple[int, ...] = (5010, 5082)
    observed_root_decisions: tuple[int, ...] = (7, 9)
    observed_searches: tuple[int, ...] = (2, 2)
    observed_original_records: tuple[int, ...] = (243, 202)
    artifact: str = _POLICY_ARTIFACT
    artifact_sha256: str = Field(default=_POLICY_ARTIFACT_SHA256, pattern=_HASH)


class CroppedResearchAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    allowed: bool
    reason: str
    long_tail: bool
    critical_operation_key: str | None = None
    absence_proven: Literal[False] = False


class _Original(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_id: str = Field(min_length=1, max_length=512)
    chunk_id: str | None = Field(default=None, max_length=512)
    text_hash: str = Field(pattern=_HASH)

    @property
    def key(self) -> tuple[str, str | None, str]:
        return self.source_id, self.chunk_id, self.text_hash


class _OwnerCalls(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    research: int = Field(default=0, ge=0, le=_MAX_CALLS)
    final: int = Field(default=0, ge=0, le=_MAX_CALLS)


class _CriticalRead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    operation: dict[str, JsonValue]
    target_key: str = Field(min_length=1, max_length=512)


class _NamedSourceAlias(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    number: str = Field(pattern=r"^[1-9]\d{2,4}$")
    article: str = Field(min_length=1, max_length=64)
    citation: int = Field(ge=1)
    text_hash: str = Field(pattern=_HASH)
    quote: str = Field(min_length=1, max_length=4096)


class _FailedNamedSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_names: list[str] = Field(min_length=1, max_length=16)
    articles: list[str] = Field(min_length=1, max_length=64)
    witnessed_aliases: list[_NamedSourceAlias] = Field(
        default_factory=list, max_length=1
    )
    call_id: str = Field(min_length=1, max_length=512)


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    owner_calls: dict[str, _OwnerCalls] = Field(default_factory=dict)
    search_calls: list[str] = Field(default_factory=list, max_length=_MAX_SEARCHES)
    originals: list[_Original] = Field(default_factory=list, max_length=_MAX_ORIGINALS)
    critical_reads: dict[str, _CriticalRead] = Field(default_factory=dict)
    failed_named_sources: dict[str, _FailedNamedSource] = Field(
        default_factory=dict, max_length=128
    )
    read_planned_owners: list[str] = Field(default_factory=list, max_length=64)
    delegated_research_owners: list[str] = Field(
        default_factory=list, max_length=_MAX_DELEGATED_OWNERS
    )
    forced_finish_reason: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def consistent(self) -> _State:
        if (
            sum(row.research + row.final for row in self.owner_calls.values())
            > _MAX_CALLS
        ):
            raise ValueError("Cropped model accounting capacity exceeded")
        if any(not owner.strip() or len(owner) > 512 for owner in self.owner_calls):
            raise ValueError("Cropped model owner is invalid")
        if len(set(self.delegated_research_owners)) != len(
            self.delegated_research_owners
        ) or any(
            not owner.strip()
            or len(owner) > 512
            or owner == "root"
            or owner not in self.owner_calls
            or self.owner_calls[owner].research == 0
            for owner in self.delegated_research_owners
        ):
            raise ValueError("Cropped delegated research owner is invalid")
        if len(set(self.read_planned_owners)) != len(self.read_planned_owners) or any(
            not owner.strip() or len(owner) > 512 for owner in self.read_planned_owners
        ):
            raise ValueError("Cropped read-planning owner is invalid")
        if len(set(self.search_calls)) != len(self.search_calls) or any(
            not value.strip() or len(value) > 512 for value in self.search_calls
        ):
            raise ValueError("Cropped search identities are invalid")
        if len({row.key for row in self.originals}) != len(self.originals):
            raise ValueError("Cropped original identities are duplicated")
        for key, row in self.critical_reads.items():
            if key != _digest(row.operation):
                raise ValueError("Cropped critical operation identity changed")
            if row.operation.get("tool") == "resolve_source" and (
                set(row.operation) != {"tool", "failed_named_source_key"}
                or row.operation.get("failed_named_source_key")
                not in self.failed_named_sources
            ):
                raise ValueError("Cropped source resolution lacks a failed named read")
        for key, row in self.failed_named_sources.items():
            if (
                not row.call_id.strip()
                or any(_named_source_key(name) != key for name in row.source_names)
                or len({_source_title(name) for name in row.source_names})
                != len(row.source_names)
                or any(_article_identity(article) is None for article in row.articles)
                or any(not _alias_agrees(alias, row) for alias in row.witnessed_aliases)
            ):
                raise ValueError("Cropped failed named source identity changed")
        return self


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[3] = 3
    run_id: str
    scope_hash: str = Field(pattern=_HASH)
    request_hash: str = Field(pattern=_HASH)
    policy_hash: str = Field(pattern=_HASH)
    state: _State
    state_hash: str = Field(pattern=_HASH)


_FREE_TOOLS = frozenset(
    {
        "read_evidence",
        "inspect_evidence_path",
        "inspect_research",
        "read_research_state",
        "discover_tools",
        "list_researchers",
        "wait_researcher",
        "update_research",
        "submit_answer",
        "submit_partial_answer",
        "submit_retained_answer",
        "submit_retained_partial_answer",
        "compose_tool_calls",
    }
)
_TARGET_FIELDS: dict[str, tuple[str, ...]] = {
    "read_provision": ("source_id", "article", "paragraph", "clause", "start"),
    "read_named_provision": ("source_name", "article", "paragraph", "clause", "start"),
    "read_chunk": ("source_id", "chunk_id"),
    "read_chunk_context": ("source_id", "chunk_id", "offset"),
    "follow_reference": ("source_id", "chunk_id", "target_source_id", "article"),
}
_ARTICLE_ID = re.compile(
    r"(?:(?P<qualifier>ek|gecici|geçici|mukerrer|mükerrer)\s+)?"
    r"(?:(?:madde|article|art\.?)\s*[:.]?\s*)?"
    r"(?P<number>\d+(?:/?[a-z])?)",
    re.IGNORECASE,
)
_ARTICLE_HEADING = re.compile(
    r"(?:(?:ek|gecici|geçici|mukerrer|mükerrer)\s+)?"
    r"(?:madde|article|art\.?)\s*[:.]?\s*\d+(?:/?[a-z])?",
    re.IGNORECASE,
)


def _fold_article(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    ).strip()


def _source_title(value: str) -> str:
    folded = _fold_article(value).replace("ı", "i")
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def _own_law_number(value: str) -> str | None:
    # A number cited by another instrument is not that source's own identity.
    numbers = set(re.findall(r"\b(\d{3,5})\s+sayili\b", _source_title(value)))
    return named_law_number(value) if len(numbers) <= 1 else None


def _named_source_key(value: str) -> str | None:
    if not value.strip() or len(value) > 512:
        return None
    title = _source_title(value)
    if not title or title in {
        "kanun",
        "kanunu",
        "gumruk",
        "yonetmelik",
        "yonetmeligi",
        "teblig",
        "tebligi",
        "karar",
        "karari",
        "genelge",
        "kaynak",
        "law",
        "act",
        "regulation",
    }:
        return None
    number = _own_law_number(value)
    return (
        _digest({"kind": "failed_named_source", "law_number": number})
        if number
        else _digest({"kind": "failed_named_source", "source_title": title})
    )


def _failed_named_read(operation: JsonValue) -> tuple[str, str, str] | None:
    if (
        not isinstance(operation, dict)
        or operation.get("name") != "read_named_provision"
    ):
        return None
    arguments, data, call_id = (
        operation.get(key) for key in ("arguments", "data", "call_id")
    )
    if (
        not isinstance(arguments, dict)
        or not isinstance(data, dict)
        or not isinstance(call_id, str)
        or not call_id.strip()
        or len(call_id) > 512
    ):
        return None
    name, article = arguments.get("source_name"), arguments.get("article")
    if (
        operation.get("status") not in {"not_found", "ambiguous"}
        or not isinstance(name, str)
        or _named_source_key(name) is None
        or not isinstance(article, str)
        or _article_identity(article) is None
        or arguments.get("source_id") is not None
        or data.get("source_id") is not None
        or data.get("absence_proven") is not False
        or data.get("not_read") is True
        or data.get("blocked_by_research_limit") is True
        or any(
            operation.get(field)
            for field in ("evidence", "evidence_ids", "original_reads")
        )
    ):
        return None
    sources, more = data.get("sources"), data.get("has_more")
    if (
        not isinstance(sources, list)
        or type(more) is not bool
        or any(
            not isinstance(source, dict)
            or not isinstance(source.get("source_id"), str)
            or not isinstance(source.get("name"), str)
            for source in sources
        )
        or (operation["status"] == "not_found" and (sources or more))
        or (operation["status"] == "ambiguous" and not sources and not more)
    ):
        return None
    return name, call_id, article


def _article_identity(value: str) -> RegulatoryProvisionReference | None:
    matched = _ARTICLE_ID.fullmatch(_fold_article(value))
    if matched is None:
        return None
    qualifier = matched.group("qualifier") or ""
    return extract_single_regulatory_provision_reference(
        f"{qualifier} MADDE {matched.group('number')}"
    )


def _alias_agrees(alias: _NamedSourceAlias, failed: _FailedNamedSource) -> bool:
    if not any(
        _article_identity(alias.article) == _article_identity(article)
        for article in failed.articles
    ):
        return False
    for name in failed.source_names:
        number = _own_law_number(name)
        if number is not None and number != alias.number:
            continue
        title = _source_title(name)
        if number:
            title = re.sub(r"^\d+\s+sayili\s+", "", title)
        words = title.split()
        if not words or words[-1] not in {"kanun", "kanunu"}:
            continue
        # Only case inflections of the complete registered law title vary.
        own_title = r"\s+".join(re.escape(word) for word in words[:-1])
        suffix = r"kanun(?:u|un|unun|a|una|da|unda|dan|undan)?"
        pattern = rf"\b{re.escape(alias.number)}\s+sayili\s+{own_title}{r'\s+' if own_title else ''}{suffix}\b"
        if re.search(pattern, _source_title(alias.quote)) is not None:
            return True
    return False


def _article_locator(arguments: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    source = arguments.get("source_id")
    pattern = arguments.get("pattern")
    start = arguments.get("start", 0)
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(pattern, str)
        or _ARTICLE_HEADING.fullmatch(_fold_article(pattern)) is None
        or arguments.get("mode", "literal") != "literal"
        or isinstance(start, bool)
        or not isinstance(start, int)
        or start != 0
    ):
        return None
    identity = _article_identity(pattern)
    if identity is None:
        return None
    return {
        "tool": "search_source_text",
        "source_id": " ".join(source.split()).casefold(),
        "article": " ".join(
            value for value in (identity.qualifier, identity.article_no) if value
        ).casefold(),
        "mode": "literal",
    }


def _critical_operation(
    name: str, arguments: dict[str, JsonValue]
) -> dict[str, JsonValue] | None:
    if name == "search_source_text":
        return _article_locator(arguments)
    fields = _TARGET_FIELDS.get(name)
    if fields is None:
        return None
    required = (
        ("source_name", "article")
        if name == "read_named_provision"
        else (
            ("source_id", "article")
            if name == "read_provision"
            else ("source_id", "chunk_id", "target_source_id", "article")
            if name == "follow_reference"
            else ("source_id", "chunk_id")
        )
    )
    if any(
        not isinstance(arguments.get(field), str) or not str(arguments[field]).strip()
        for field in required
    ):
        return None
    operation: dict[str, JsonValue] = {"tool": name}
    for field in fields:
        value = arguments.get(field)
        if value is None:
            continue
        if field in {"start", "offset", "limit"}:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if field == "limit" else 0)
            ):
                return None
        elif not isinstance(value, str) or not value.strip():
            return None
        operation[field] = (
            " ".join(value.split()).casefold() if isinstance(value, str) else value
        )
    if name == "read_chunk_context":
        operation.setdefault("offset", 0)
        # Page size is not a new identity; changing it does not authorize a repeat.
        size = arguments.get("limit", 100)
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 100:
            return None
    return operation


class CroppedResearchLimit:
    """Share discovery progress across root and workers; cost remains a host gate."""

    def __init__(
        self,
        context: RunContext,
        request: str,
        *,
        policy: CroppedResearchPolicy | None = None,
    ) -> None:
        self._context = context
        self.request = request
        self.policy = policy or CroppedResearchPolicy()
        self.run_id = context.run_id
        self.scope_hash = _digest(context.scope)
        self.request_hash = _digest(request)
        self.policy_hash = _digest(self.policy.model_dump(mode="json"))
        self._state = _State()
        self._lock = threading.RLock()

    def _fence(self, context: RunContext | None = None) -> None:
        current = context or self._context
        if (
            current.run_id,
            _digest(current.scope),
            _digest(self.request),
            _digest(self.policy.model_dump(mode="json")),
        ) != (self.run_id, self.scope_hash, self.request_hash, self.policy_hash):
            raise ValueError(
                "Cropped research limit run, scope, request or policy changed"
            )

    def check_context(self, context: RunContext) -> None:
        """Allow sharing only with a worker carrying this run's captured authority."""
        self._fence(context)

    def _long_tail(self) -> bool:
        return bool(self._state.forced_finish_reason) or (
            sum(row.research for row in self._state.owner_calls.values())
            >= self.policy.research_model_calls
            and len(self._state.search_calls) > self.policy.searches
            and len(self._state.originals) >= self.policy.original_records
        )

    @property
    def long_tail(self) -> bool:
        self._fence()
        with self._lock:
            return self._long_tail()

    @property
    def forced_finish(self) -> bool:
        """Report a host-requested transition, independently of observed volume."""
        self._fence()
        with self._lock:
            return bool(self._state.forced_finish_reason)

    def _admission_closed(self) -> bool:
        return self.forced_finish or (
            self.policy.enforce_observed_limits and self._long_tail()
        )

    @property
    def has_delegated_research(self) -> bool:
        """Inspect host-reserved research entry, never source or legal completion."""
        self._fence()
        with self._lock:
            return (
                self._context.services.get("cropped_research_controller") is self
                and self._context.services.get("asv3_exp_cropped") is True
                and self._context.services.get("asv3_exp_cheap") is not True
                and bool(self._state.delegated_research_owners)
            )

    def _writer_transition(self) -> bool:
        return self.has_delegated_research and (
            sum(row.research for row in self._state.owner_calls.values())
            >= self.policy.research_model_calls
        )

    @property
    def budget_limited(self) -> bool:
        """Bound paid continuation without declaring source discovery closed."""
        self._fence()
        with self._lock:
            return self.forced_finish or (
                self.policy.enforce_observed_limits
                and (self._long_tail() or self._writer_transition())
            )

    def progress(self) -> dict[str, JsonValue]:
        self._fence()
        with self._lock:
            return {
                "research_model_calls": sum(
                    row.research for row in self._state.owner_calls.values()
                ),
                "final_model_calls": sum(
                    row.final for row in self._state.owner_calls.values()
                ),
                "searches": len(self._state.search_calls),
                "original_records": len(self._state.originals),
                "original_source_ids": len(
                    {row.source_id for row in self._state.originals}
                ),
                "critical_reads_reserved": len(self._state.critical_reads),
                "failed_named_source_targets": len(self._state.failed_named_sources),
                "long_tail": self._long_tail(),
                "forced_finish": self.forced_finish,
                "forced_finish_reason": self._state.forced_finish_reason,
            }

    def available_tools(
        self, tools: list[dict[str, JsonValue]], context: RunContext
    ) -> list[dict[str, JsonValue]]:
        """Omit only schemas whose operations are already closed by this controller."""
        self._fence(context)
        with self._lock:
            if not self._admission_closed():
                return tools
            if not self.policy.enforce_observed_limits:
                return [
                    tool
                    for tool in tools
                    if isinstance(function := tool.get("function"), dict)
                    and function.get("name")
                    in _FREE_TOOLS - {"compose_tool_calls", "discover_tools"}
                ]
            # Complete cached range reads are admitted before the tail gate.
            names = (
                _FREE_TOOLS
                | _TARGET_FIELDS.keys()
                | {
                    "search_source_text",
                    "read_source_range",
                }
            )
            if len(self._state.critical_reads) < self.policy.critical_reads and any(
                operation is not None
                and _digest(operation) not in self._state.critical_reads
                for failed in self._state.failed_named_sources.values()
                if (
                    operation := self._source_resolution(
                        {"query": failed.source_names[0]}
                    )
                )
                is not None
            ):
                names = names | {"resolve_source"}
            return [
                tool
                for tool in tools
                if isinstance(function := tool.get("function"), dict)
                and function.get("name") in names
            ]

    def reserve_read_plan(self, context: RunContext, owner: str) -> bool:
        """Reserve one planning attempt before I/O; failure is not research completion."""
        self._fence(context)
        if not owner.strip() or len(owner) > 512:
            raise ValueError("Cropped read-planning owner is invalid")
        with self._lock:
            if (
                not self.policy.enforce_observed_limits
                or not self._long_tail()
                or context.depth
                or owner in self._state.read_planned_owners
            ):
                return False
            if len(self._state.read_planned_owners) >= 64:
                raise ValueError("Cropped read-planning capacity exceeded")
            self._state.read_planned_owners.append(owner)
            return True

    def read_plan_reserved(self, context: RunContext, owner: str) -> bool:
        """Inspect reservation without rebuilding or registering a planning packet."""
        self._fence(context)
        if not owner.strip() or len(owner) > 512:
            raise ValueError("Cropped read-planning owner is invalid")
        with self._lock:
            return owner in self._state.read_planned_owners

    def register_failed_named_reads(
        self, operations: list[JsonValue], context: RunContext
    ) -> int:
        """Register this run's host-validated operations, never model proposals."""
        self._fence(context)
        failed = [
            row
            for operation in operations
            if (row := _failed_named_read(operation)) is not None
        ]
        with self._lock:
            updated = dict(self._state.failed_named_sources)
            for name, call_id, article in failed:
                key = _named_source_key(name)
                assert key is not None
                previous = updated.get(key)
                if previous is None:
                    updated[key] = _FailedNamedSource(
                        source_names=[name], call_id=call_id, articles=[article]
                    )
                else:
                    names = previous.source_names
                    if _source_title(name) not in {
                        _source_title(value) for value in names
                    }:
                        names = [*names, name]
                    articles = previous.articles
                    if not any(
                        _article_identity(article) == _article_identity(value)
                        for value in articles
                    ):
                        articles = [*articles, article]
                    updated[key] = previous.model_copy(
                        update={"source_names": names, "articles": articles}
                    )
            if len(updated) > 128 or any(
                len(row.source_names) > 16 or len(row.articles) > 64
                for row in updated.values()
            ):
                raise ValueError("Cropped failed named source capacity exceeded")
            added = len(updated) - len(self._state.failed_named_sources)
            self._state.failed_named_sources = updated
            return added

    def register_failed_named_source_aliases(
        self, references: list[JsonValue], context: RunContext
    ) -> int:
        """Bind a numeric lookup alias to an actual same-run literal NAV witness."""
        self._fence(context)
        ledger = context.services.get("evidence")
        if not isinstance(
            ledger, EvidenceLedger
        ) or ledger is not self._context.services.get("evidence"):
            raise ValueError(
                "Cropped source aliases require this run's actual evidence ledger"
            )
        with self._lock:
            added = 0
            for reference in references:
                if not isinstance(reference, dict):
                    continue
                target = reference.get("target")
                if not isinstance(target, dict):
                    continue
                number, article = target.get("instrument_number"), target.get("article")
                if (
                    not isinstance(number, str)
                    or re.fullmatch(r"[1-9]\d{2,4}", number) is None
                    or not isinstance(article, str)
                ):
                    continue
                qualifier = target.get("qualifier")
                if qualifier is not None and not isinstance(qualifier, str):
                    continue
                article = f"{qualifier} {article}" if qualifier else article
                witnesses = reference.get("exact_witnesses")
                if not isinstance(witnesses, list):
                    continue
                for key, failed in list(self._state.failed_named_sources.items()):
                    named = target.get("source_name")
                    if named is not None and (
                        not isinstance(named, str)
                        or not any(
                            _source_title(named) == _source_title(name)
                            for name in failed.source_names
                        )
                    ):
                        continue
                    if failed.witnessed_aliases:
                        continue
                    for witness in witnesses:
                        if not isinstance(witness, dict):
                            continue
                        citation, quote = (
                            witness.get("citation"),
                            witness.get("source_quote"),
                        )
                        if (
                            type(citation) is not int
                            or not isinstance(quote, str)
                            or not 0 < len(quote) <= 4096
                        ):
                            continue
                        item = ledger.get(citation)
                        if item is None or quote not in item.text:
                            continue
                        alias = _NamedSourceAlias(
                            number=number,
                            article=article,
                            citation=citation,
                            text_hash=item.text_hash,
                            quote=quote,
                        )
                        if not _alias_agrees(alias, failed):
                            continue
                        self._state.failed_named_sources[key] = failed.model_copy(
                            update={"witnessed_aliases": [alias]}
                        )
                        added += 1
                        break
            return added

    def _source_numbers(self, failed: _FailedNamedSource) -> set[str]:
        numbers = {
            number
            for name in failed.source_names
            if (number := _own_law_number(name)) is not None
        }
        ledger = self._context.services.get("evidence")
        if isinstance(ledger, EvidenceLedger):
            for alias in failed.witnessed_aliases:
                item = ledger.get(alias.citation)
                if (
                    item is not None
                    and item.text_hash == alias.text_hash
                    and alias.quote in item.text
                    and _alias_agrees(alias, failed)
                ):
                    numbers.add(alias.number)
        return numbers

    def _source_resolution(
        self, arguments: dict[str, JsonValue]
    ) -> dict[str, JsonValue] | None:
        query = arguments.get("query")
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 512
            or type(arguments.get("offset", 0)) is not int
            or arguments.get("offset", 0) != 0
            or type(limit := arguments.get("limit", 20)) is not int
            or not 1 <= limit <= 100
        ):
            return None
        title = _source_title(query)
        for key, failed in self._state.failed_named_sources.items():
            if any(
                title == _source_title(name) for name in failed.source_names
            ) or query.strip() in self._source_numbers(failed):
                numbers = self._source_numbers(failed)
                linked = [
                    source_key
                    for source_key, row in self._state.failed_named_sources.items()
                    if source_key == key
                    or numbers.intersection(self._source_numbers(row))
                ]
                # A witnessed alias cannot reopen a source reserved under its full title.
                for row in self._state.critical_reads.values():
                    if (
                        row.operation.get("tool") == "resolve_source"
                        and row.operation.get("failed_named_source_key") in linked
                    ):
                        return copy.deepcopy(row.operation)
                return {"tool": "resolve_source", "failed_named_source_key": linked[0]}
        return None

    def reserve_model(
        self,
        owner: str,
        *,
        research: bool = True,
        context: RunContext | None = None,
    ) -> dict[str, JsonValue]:
        """Count one real ResearchModel invocation before paid I/O, even if it fails."""
        self._fence()
        if not owner.strip() or len(owner) > 512 or not isinstance(research, bool):
            raise ValueError(
                "Cropped model accounting requires a bounded owner and mode"
            )
        delegated = False
        if context is not None:
            self._fence(context)
            if context.services.get("cropped_research_controller") is not self:
                raise ValueError("Cropped model context does not own this controller")
            delegated = (
                research
                and context.services.get("asv3_exp_cropped") is True
                and context.services.get("asv3_exp_cheap") is not True
                and type(context.depth) is int
                and context.depth > 0
                and context.services.get("independent_question") is not True
            )
            if delegated:
                task_id = context.services.get("task_id")
                if task_id is None or task_id == "":
                    delegated = False
                elif not isinstance(task_id, str) or task_id != owner:
                    raise ValueError(
                        "Cropped delegated model owner differs from its task"
                    )
                elif owner == "root":
                    delegated = False
        with self._lock:
            if (
                sum(
                    row.research + row.final for row in self._state.owner_calls.values()
                )
                >= _MAX_CALLS
            ):
                raise ValueError("Cropped model accounting capacity exceeded")
            if (
                delegated
                and owner not in self._state.delegated_research_owners
                and len(self._state.delegated_research_owners) >= _MAX_DELEGATED_OWNERS
            ):
                raise ValueError("Cropped delegated research capacity exceeded")
            row = self._state.owner_calls.get(owner, _OwnerCalls())
            field = "research" if research else "final"
            self._state.owner_calls[owner] = row.model_copy(
                update={field: getattr(row, field) + 1}
            )
            if delegated and owner not in self._state.delegated_research_owners:
                self._state.delegated_research_owners.append(owner)
            return self.progress()

    def record_search(self, call_id: str) -> None:
        self._fence()
        if not call_id.strip() or len(call_id) > 512:
            raise ValueError("Cropped search needs a bounded call identity")
        with self._lock:
            if call_id in self._state.search_calls:
                return
            if len(self._state.search_calls) >= _MAX_SEARCHES:
                raise ValueError("Cropped search accounting capacity exceeded")
            self._state.search_calls.append(call_id)

    def record_sources(self, items: Iterable[EvidenceItem]) -> None:
        """Retain every returned original; thresholds never clip operative text."""
        self._fence()
        originals = [
            _Original(
                source_id=item.source_id,
                chunk_id=item.chunk_id,
                text_hash=item.text_hash,
            )
            for item in items
        ]
        with self._lock:
            retained = {row.key: row for row in self._state.originals}
            retained.update({row.key: row for row in originals})
            if len(retained) > _MAX_ORIGINALS:
                raise ValueError("Cropped original accounting capacity exceeded")
            self._state.originals = list(retained.values())

    def force_finish(self, reason: str) -> None:
        self._fence()
        if not reason.strip() or len(reason) > 512:
            raise ValueError("Cropped forced finish requires a precise bounded reason")
        with self._lock:
            self._state.forced_finish_reason = (
                self._state.forced_finish_reason or reason
            )

    def admit_tool(
        self,
        name: str,
        arguments: dict[str, JsonValue],
        *,
        critical_target_key: str | None = None,
        reserve: bool = True,
    ) -> CroppedResearchAdmission:
        self._fence()
        with self._lock:
            long_tail = self._long_tail()
            closed = self._admission_closed()
            free = name in _FREE_TOOLS and (
                self.policy.enforce_observed_limits
                or name not in {"compose_tool_calls", "discover_tools"}
            )
            if not closed or free:
                return CroppedResearchAdmission(
                    allowed=True,
                    reason="ordinary_research"
                    if not closed
                    else "retained_evidence_or_host_control",
                    long_tail=long_tail,
                )
            if not self.policy.enforce_observed_limits:
                return CroppedResearchAdmission(
                    allowed=False,
                    reason="host_finish_requires_available_evidence_only_unread_not_absent",
                    long_tail=long_tail,
                )
            operation = _critical_operation(name, arguments)
            if name == "resolve_source":
                operation = self._source_resolution(arguments)
                if operation is not None:
                    critical_target_key = str(operation["failed_named_source_key"])
            if operation is not None and name == "search_source_text":
                source = operation["source_id"]
                article = operation["article"]
                if not any(
                    " ".join(row.source_id.split()).casefold() == source
                    for row in self._state.originals
                ) or not any(
                    (
                        row.operation.get("tool") == "read_provision"
                        and row.operation.get("source_id") == source
                        or row.operation.get("tool") == "follow_reference"
                        and row.operation.get("target_source_id") == source
                    )
                    and isinstance(previous := row.operation.get("article"), str)
                    and _article_identity(previous) == _article_identity(str(article))
                    for row in self._state.critical_reads.values()
                ):
                    operation = None
            if (
                operation is None
                or not critical_target_key
                or not critical_target_key.strip()
                or len(critical_target_key) > 512
            ):
                return CroppedResearchAdmission(
                    allowed=False,
                    reason="broad_discovery_closed_or_no_exact_critical_target",
                    long_tail=True,
                )
            key = _digest(operation)
            if key in self._state.critical_reads:
                return CroppedResearchAdmission(
                    allowed=False,
                    reason="critical_read_already_reserved_reuse_recorded_evidence",
                    long_tail=True,
                    critical_operation_key=key,
                )
            if len(self._state.critical_reads) >= self.policy.critical_reads:
                return CroppedResearchAdmission(
                    allowed=False,
                    reason="critical_closure_allowance_exhausted_unread_not_absent",
                    long_tail=True,
                )
            if reserve:
                self._state.critical_reads[key] = _CriticalRead(
                    operation=operation, target_key=critical_target_key
                )
            return CroppedResearchAdmission(
                allowed=True,
                reason="one_exact_critical_closure_attempt",
                long_tail=True,
                critical_operation_key=key,
            )

    @staticmethod
    def blocked_tool(admission: CroppedResearchAdmission) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Broad discovery is closed. Reuse recorded originals and finish with supported findings or precise unresolved source limitations.",
            data={
                "blocked_by_research_limit": True,
                "not_read": True,
                "absence_proven": False,
                "admission": admission.model_dump(mode="json"),
            },
        )

    def finish_request(
        self, view: HarnessView, context: RunContext
    ) -> dict[str, JsonValue] | None:
        self._fence(context)
        with self._lock:
            if not self.policy.enforce_observed_limits:
                if not self.forced_finish:
                    return None
                return {
                    "mode": "finish_with_available_evidence",
                    "progress": self.progress(),
                    "policy": self.policy.model_dump(mode="json"),
                    "pending_publication_gap": copy.deepcopy(view.publication_gap),
                    "required_evidence_numbers": list(view.required_evidence_numbers),
                    "research_statuses_unchanged": True,
                    "absence_proven": False,
                    "instruction": "Research is closed by an explicit host transition. Reuse recorded facts, findings and complete delivered originals for supported writing. Do not start new reads, searches or delegation. Preserve supported qualifications and exact unresolved source limitations; unread evidence does not prove absence and unresolved conditions remain unresolved. Submit through the existing publication guards.",
                }
            if not self._long_tail():
                if (
                    not self._writer_transition()
                    or context.depth != 0
                    or context.services.get("cropped_research_controller") is not self
                    or context.services.get("asv3_exp_cropped") is not True
                    or context.services.get("asv3_exp_cheap") is True
                    or context.services.get("independent_question") is True
                    or view.draft_to_repair is not None
                    or view.publication_gap is not None
                ):
                    return None
                return {
                    "mode": "fresh_source_writer_transition",
                    "progress": self.progress(),
                    "policy": self.policy.model_dump(mode="json"),
                    "pending_publication_gap": None,
                    "required_evidence_numbers": list(view.required_evidence_numbers),
                    "research_statuses_unchanged": True,
                    "absence_proven": False,
                    "instruction": (
                        "Assess every requested outcome and write the supported conclusion "
                        "from all delivered originals, facts and findings. If known material "
                        "originals are still needed, request the independent reads together. "
                        "Use source navigation to select complete operative provisions, "
                        "including conditions, exceptions, material references and their "
                        "continuations. Preserve every supported qualification and required "
                        "citation. An unread or unresolved source is not absent; do not "
                        "mark unresolved conditions complete. This transition does not "
                        "close source discovery or establish legal completeness. Submit "
                        "through the existing publication guards with precise unresolved "
                        "limits when necessary."
                    ),
                }
            return {
                "mode": "finish_with_required_evidence",
                "progress": self.progress(),
                "policy": self.policy.model_dump(mode="json"),
                "pending_publication_gap": copy.deepcopy(view.publication_gap),
                "required_evidence_numbers": list(view.required_evidence_numbers),
                "research_statuses_unchanged": True,
                "absence_proven": False,
                "instruction": "Close broad discovery, general source-range pagination and new delegation. Reuse delivered originals. Resolve only exact critical operative conditions, exceptions or missing original identities within the remaining closure allowance, using complete provision reads or an owned chunk/reference. Source resolution is available once only for a host-recorded failed named source read: use its complete source title or independently verified own law number, not partial topic terms or pagination. A literal article-heading locator is available only for the same known source/article after its exact read was attempted; generic source-text patterns and regex discovery stay closed. Preserve all supported detail and required evidence. A blocked or exhausted read is unverified, never proof of corpus absence; do not mark an unresolved condition complete. Submit a supported answer with precise unresolved limits through the existing publication guards.",
            }

    def export(self) -> dict[str, JsonValue]:
        self._fence()
        with self._lock:
            state = self._state.model_dump(mode="json")
            return {
                "version": 3,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "policy_hash": self.policy_hash,
                "state": state,
                "state_hash": _digest(state),
            }

    def restore(self, snapshot: dict[str, JsonValue]) -> None:
        self._fence()
        if type(snapshot.get("version")) is not int:
            raise ValueError("Invalid Cropped research checkpoint version")
        saved = _Checkpoint.model_validate(snapshot)
        if (saved.run_id, saved.scope_hash, saved.request_hash, saved.policy_hash) != (
            self.run_id,
            self.scope_hash,
            self.request_hash,
            self.policy_hash,
        ):
            raise ValueError("Cropped research checkpoint identity or policy mismatch")
        if (
            saved.state_hash != _digest(saved.state.model_dump(mode="json"))
            or len(saved.state.critical_reads) > self.policy.critical_reads
        ):
            raise ValueError("Cropped research checkpoint integrity changed")
        with self._lock:
            if self._state != _State() and self._state != saved.state:
                raise ValueError(
                    "Cropped research checkpoint cannot replace live progress"
                )
            self._state = saved.state.model_copy(deep=True)
