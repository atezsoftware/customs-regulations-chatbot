"""Revalidated prior sources remain unread until the new question requests them."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, ToolReceipt
from onyx.regulatory.heading_path import parse_regulatory_article_heading

_SERVICE = "cropped_session_reuse"
_CHECKPOINT = "cropped_source_archive"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _enabled(context: RunContext) -> bool:
    return context.services.get("asv3_exp_cropped") is True and (
        context.services.get("asv3_exp_cheap") is not True
    )


class _ArchiveCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    run_id: str
    scope_hash: str
    request_hash: str
    user_id: str | None
    archived_citations: list[int]
    promoted_citations: list[int]
    originals_hash: str
    integrity: str


class _Binding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    citation: int = Field(ge=1)
    source_id: str
    chunk_id: str | None
    text_hash: str
    metadata_hash: str


def _binding(number: int, ledger: EvidenceLedger) -> _Binding:
    item = ledger.get(number)
    if item is None or hashlib.sha256(item.text.encode()).hexdigest() != item.text_hash:
        raise ValueError("Cropped session archive original changed")
    return _Binding(
        citation=number,
        source_id=item.source_id,
        chunk_id=item.chunk_id,
        text_hash=item.text_hash,
        metadata_hash=_hash(item.metadata),
    )


class CroppedSessionReuse:
    """Fence the host archive without assuming actual model source delivery."""

    def __init__(
        self,
        context: RunContext,
        request: str,
        numbers: Iterable[int],
        ledger: EvidenceLedger,
    ) -> None:
        if not _enabled(context):
            raise ValueError(
                "Lazy session sources require the isolated Cropped workflow"
            )
        selected = list(numbers)
        if any(type(n) is not int or n < 1 for n in selected) or len(
            set(selected)
        ) != len(selected):
            raise ValueError("Cropped session archive citations are invalid")
        self.run_id = context.run_id
        self.scope_hash = _hash(context.scope)
        common_request = context.services.get("cropped_user_context")
        self.request_hash = _hash(
            common_request if isinstance(common_request, str) else request
        )
        user_id = context.services.get("user_id")
        self.user_id = user_id if isinstance(user_id, str) else None
        self._bindings = {n: _binding(n, ledger) for n in sorted(selected)}
        self._promoted: set[int] = set()
        self._lock = threading.RLock()

    def check(self, context: RunContext, ledger: EvidenceLedger) -> None:
        user_id = context.services.get("user_id")
        if not _enabled(context) or (
            context.run_id,
            _hash(context.scope),
            user_id if isinstance(user_id, str) else None,
        ) != (self.run_id, self.scope_hash, self.user_id):
            raise ValueError("Cropped session archive owner or source scope changed")
        request = context.services.get("cropped_user_context")
        if isinstance(request, str) and _hash(request) != self.request_hash:
            raise ValueError("Cropped session archive question changed")
        for number, binding in self._bindings.items():
            if _binding(number, ledger) != binding:
                raise ValueError("Cropped session archive source version changed")

    def record_receipt(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> None:
        self.check(context, ledger)
        actual: set[int] = set()
        for read in receipt.outcome.original_reads:
            item = ledger.get(read.citation)
            if (
                item is None
                or item.text_hash != read.text_hash
                or not (0 <= read.start_char < read.end_char <= len(item.text))
            ):
                raise ValueError("Cropped session requested original range changed")
            actual.add(read.citation)
        identities = {item.identity for item in receipt.outcome.evidence}
        for number in receipt.evidence_ids:
            item = ledger.get(number)
            if item is not None and item.identity in identities:
                actual.add(number)
        with self._lock:
            self._promoted.update(actual & self._bindings.keys())
        self._save(context)

    def active_numbers(self, context: RunContext, ledger: EvidenceLedger) -> list[int]:
        self.check(context, ledger)
        with self._lock:
            # Same-article siblings may have been physically delivered by native fitting.
            for number in self._bindings.keys() - self._promoted:
                deliveries = ledger.inspect(number).get("deliveries")
                if isinstance(deliveries, list) and deliveries:
                    self._promoted.add(number)
            selected = sorted(
                (set(ledger.citation_numbers()) - self._bindings.keys())
                | self._promoted
            )
        self._save(context)
        return selected

    def navigation(self, ledger: EvidenceLedger) -> list[dict[str, JsonValue]]:
        groups: dict[str, dict[str, JsonValue]] = {}
        for row in ledger.provision_metadata():
            number = row.get("citation")
            if type(number) is not int or number not in self._bindings:
                continue
            source = self._bindings[number].source_id
            metadata = row.get("metadata")
            assert isinstance(metadata, dict)
            article, qualifier = metadata.get("article_no"), None
            headings = metadata.get("heading_path")
            for heading in headings if isinstance(headings, list) else []:
                if not isinstance(heading, str):
                    continue
                parsed = parse_regulatory_article_heading(heading)
                if parsed is not None:
                    article, qualifier = parsed.article_no, parsed.qualifier
                    break
            version = {
                key: metadata.get(key)
                for key in (
                    "document_date",
                    "read_as_of_date",
                    "validity_start",
                    "validity_end",
                    "version",
                    "version_unknown",
                    "source_sha256",
                )
            }
            group = groups.setdefault(
                _hash([source, version]),
                {
                    "source_id": source,
                    "title": metadata.get("title"),
                    "document_type": metadata.get("document_type"),
                    "captured_version": version,
                    "articles": [],
                    "unread_this_question": True,
                    "reopen": "read_evidence using this new run's citation numbers",
                },
            )
            articles = cast(list[JsonValue], group["articles"])
            identity = [article, qualifier]
            found = next(
                (
                    article
                    for article in articles
                    if isinstance(article, dict) and article.get("identity") == identity
                ),
                None,
            )
            if found is None:
                found = {"identity": identity, "citations": []}
                articles.append(found)
            cast(list[JsonValue], found["citations"]).append(number)
        return list(groups.values())

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            content: dict[str, JsonValue] = {
                "version": 1,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "user_id": self.user_id,
                "archived_citations": sorted(self._bindings),
                "promoted_citations": sorted(self._promoted),
                "originals_hash": _hash(
                    [
                        binding.model_dump(mode="json")
                        for binding in self._bindings.values()
                    ]
                ),
            }
            return {**content, "integrity": _hash(content)}

    def restore(self, payload: dict[str, JsonValue]) -> None:
        saved = _ArchiveCheckpoint.model_validate(payload)
        own = self.export()
        if (
            type(payload.get("version")) is not int
            or saved.integrity
            != _hash({k: v for k, v in payload.items() if k != "integrity"})
            or any(
                payload.get(key) != own[key]
                for key in (
                    "run_id",
                    "scope_hash",
                    "request_hash",
                    "user_id",
                    "archived_citations",
                    "originals_hash",
                )
            )
            or any(type(n) is not int for n in saved.promoted_citations)
            or len(set(saved.promoted_citations)) != len(saved.promoted_citations)
            or set(saved.promoted_citations) - self._bindings.keys()
        ):
            raise ValueError("Cropped session archive checkpoint changed")
        with self._lock:
            self._promoted = set(saved.promoted_citations)

    def _save(self, context: RunContext) -> None:
        memory = context.services.get("session_research")
        if isinstance(memory, dict):
            with self._lock:
                memory[_CHECKPOINT] = self.export()


def install_cropped_session_reuse(
    context: RunContext,
    ledger: EvidenceLedger,
    request: str,
    *,
    previous: dict[str, JsonValue] | None = None,
) -> CroppedSessionReuse | None:
    """Called after existing ACL/version revalidation, never instead of it."""
    if not _enabled(context):
        return None
    memory = context.services.get("session_research")
    if not isinstance(memory, dict):
        return None
    saved = memory.get(_CHECKPOINT)
    numbers = (
        saved.get("archived_citations")
        if isinstance(saved, dict)
        else memory.get("reused_evidence_numbers")
    )
    if not isinstance(numbers, list) or not numbers:
        return None
    archive = CroppedSessionReuse(context, request, cast(list[int], numbers), ledger)
    if isinstance(saved, dict):
        archive.restore(saved)
    context.services[_SERVICE] = archive
    memory["prior_source_navigation"] = archive.navigation(ledger)
    if previous is not None:
        old_evidence = previous.get("evidence")
        old_rows = (
            old_evidence.get("records") if isinstance(old_evidence, dict) else None
        )
        current = {
            (row.source_id, row.chunk_id, row.text_hash): number
            for number, row in archive._bindings.items()
        }
        mapping: list[JsonValue] = []
        for row in old_rows if isinstance(old_rows, list) else []:
            if not isinstance(row, dict) or type(row.get("citation")) is not int:
                continue
            try:
                item = EvidenceItem.model_validate(row.get("item"))
            except ValueError:
                continue
            if (number := current.get(item.identity)) is not None:
                mapping.append([row["citation"], number])
        memory["prior_citation_map"] = mapping
    memory["source_reuse_notice"] = (
        "Earlier originals remain revalidated in the authorized host archive, unread by "
        "this question until a requested source action delivers them. Source/article cards "
        "are navigation, not support, complete article closure or evidence of absence. "
        "Keep the actual prior user case facts. The full labelled conversation remains "
        "available; prior assistant assertions are model assessments, not new user facts. "
        "Read the new run's citation with read_evidence, or request the exact canonical "
        "article using read_provision/read_named_provision. Earlier citation numbering "
        "may differ; do not assume a prior answer's citation is this run's source."
    )
    archive._save(context)
    return archive


def _archive(context: RunContext) -> CroppedSessionReuse | None:
    if not _enabled(context):
        return None
    value = context.services.get(_SERVICE)
    if value is not None and not isinstance(value, CroppedSessionReuse):
        raise ValueError("Cropped session archive service is invalid")
    return value


def record_cropped_session_receipt(
    receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
) -> None:
    if (archive := _archive(context)) is not None:
        archive.record_receipt(receipt, context, ledger)


def active_cropped_session_numbers(
    ledger: EvidenceLedger, context: RunContext
) -> list[int]:
    archive = _archive(context)
    return (
        archive.active_numbers(context, ledger)
        if archive
        else list(ledger.citation_numbers())
    )


def session_numbers_to_remember(
    numbers: list[JsonValue], context: RunContext, ledger: EvidenceLedger
) -> list[JsonValue]:
    if (archive := _archive(context)) is None:
        return numbers
    active = set(archive.active_numbers(context, ledger))
    return [number for number in numbers if type(number) is int and number in active]
