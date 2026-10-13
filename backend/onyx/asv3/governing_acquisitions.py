"""Request-bound governing-original acquisition episodes for tuned ASv3."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.asv3.authority import folded, known_statute_source_ids
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext, ToolReceipt


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _text(value: JsonValue) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


class GoverningOriginalTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instrument_number: str | None = None
    formal_name: str | None = None
    article: str | None = None
    qualifier: str | None = None
    paragraph: str | None = None
    clause: str | None = None

    @property
    def key(self) -> str:
        return _hash(self.model_dump(mode="json"))

    @property
    def source_name(self) -> str:
        return (
            f"{self.instrument_number} sayılı Kanun"
            if self.instrument_number
            else self.formal_name or ""
        )

    @property
    def article_query(self) -> str:
        return " ".join(filter(None, (self.qualifier, self.article)))

    @property
    def reference_query(self) -> str:
        return self.article_query + (f"/{self.paragraph}" if self.paragraph else "")

    def provision_arguments(self) -> dict[str, JsonValue]:
        return {
            "article": self.article_query,
            **({"paragraph": self.paragraph} if self.paragraph else {}),
            **({"clause": self.clause} if self.clause else {}),
        }


def governing_original_targets(
    gap: dict[str, JsonValue],
) -> list[GoverningOriginalTarget]:
    """Merge the two guard kinds without making prose or unit IDs retry keys."""
    targets: dict[str, GoverningOriginalTarget] = {}
    for field in ("named_authority_gaps", "unread_cited_statute_references"):
        rows = gap.get(field, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Invalid governing-original gap")
        for row in rows:
            assert isinstance(row, dict)
            matching = row.get("matching_original_evidence", [])
            if isinstance(matching, list) and matching:
                continue
            number = _text(row.get("instrument_number"))
            formal = _text(row.get("formal_name"))
            if not number and not formal:
                continue
            article = _text(row.get("article", row.get("article_no")))
            paragraph = row.get("paragraph")
            paragraph_text = (
                str(paragraph)
                if type(paragraph) is int and paragraph > 0
                else _text(paragraph)
            )
            split = re.fullmatch(r"(.+?)/(\d+)", article.strip()) if article else None
            if split is not None:
                article = split[1]
                if paragraph_text and paragraph_text != split[2]:
                    raise ValueError("Conflicting governing paragraph identity")
                paragraph_text = split[2]
            target = GoverningOriginalTarget(
                instrument_number=number,
                formal_name=folded(formal) if formal and not number else None,
                article=re.sub(r"\s+", "", article).upper() if article else None,
                qualifier=folded(value)
                if (value := _text(row.get("qualifier", row.get("article_qualifier"))))
                else None,
                paragraph=paragraph_text,
                clause=_text(row.get("clause", row.get("clause_label"))),
            )
            targets.setdefault(target.key, target)
    return list(targets.values())


class _Episode(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target: GoverningOriginalTarget
    owner: str = Field(min_length=1, max_length=512)
    status: Literal["in_progress", "available", "unavailable", "failed"]
    active_call: CapabilityCall | None = None
    next_action: dict[str, JsonValue] | None = None
    evidence_numbers: list[int] = Field(default_factory=list)
    observed_calls: list[str] = Field(default_factory=list)
    detail: str = ""

    @model_validator(mode="after")
    def consistent(self) -> _Episode:
        if self.active_call is not None and self.next_action is not None:
            raise ValueError("Governing episode cannot have two pending actions")
        if self.status != "in_progress" and (
            self.active_call is not None or self.next_action is not None
        ):
            raise ValueError("Completed governing episode contains a pending action")
        if self.status == "available" and not self.evidence_numbers:
            raise ValueError("Available governing episode needs actual originals")
        if len(self.observed_calls) != len(set(self.observed_calls)) or any(
            not value for value in self.observed_calls
        ):
            raise ValueError("Governing episode has duplicate actual receipt IDs")
        if len(self.evidence_numbers) != len(set(self.evidence_numbers)) or any(
            type(value) is not int or value < 1 for value in self.evidence_numbers
        ):
            raise ValueError("Governing episode original references changed")
        if self.next_action is not None and (
            set(self.next_action) != {"name", "arguments"}
            or self.next_action["name"] not in {"read_provision", "read_source_range"}
            or not isinstance(self.next_action["arguments"], dict)
        ):
            raise ValueError("Invalid governing acquisition continuation")
        return self


class GoverningOriginalAcquisitions:
    def __init__(self, context: RunContext, request: str) -> None:
        self.context = context
        self.run_id = context.run_id
        self.scope_hash = _hash(context.scope)
        self.request_hash = _hash(request)
        self._lock = threading.RLock()
        self._episodes: dict[str, _Episode] = {}

    def check_context(self, context: RunContext) -> None:
        if (
            context.run_id != self.run_id
            or _hash(context.scope) != self.scope_hash
            or _hash(self.context.scope) != self.scope_hash
        ):
            raise ValueError("Governing-original acquisition scope changed")

    @staticmethod
    def matching_originals(
        target: GoverningOriginalTarget, ledger: EvidenceLedger
    ) -> list[int]:
        from onyx.asv3.canonical_authority import canonical_matching_originals

        probe = (
            f"{target.source_name} "
            + (f"{target.reference_query}. maddesi " if target.article else "")
            + (f"({target.clause}) bendi " if target.clause else "")
            + "uygulanır "
            + "."
        )
        return canonical_matching_originals(probe, ledger)

    def next_call(
        self,
        target: GoverningOriginalTarget,
        context: RunContext,
        ledger: EvidenceLedger,
    ) -> CapabilityCall | None:
        self.check_context(context)
        matching = self.matching_originals(target, ledger)
        owner = str(context.services.get("task_id") or "root")
        with self._lock:
            episode = self._episodes.get(target.key)
            if matching and not (
                episode is not None
                and episode.status == "in_progress"
                and (episode.active_call is not None or episode.next_action is not None)
            ):
                if episode is not None:
                    episode.status = "available"
                    episode.evidence_numbers = matching
                    episode.active_call = None
                    episode.next_action = None
                return None
            if episode is not None:
                if episode.status != "in_progress" or episode.owner != owner:
                    return None
                if episode.active_call is not None or episode.next_action is None:
                    return None
                action = episode.next_action
            else:
                source_ids = known_statute_source_ids(target.source_name, ledger)
                if target.article:
                    action = {
                        "name": "read_provision"
                        if len(source_ids) == 1
                        else "read_named_provision",
                        "arguments": {
                            **target.provision_arguments(),
                            "source_id"
                            if len(source_ids) == 1
                            else "source_name": source_ids[0]
                            if len(source_ids) == 1
                            else target.source_name,
                        },
                    }
                else:
                    action = {
                        "name": "resolve_source",
                        "arguments": {"query": target.source_name},
                    }
            return CapabilityCall(
                name=str(action["name"]),
                arguments=cast(
                    dict[str, JsonValue], copy.deepcopy(action["arguments"])
                ),
            )

    def reserve(
        self, target: GoverningOriginalTarget, call: CapabilityCall, context: RunContext
    ) -> bool:
        self.check_context(context)
        owner = str(context.services.get("task_id") or "root")
        with self._lock:
            episode = self._episodes.get(target.key)
            if episode is None:
                episode = _Episode(target=target, owner=owner, status="in_progress")
                self._episodes[target.key] = episode
            if (
                episode.status != "in_progress"
                or episode.owner != owner
                or episode.active_call is not None
            ):
                return False
            episode.active_call = call.model_copy(deep=True)
            episode.next_action = None
            return True

    def record_receipt(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> bool:
        self.check_context(context)
        with self._lock:
            episode = next(
                (
                    value
                    for value in self._episodes.values()
                    if value.active_call is not None
                    and value.active_call.call_id == receipt.call.call_id
                ),
                None,
            )
            if episode is None:
                return False
            if episode.active_call != receipt.call:
                raise ValueError("Governing acquisition actual call changed")
            episode.observed_calls.append(receipt.call.call_id)
            episode.active_call = None
            episode.evidence_numbers = list(
                dict.fromkeys([*episode.evidence_numbers, *receipt.evidence_ids])
            )
            data = receipt.outcome.data
            if receipt.call.name == "resolve_source":
                sources = data.get("sources")
                resolved_source_id = (
                    sources[0].get("source_id")
                    if isinstance(sources, list)
                    and len(sources) == 1
                    and isinstance(sources[0], dict)
                    else None
                )
                if (
                    receipt.outcome.status == OutcomeStatus.FOUND
                    and isinstance(sources, list)
                    and len(sources) == 1
                    and isinstance(sources[0], dict)
                    and isinstance(resolved_source_id, str)
                    and data.get("has_more") is False
                ):
                    episode.next_action = {
                        "name": "read_source_range",
                        "arguments": {"source_id": resolved_source_id},
                    }
                    return True
            elif receipt.outcome.status in {
                OutcomeStatus.FOUND,
                OutcomeStatus.PARTIAL,
            }:
                more = any(
                    data.get(field) is True
                    for field in ("scan_truncated", "evidence_truncated", "has_more")
                )
                if not more and self.matching_originals(episode.target, ledger):
                    episode.status = "available"
                    return True
                position = (
                    data.get("evidence_next_position")
                    if data.get("evidence_truncated") is True
                    else data.get("next_position")
                )
                source_id = receipt.call.arguments.get(
                    "source_id", data.get("source_id")
                )
                previous = receipt.call.arguments.get("start", 0)
                if (
                    more
                    and type(position) is int
                    and type(previous) is int
                    and position > previous
                    and isinstance(source_id, str)
                ):
                    episode.next_action = {
                        "name": "read_provision"
                        if episode.target.article
                        else "read_source_range",
                        "arguments": {
                            "source_id": source_id,
                            "start": position,
                            **(
                                episode.target.provision_arguments()
                                if episode.target.article
                                else {}
                            ),
                        },
                    }
                    return True
            episode.status = (
                "unavailable"
                if receipt.outcome.status
                in {
                    OutcomeStatus.NOT_FOUND,
                    OutcomeStatus.AMBIGUOUS,
                    OutcomeStatus.FOUND,
                    OutcomeStatus.PARTIAL,
                }
                else "failed"
            )
            episode.detail = (
                f"Actual {receipt.call.name} outcome: {receipt.outcome.status.value}. "
                "The governing original was not verified by this acquisition episode; "
                "source absence and legal effect are not established."
            )
            return True

    def unavailable_keys(
        self, gap: dict[str, JsonValue], ledger: EvidenceLedger
    ) -> set[str]:
        targets = governing_original_targets(gap)
        missing = {
            target.key
            for target in targets
            if not self.matching_originals(target, ledger)
        }
        with self._lock:
            return {
                key
                for key in missing
                if (episode := self._episodes.get(key)) is not None
                and episode.status in {"unavailable", "failed"}
            }

    def records_for(
        self, targets: list[GoverningOriginalTarget]
    ) -> list[dict[str, JsonValue]]:
        with self._lock:
            return [
                {
                    "key": target.key,
                    "target": target.model_dump(mode="json"),
                    "status": episode.status,
                    "candidate_original_evidence_numbers": list(
                        episode.evidence_numbers
                    ),
                    "actual_call_ids": list(episode.observed_calls),
                    "detail": episode.detail,
                    "absence_proven": False,
                    "legal_approval": False,
                    "source_gap_notice": (
                        f"{target.source_name} {target.reference_query}. maddesinin özgün metni elde edilemedi."
                        if target.article
                        else f"{target.source_name} özgün metni elde edilemedi."
                    )
                    if episode.status in {"unavailable", "failed"}
                    else None,
                }
                for target in targets
                if (episode := self._episodes.get(target.key)) is not None
            ]

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            episodes = {
                key: value.model_dump(mode="json")
                for key, value in self._episodes.items()
            }
            return {
                "version": 1,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "episodes": cast(JsonValue, episodes),
                "episodes_hash": _hash(episodes),
            }

    def restore(self, payload: dict[str, JsonValue]) -> None:
        if (
            type(payload.get("version")) is not int
            or set(payload)
            != {
                "version",
                "run_id",
                "scope_hash",
                "request_hash",
                "episodes",
                "episodes_hash",
            }
            or any(
                payload.get(field) != self.export()[field]
                for field in ("version", "run_id", "scope_hash", "request_hash")
            )
        ):
            raise ValueError("Governing acquisition checkpoint identity changed")
        episodes = payload["episodes"]
        if not isinstance(episodes, dict) or payload["episodes_hash"] != _hash(
            episodes
        ):
            raise ValueError("Governing acquisition checkpoint integrity changed")
        restored = {
            key: _Episode.model_validate(value) for key, value in episodes.items()
        }
        if any(key != value.target.key for key, value in restored.items()):
            raise ValueError("Governing acquisition target identity changed")
        # A durable reservation never becomes another lookup after interruption.
        for episode in restored.values():
            if episode.active_call is not None:
                episode.active_call = None
                episode.next_action = None
                episode.status = "failed"
                episode.detail = "Acquisition interrupted before an actual result; no verified finding or proven absence."
        with self._lock:
            if self._episodes:
                raise ValueError("Cannot replace live governing acquisition episodes")
            self._episodes = restored
