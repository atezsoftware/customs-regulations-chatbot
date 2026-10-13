"""Fresh repair views retain complete originals and the current research state."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.models import HarnessView

_TERMINAL_NAMES = frozenset(
    {
        "submit_answer",
        "submit_partial_answer",
        "submit_retained_answer",
        "submit_retained_partial_answer",
    }
)
_DRAFT_ARGUMENTS = frozenset({"answer", "retained_answer_id", "retained_answer_edits"})


def cheap_repair_observations(view: HarnessView) -> list[dict[str, JsonValue]]:
    """Retain exact method results as untrusted data, outside native tool protocol.

    Literal originals are transported in the validated complete packet. The owned
    current draft replaces prior terminal bodies; every other method/result survives.
    """
    observations: list[dict[str, JsonValue]] = []
    for turn in view.turns:
        for index, (call, result) in enumerate(
            zip(turn.assistant.tool_calls or [], turn.results, strict=True)
        ):
            payload = json.loads(result.content)
            if not isinstance(payload, dict):
                raise ValueError("Fresh repair observations require structured results")
            payload.pop("original_evidence", None)
            arguments = call.function.arguments
            if call.function.name in _TERMINAL_NAMES:
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("Retained terminal arguments must be an object")
                arguments = json.dumps(
                    {
                        key: value
                        for key, value in parsed.items()
                        if key not in _DRAFT_ARGUMENTS
                    },
                    ensure_ascii=False,
                )
            observation: dict[str, JsonValue] = {
                "name": call.function.name,
                "arguments_json": arguments,
                "result": payload,
            }
            if index == 0 and turn.assistant.content:
                observation["untrusted_assistant_assessment"] = (
                    turn.assistant.model_dump(mode="json")["content"]
                )
            observations.append(observation)
    return observations


def _complete_original(record: dict[str, JsonValue]) -> bool:
    number, text, digest = (
        record.get("citation"),
        record.get("text"),
        record.get("text_hash"),
    )
    return (
        type(number) is int
        and number > 0
        and isinstance(text, str)
        and bool(text)
        and isinstance(digest, str)
        and hashlib.sha256(text.encode()).hexdigest() == digest
        and all(
            isinstance(record.get(key), str) and bool(record.get(key))
            for key in ("source_id", "chunk_id")
        )
        and record.get("truncated") is False
        and type(record.get("start_char", 0)) is int
        and record.get("start_char", 0) == 0
        and type(record.get("end_char", len(text))) is int
        and record.get("end_char", len(text)) == len(text)
    )


def _prior_passages(view: HarnessView) -> list[dict[str, JsonValue]] | None:
    passages = list(view.original_evidence)
    for turn in view.turns:
        for call in turn.assistant.tool_calls or []:
            if call.function.name in _TERMINAL_NAMES:
                try:
                    arguments = json.loads(call.function.arguments)
                except (TypeError, ValueError):
                    return None
                if not isinstance(arguments, dict):
                    return None
        for result in turn.results:
            try:
                payload = json.loads(result.content)
            except (TypeError, ValueError):
                return None
            if not isinstance(payload, dict):
                return None
            originals = payload.get("original_evidence", [])
            if not isinstance(originals, list) or any(
                not isinstance(row, dict) for row in originals
            ):
                return None
            passages.extend(originals)
    return passages


def _passage_retained(
    passage: dict[str, JsonValue], original: dict[str, JsonValue]
) -> bool:
    text, start = passage.get("text"), passage.get("start_char", 0)
    full_text = original["text"]
    if not isinstance(text, str) or type(start) is not int or start < 0:
        return False
    if not isinstance(full_text, str) or full_text[start : start + len(text)] != text:
        return False
    if passage.get("end_char", start + len(text)) != start + len(text):
        return False
    if "identity_ref" in passage and passage["identity_ref"] != original["citation"]:
        return False
    reference = passage.get("metadata_ref")
    if reference is not None and (
        not isinstance(reference, dict)
        or reference.get("citation") != original["citation"]
        or reference.get("text_hash") != original["text_hash"]
    ):
        return False
    return all(
        key not in passage or passage[key] == original.get(key)
        for key in ("source_id", "chunk_id", "text_hash", "citable", "metadata")
    )


def prepare_cheap_repair_view(
    view: HarnessView, originals: Sequence[dict[str, JsonValue]]
) -> HarnessView | None:
    """Remove native turns only when every prior passage and draft citation survives.

    This changes context, not evidence ownership. The caller must serialize the returned
    view through its normal fit/invoke path and record actual new-call deliveries.
    No source subset is inferred from host publication flags or provision navigation.
    """
    if not view.draft_to_repair or view.publication_gap is None or not originals:
        return None
    by_number: dict[int, dict[str, JsonValue]] = {}
    for record in originals:
        if not _complete_original(record):
            return None
        number = record["citation"]
        assert isinstance(number, int)
        previous = by_number.get(number)
        if previous is not None and previous != record:
            return None
        by_number[number] = record
    required = set(view.required_evidence_numbers)
    required.update(extract_citation_numbers(view.draft_to_repair))
    if not required.issubset(by_number) or any(
        by_number[number].get("citable") is not True for number in required
    ):
        return None
    prior = _prior_passages(view)
    if prior is None:
        return None
    for passage in prior:
        number = passage.get("citation")
        if (
            type(number) is not int
            or number not in by_number
            or not _passage_retained(passage, by_number[number])
        ):
            return None
    prepared = view.model_copy(deep=True)
    prepared.turns = []
    prepared.original_evidence = copy.deepcopy(list(by_number.values()))
    prepared.required_evidence_numbers = list(
        dict.fromkeys([*view.required_evidence_numbers, *by_number])
    )
    return prepared
