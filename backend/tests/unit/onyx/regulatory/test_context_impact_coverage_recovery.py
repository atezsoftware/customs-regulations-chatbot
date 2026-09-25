"""Coverage recovery must retain exact keys, proofs, deadlines and checkpoints."""

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from datetime import date
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from onyx.llm.interfaces import LLM, LLMConfig
from onyx.regulatory import amendment_projection_impact as impact
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
    contextual_model_fingerprint,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)
from tests.unit.onyx.regulatory.annexes.test_publication_timeline import snapshot


def inputs(
    count: int = 3,
) -> tuple[
    list[AnnexCanonicalSnapshot],
    list[AnnexCanonicalSnapshot],
    list[AnnexTemporalProjection],
    LLM,
]:
    old = snapshot("old", None, None).model_copy(update={"text": "Fee is 5%."})
    consumers = [snapshot(f"consumer-{i}", None, None) for i in range(count)]
    before = [old, *consumers]
    after = [
        old.model_copy(update={"validity_end_date": date(2031, 1, 1)}),
        *consumers,
        snapshot("new", date(2031, 1, 1), None).model_copy(
            update={"text": "Fee is 7%.", "supersedes_chunk_id": "old"}
        ),
    ]
    bindings = [
        MagicMock(
            effective_start=None,
            effective_end=None,
            projection=MagicMock(
                source_json=json.dumps(
                    {
                        "regulatory_chunk_id": row.id,
                        "chunk_context": f"Context {i}: Fee is 5%.",
                    }
                )
            ),
        )
        for i, row in enumerate(consumers)
    ]
    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=10000,
    )
    return before, after, cast(list[AnnexTemporalProjection], bindings), llm


def decisions(keys: list[str]) -> impact.ContextImpactResult:
    return impact.ContextImpactResult(
        decisions=[
            impact.ContextImpactDecision(
                key=key,
                affected=True,
                quote="Fee is 5%.",
                reason="The fee changed.",
                source_side="after",
                source_id="new",
                source_quote="Fee is 7%.",
            )
            for key in keys
        ]
    )


def run(
    data: tuple[
        list[AnnexCanonicalSnapshot],
        list[AnnexCanonicalSnapshot],
        list[AnnexTemporalProjection],
        LLM,
    ],
    cache: impact.ContextAuditResolver | None = None,
) -> dict[str, list[impact.Window]]:
    before, after, bindings, llm = data
    return impact.include_context_consumers(
        impact.structural_windows(before, after),
        before=before,
        after=after,
        bindings=bindings,
        llm=llm,
        audit_cache=cache,
    )


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign"])
def test_persistent_batch_coverage_recovers_with_original_singleton_keys(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    requests: list[dict[str, Any]] = []
    deadlines: list[object] = []

    def audit(*_args: object, **kwargs: object) -> impact.ContextImpactResult:
        data = json.loads(str(kwargs["user_prompt"]))
        requests.append(data)
        deadlines.append(kwargs["deadline"])
        keys = list(data["contexts"])
        if len(keys) == 1:
            assert kwargs["max_attempts"] == 1
            assert kwargs["provider_max_attempts"] == 1
        if len(keys) > 1:
            keys = keys[:-1]
            if fault == "duplicate":
                keys.append(keys[0])
            elif fault == "foreign":
                keys.append("foreign-key")
        return decisions(keys)

    monkeypatch.setattr(impact, "generate_structured", audit)
    result = run(inputs())
    assert set(result) == {"old", "new", "consumer-0", "consumer-1", "consumer-2"}
    assert [list(request["contexts"]) for request in requests] == [
        ["0", "1", "2"],
        ["0", "1", "2"],
        ["0"],
        ["1"],
        ["2"],
    ]
    assert len(set(deadlines)) == 1
    for request in requests[2:]:
        key = next(iter(request["contexts"]))
        assert request["contexts"][key] == requests[0]["contexts"][key]
        assert request["changes"] == requests[0]["changes"]
    feedback = requests[1]["validation_feedback"]
    assert feedback["coverage"]["expected_keys"] == ["0", "1", "2"]
    assert feedback["coverage"]["missing_keys"] == ["2"]
    assert feedback["coverage"]["duplicate_keys"] == (
        ["0"] if fault == "duplicate" else []
    )
    assert feedback["coverage"]["unexpected_keys"] == (
        ["sha256:" + hashlib.sha256(b"foreign-key").hexdigest()[:12]]
        if fault == "foreign"
        else []
    )


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "foreign", "source_quote", "uncertain"]
)
def test_invalid_singleton_never_becomes_unchanged_or_gets_remapped(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    calls: list[list[str]] = []

    def audit(*_args: object, **kwargs: object) -> impact.ContextImpactResult:
        keys = list(json.loads(str(kwargs["user_prompt"]))["contexts"])
        calls.append(keys)
        if len(keys) > 1:
            return decisions(keys[:-1])
        if keys == ["1"]:
            if fault in {"missing", "duplicate", "foreign"}:
                return decisions(
                    []
                    if fault == "missing"
                    else ["1", "1"]
                    if fault == "duplicate"
                    else ["0"]
                )
            response = decisions(keys)
            response.decisions[0] = response.decisions[0].model_copy(
                update={"source_quote": "fabricated"}
                if fault == "source_quote"
                else {"uncertain": True}
            )
            return response
        return decisions(keys)

    monkeypatch.setattr(impact, "generate_structured", audit)
    with pytest.raises(ValueError):
        run(inputs())
    assert calls == [["0", "1", "2"], ["0", "1", "2"], ["0"], ["1"]]


def test_expired_parent_deadline_prevents_later_singleton_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    calls: list[list[str]] = []

    def audit(*_args: object, **kwargs: object) -> impact.ContextImpactResult:
        keys = list(json.loads(str(kwargs["user_prompt"]))["contexts"])
        calls.append(keys)
        assert kwargs["deadline"] == 190.0
        if len(keys) > 1:
            return decisions(keys[:-1])
        now[0] = 191.0
        return decisions(keys)

    monkeypatch.setattr(impact, "monotonic", lambda: now[0])
    monkeypatch.setattr(impact, "generate_structured", audit)
    with pytest.raises(TimeoutError, match="deadline"):
        run(inputs())
    assert calls == [["0", "1", "2"], ["0", "1", "2"], ["0"]]


@pytest.mark.parametrize("fault", ["source_quote", "uncertain"])
def test_incomplete_batch_with_semantic_error_does_not_trigger_coverage_fallback(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    calls: list[list[str]] = []

    def audit(*_args: object, **kwargs: object) -> impact.ContextImpactResult:
        keys = list(json.loads(str(kwargs["user_prompt"]))["contexts"])
        calls.append(keys)
        response = decisions(keys[:-1] if len(keys) > 1 else keys)
        if len(keys) > 1:
            response.decisions[0] = response.decisions[0].model_copy(
                update={"source_quote": "fabricated"}
                if fault == "source_quote"
                else {"uncertain": True}
            )
        return response

    monkeypatch.setattr(impact, "generate_structured", audit)
    with pytest.raises(ValueError, match="changed source|unresolved context impact"):
        run(inputs())
    assert calls == [["0", "1", "2"], ["0", "1", "2"]]


def test_existing_parent_cache_and_successful_singletons_resume_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = inputs(10)
    cache: dict[str, impact.ContextImpactResult] = {}
    calls: Counter[tuple[str, ...]] = Counter()
    blocked = [True]

    def resolve(
        key: str, generate: Callable[[], impact.ContextImpactResult]
    ) -> impact.ContextImpactResult:
        if key not in cache:
            cache[key] = generate()
        return cache[key]

    def audit(*_args: object, **kwargs: object) -> impact.ContextImpactResult:
        request = json.loads(str(kwargs["user_prompt"]))
        keys = tuple(request["contexts"])
        calls[keys] += 1
        if keys == tuple(map(str, range(8))):
            # The old successful group key remains compatible with the checkpoint.
            old_key = context_hash(
                [
                    request["contexts"],
                    request["changes"],
                    contextual_model_fingerprint(data[3]),
                    impact.CONTEXT_IMPACT_AUDIT,
                ]
            )
            cache[old_key] = decisions(list(keys))
            return cache[old_key]
        if len(keys) > 1 or blocked[0] and keys == ("9",):
            return decisions(list(keys[:-1]))
        return decisions(list(keys))

    monkeypatch.setattr(impact, "generate_structured", audit)
    with pytest.raises(ValueError):
        run(data, resolve)
    assert calls[("8",)] == 1
    assert len(cache) == 2  # Completed parent and singleton; never incomplete output.
    blocked[0] = False
    result = run(data, resolve)
    assert len(result) == 12
    assert calls[tuple(map(str, range(8)))] == 1
    assert calls[("8",)] == 1
    assert calls[("9",)] == 2
    previous = calls.copy()
    assert run(data, resolve) == result
    assert calls == previous
