"""Keep acquisition safeguards without compulsory semantic review generations."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import runtime
from onyx.asv3.models import OutcomeStatus, ToolOutcome
from onyx.asv3.publication_gaps import combine_source_publication_gaps
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.interfaces import LLM
from tests.unit.onyx.asv3.test_runtime import delivered_originals, response, setup_run


def test_tuned_runtime_delivers_all_originals_without_automatic_review_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    selected.config = selected.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.7-flash"}
    )
    runtime.run_asv3_loop(**kwargs)

    # The real native dispatcher reads, answers, validates and publishes once.
    assert selected.invoke.call_count == 2
    originals = delivered_originals(selected.invoke.call_args.kwargs)
    assert {row["text"] for row in originals} == {
        chunk.text for chunk in broker.chunks.values()
    }
    assert {item.text for item in broker.revalidated} == {
        chunk.text for chunk in broker.chunks.values()
    }
    for invocation in selected.invoke.call_args_list:
        schema = (invocation.kwargs.get("structured_response_format") or {}).get(
            "json_schema", {}
        )
        assert schema.get("name") not in {"SourceUseInventory", "SourceUseReview"}
    assert kwargs["state_container"].answer_tokens == (
        "Tamir sonucu [[1]](https://example.test/law-0); "
        "değiştirme sonucu [[2]](https://example.test/law-1)."
    )


def test_source_answer_runtime_keeps_complete_delivery_and_dynamic_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, research, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    research.config = research.config.model_copy(
        update={
            "model_provider": "vertex_ai",
            "model_name": "gemini-3.8-flash",
            "temperature": 0.1,
        }
    )
    research.with_temperature.return_value = research
    writer = MagicMock(spec=LLM)
    writer.config = research.config.model_copy(
        update={
            "model_provider": "anthropic",
            "model_name": "claude-sonnet-5-5",
            "temperature": 1,
        }
    )
    writer.invoke.return_value = response("Tamir sonucu [1]; değiştirme sonucu [2].")
    resolver = MagicMock(return_value=writer)
    monkeypatch.setattr(runtime, "source_answer_model", resolver)
    runtime.run_asv3_loop(**kwargs)
    resolver.assert_called_once_with(kwargs["user"])
    research.with_temperature.assert_called_once_with(0.1)
    assert research.invoke.call_count == 2
    assert writer.invoke.call_count == 1
    assert {
        row["text"] for row in delivered_originals(writer.invoke.call_args.kwargs)
    } == {chunk.text for chunk in broker.chunks.values()}
    assert {item.text for item in broker.revalidated} == {
        chunk.text for chunk in broker.chunks.values()
    }
    assert kwargs["state_container"].answer_tokens == (
        "Tamir sonucu [[1]](https://example.test/law-0); "
        "değiştirme sonucu [[2]](https://example.test/law-1)."
    )
    assert (
        checkpoints[-1]["native_coordinator_sampling"]["settings"][
            "source_answer_model"
        ]["model"]
        == "claude-sonnet-5-5"
    )


def test_batched_publication_gaps_preserve_all_acquisition_bindings() -> None:
    gaps = [
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Named source",
            data={
                "named_authority_gaps": [{"unit_id": "first"}],
                "instruction": "Use its own original",
            },
        ),
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Referenced original",
            data={
                "unread_cited_statute_references": [{"article": "7"}],
                "instruction": "Read the material reference",
            },
        ),
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Related source",
            data={
                "pending_related_source_review": True,
                "unread_related_sources": [{"lead_id": "known"}],
            },
        ),
    ]
    combined = combine_source_publication_gaps(gaps)
    assert combined and combined.status == OutcomeStatus.PARTIAL
    for gap in gaps:
        for key, value in gap.data.items():
            if key != "instruction":
                assert combined.data[key] == value
    assert len(cast(list[Any], combined.data["publication_check_instructions"])) == 3


def test_conflicting_acquisition_receipts_cannot_be_silently_overwritten() -> None:
    gaps = [
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Read the original",
            data={"unread": [number]},
        )
        for number in (1, 2)
    ]
    with pytest.raises(ValueError, match="Conflicting"):
        combine_source_publication_gaps(gaps)
