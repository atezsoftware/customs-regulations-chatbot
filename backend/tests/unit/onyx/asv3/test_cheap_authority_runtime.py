"""Cheap must remember a rejected named basis before citation delivery fails."""

import json
from typing import Any

import pytest

from onyx.asv3 import runtime
from onyx.asv3.models import RunStopped
from onyx.asv3.workflow_variant import ASV3_EXP_CHEAP_VARIANT, ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_runtime import response, setup_run, user_payload

pytestmark = pytest.mark.usefixtures("empty_source_inventory")

NAMED = "5326 sayılı Kabahatler Kanunu m.15/3 uyarınca yaptırım uygulanır [99]."
PARAPHRASED = "Bu fiil için yaptırım uygulanır [1] ve [2]."


@pytest.mark.parametrize("cheap", [True, False])
def test_early_unknown_citation_preserves_cheap_authority_dependency_only(
    monkeypatch: pytest.MonkeyPatch, cheap: bool
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(
        research_profile="normal",
        workflow_variant=ASV3_EXP_CHEAP_VARIANT if cheap else ASV3_TUNED_VARIANT,
    )
    first = selected.invoke.side_effect
    calls = 0
    baseline_checked_answers: list[str] = []
    baseline_named_gap = runtime.native_named_authority_gap

    def record_baseline(answer: str, *arguments: Any, **options: Any):
        baseline_checked_answers.append(answer)
        return baseline_named_gap(answer, *arguments, **options)

    monkeypatch.setattr(runtime, "native_named_authority_gap", record_baseline)

    def invoke(**arguments: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first(**arguments)
        if calls == 2:
            return response(NAMED)
        if cheap and calls == 3:
            payload = user_payload(arguments["prompt"][-1])
            assert payload["required_evidence_numbers"] == [1, 2]
            assert "answer_units" not in payload
            return response(
                json.dumps(
                    {
                        "source_scope_id": payload["source_scope_id"],
                        "whole_scope_completed": True,
                        "requirements": [],
                    }
                )
            )
        if calls == (4 if cheap else 3):
            payload = user_payload(arguments["prompt"][-1])
            if cheap:
                requirements = payload["governing_source_requirements"][
                    "retained_authority_requirements"
                ]
                assert len(requirements) == 1
                assert requirements[0]["instrument_number"] == "5326"
                assert requirements[0]["article"] == "15"
            else:
                assert "governing_source_requirements" not in payload
            return response(PARAPHRASED)
        assert cheap and calls == 5
        # The paraphrase has been rejected and saved; do not start another fake call.
        broker.cancelled.set()
        raise RunStopped("Research cancelled")

    selected.invoke.side_effect = invoke
    if cheap:
        with pytest.raises(RunStopped, match="cancelled"):
            runtime.run_asv3_loop(**kwargs)
        assert calls == 5
        assert not kwargs["state_container"].answer_tokens
        rejected = next(
            saved
            for saved in reversed(checkpoints)
            if saved.get("last_draft") == PARAPHRASED
            and saved.get("final_publication_gap")
        )
        gap = rejected["final_publication_gap"]["data"]
        assert gap["retained_authority_requirements"][0]["article"] == "15"
        policy = rejected["cheap_authority_policy"]
        assert policy["requirements"]["records"][0]["instrument_number"] == "5326"
    else:
        runtime.run_asv3_loop(**kwargs)
        assert calls == 3
        assert kwargs["state_container"].answer_tokens
        # The original ordering rejects [99] before checking named authority.
        assert NAMED not in baseline_checked_answers
        assert all("cheap_authority_policy" not in saved for saved in checkpoints)
