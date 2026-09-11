import json
from functools import partial
from unittest.mock import MagicMock

import pytest

from onyx.regulatory.amendments.annexes.comparison import (
    compare_annexes,
    validate_annex_comparison,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexComparison,
    AnnexLocator,
    ExtractedAnnexElement,
)
from tests.unit.onyx.regulatory.annexes.test_comparison import extraction, page


@pytest.mark.parametrize(
    "outcome",
    [
        "repaired",
        "missing_new",
        "outside",
        "aggregate",
        "overlap",
        "incomplete",
        "uncertain",
        "unchanged",
    ],
)
def test_actual_structured_wire_resolves_only_frozen_positions(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    import httpx

    from onyx.llm.factory import get_llm

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    old = extraction("5%", visual=True)
    new = extraction("5%" if outcome == "unchanged" else "7%", visual=True)
    if outcome == "unchanged":
        new.source_sha256 = "b" * 64
    for view in (old, new):
        view.elements[0].locator.normalized_box = (
            0.123456789123,
            0.234567891234,
            0.876543219876,
            0.987654321987,
        )
        view.elements.insert(
            0,
            ExtractedAnnexElement(
                kind="text",
                text="INELIGIBLE_NATIVE_AGGREGATE",
                aggregate=True,
                extraction_method="native",
                locator=AnnexLocator(page=1),
            ),
        )
        duplicate = view.elements[1].model_copy(deep=True)
        duplicate.extraction_method = "native"
        view.elements.append(duplicate)
        view.elements.append(
            ExtractedAnnexElement(
                kind="text", text="unchanged note", locator=AnnexLocator(page=1)
            )
        )
    original_snapshots = old.model_dump_json(), new.model_dump_json()
    calls: list[dict[str, object]] = []

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: object
    ) -> httpx.Response:
        body = json.loads(request.content)
        if "messages" not in body:
            # Capability discovery is unavailable in this closed transport fixture.
            return httpx.Response(503, request=request)
        calls.append(body)
        assert "image_url" in str(body["messages"])
        assert "INELIGIBLE_NATIVE_AGGREGATE" not in str(body["messages"])
        proposal = {
            "changes": []
            if outcome == "unchanged"
            else [
                {
                    "operation": "replace",
                    "old_positions": [
                        99
                        if outcome == "outside"
                        else 0
                        if outcome == "aggregate"
                        else 1
                    ],
                    "new_positions": []
                    if outcome == "missing_new"
                    or (outcome == "repaired" and len(calls) == 1)
                    else [1],
                    "explanation": "Rate changed from 5% to 7%",
                    "uncertain": outcome == "uncertain",
                }
            ],
            "old_positions": [1, 2] if outcome == "incomplete" else [1, 2, 3],
            "new_positions": [1, 2, 3],
            "old_pages": [1],
            "new_pages": [1],
            "issues": [],
        }
        if outcome == "overlap":
            proposal["changes"] *= 2
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(proposal),
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    llm = get_llm(
        provider="openai",
        model="fixture-model",
        max_input_tokens=32000,
        deployment_name=None,
        api_key="fictional-only",
        timeout=30,
        temperature=0,
    )
    monkeypatch.setattr(llm, "invoke", partial(llm.invoke, use_streaming=False))
    result = compare_annexes(
        old=old, new=new, old_pages=[page()], new_pages=[page("red")], llm=llm
    )
    assert result.ready is (outcome in {"repaired", "unchanged"})
    assert len(calls) == (2 if outcome in {"repaired", "missing_new", "overlap"} else 1)
    assert result.prompt_version == "annex-comparison-v3"
    assert (old.model_dump_json(), new.model_dump_json()) == original_snapshots
    assert result.coverage.old_positions == [1, 2, 3] == result.coverage.new_positions
    if outcome == "repaired":
        change = result.changes[0]
        assert change.old[0].position == change.new[0].position == 1
        assert change.old[0].text == "5%" and change.new[0].text == "7%"
        assert change.new[0].locator == new.elements[1].locator
        assert "invalid_operation_shape" in str(calls[1]["messages"])
        assert not validate_annex_comparison(result, old=old, new=new)
        assert AnnexComparison.model_validate_json(result.model_dump_json()) == result
    elif outcome in {"outside", "aggregate"}:
        assert "invalid_snapshot_reference" in result.issues and not result.changes
    elif outcome == "incomplete":
        assert "incomplete_comparison_coverage" in result.issues
    elif outcome == "uncertain":
        assert "uncertain_difference" in result.issues
    elif outcome == "unchanged":
        assert not result.changes
    else:
        assert result.issues and not result.changes


@pytest.mark.parametrize(
    "operation,old,new",
    [
        ("replace", [1], [2]),
        ("move", [1], [2]),
        ("insert", [], [2]),
        ("remove", [1], []),
        ("split", [1], [2, 3]),
        ("merge", [1, 2], [3]),
        ("visual", [1, 2], [3, 4]),
    ],
)
def test_proposal_preserves_operation_shapes(
    operation: str, old: list[int], new: list[int]
) -> None:
    from pydantic import ValidationError

    from onyx.regulatory.amendments.annexes.models import AnnexDifferenceProposal

    value = {
        "operation": operation,
        "old_positions": old,
        "new_positions": new,
        "explanation": "fixture",
    }
    assert AnnexDifferenceProposal.model_validate(value).operation == operation
    with pytest.raises(ValidationError, match="invalid_operation_shape"):
        AnnexDifferenceProposal.model_validate(
            {**value, "old_positions": [], "new_positions": []}
        )


@pytest.mark.parametrize("position", [True, 1.0, -1])
def test_wire_positions_are_strict_nonnegative_integers(position: object) -> None:
    from pydantic import ValidationError

    from onyx.regulatory.amendments.annexes.models import AnnexDifferenceProposal

    with pytest.raises(ValidationError):
        AnnexDifferenceProposal.model_validate(
            {
                "operation": "replace",
                "old_positions": [position],
                "new_positions": [1],
                "explanation": "fixture",
            }
        )


def test_provider_failures_are_not_mislabeled_as_invalid_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.annexes import comparison

    llm = MagicMock()
    llm.config.model_provider = "configured"
    llm.config.model_name = "fixture"
    monkeypatch.setattr(
        comparison,
        "generate_structured",
        MagicMock(side_effect=ValueError("provider failure")),
    )
    with pytest.raises(ValueError, match="provider failure"):
        compare_annexes(
            old=extraction("5%", visual=True),
            new=extraction("7%", visual=True),
            old_pages=[page()],
            new_pages=[page()],
            llm=llm,
        )
