"""Per-packet navigation sharing preserves exact originals and refresh boundaries."""

import copy
import json

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_navigation import (
    ProvisionNavigationAnchor,
    derive_provision_navigation_anchor,
)
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import (
    EvidenceItem,
    HarnessView,
    RunContext,
    RunStopped,
    SharedBudget,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.models import ChatCompletionMessage
from tests.unit.onyx.asv3.test_native_model_adapter import adaptive_tool_view, model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original

_SOURCE = "00000000-0000-0000-0000-000000000241"


def _item(
    chunk: str,
    *,
    title: str = "4458 sayılı Gümrük Kanunu",
    heading: str = "MADDE 241",
    source: str = _SOURCE,
) -> EvidenceItem:
    result = original(
        chunk,
        f"Clause {chunk}: A AND B; exception C and later procedure D. İĞŞ 東京\n\t  ",
        source=source,
        headings=[title, heading],
        metadata={"title": title, "document_type": "kanun"},
    )
    assert result.search_doc is not None
    result.search_doc.metadata["regulatory_chunk_id"] = chunk
    result.search_doc.semantic_identifier = title
    return result


def _record(anchor: ProvisionNavigationAnchor) -> dict[str, JsonValue]:
    return {
        "anchor_source_id": anchor.source_id,
        "instrument_name": anchor.instrument_name,
        "instrument_number": anchor.instrument_number,
        "article_no": anchor.article_no,
        "qualifier": anchor.qualifier,
        "status": "available",
        "has_more": False,
        "candidates": [
            {
                "source_id": "00000000-0000-0000-0000-000000000999",
                "name": "Related judicial reading lead",
                "candidate_role": "judicial_candidate",
            }
        ],
        "navigation_only": True,
        "absence_proven": False,
    }


class _Navigation:
    def __init__(self) -> None:
        self.calls: list[ProvisionNavigationAnchor] = []
        self.scopes: list[dict[str, JsonValue]] = []
        self.records: dict[ProvisionNavigationAnchor, dict[str, JsonValue]] = {}

    def __call__(
        self, item: EvidenceItem, context: RunContext
    ) -> dict[str, JsonValue] | None:
        context.check_active()
        anchor = derive_provision_navigation_anchor(item.source_id, [item])
        assert anchor is not None
        self.calls.append(anchor)
        self.scopes.append(copy.deepcopy(context.scope))
        self.records[anchor] = _record(anchor)
        return copy.deepcopy(self.records[anchor])

    def view(self) -> list[dict[str, JsonValue]]:
        return copy.deepcopy(list(self.records.values()))


def _adapter(
    items: list[EvidenceItem], *, fast: object = False
) -> tuple[ResearchModel, HarnessView, _Navigation, EvidenceLedger]:
    context = RunContext(
        run_id="00000000-0000-0000-0000-000000000003",
        budget=SharedBudget(unlimited_execution=True),
        services={
            "research_profile": "normal",
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
            "asv3_exp_cropped": True,
            "asv3_fast": fast,
        },
    )
    ledger = EvidenceLedger()
    ledger.add(items, context)
    navigation = _Navigation()
    context.services.update(
        evidence=ledger,
        legal_source_navigation_acquire=navigation,
        legal_source_navigation=navigation.view,
    )
    view = adaptive_tool_view(
        original_evidence=[full_record(ledger, n) for n in ledger.citation_numbers()]
    )
    return (
        ResearchModel(model(limit=500_000), context, lean_native_mode=True),
        view,
        navigation,
        ledger,
    )


def _wire(
    result: tuple[list[ChatCompletionMessage], list[dict[str, JsonValue]], int],
) -> str:
    messages, tools, output = result
    return json.dumps(
        {
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": tools,
            "output": output,
        },
        ensure_ascii=False,
        allow_nan=False,
    )


@pytest.mark.parametrize("fast", [False, None, 1, "true", True])
def test_only_fast_shares_success_and_preserves_exact_native_packet(
    fast: object,
) -> None:
    items = [_item("opening"), _item("exception"), _item("procedure")]
    baseline, old_view, old_navigation, _ = _adapter(items)
    expected = baseline._fit_native_decision(old_view)
    adapter, view, navigation, ledger = _adapter(items, fast=fast)
    before = ledger.export()
    actual = adapter._fit_native_decision(view)
    assert _wire(actual) == _wire(expected)
    assert len(old_navigation.calls) == 3
    assert len(navigation.calls) == (1 if fast is True else 3)
    assert ledger.export() == before
    catalogue = _NativeOriginalCatalogue.bind(actual[0], cheap_metadata=True)
    catalogue.validate(actual[0])
    assert {row["citation"] for row in catalogue.records} == {1, 2, 3}


@pytest.mark.parametrize(
    "different",
    [
        {"heading": "GEÇİCİ MADDE 241"},
        {"heading": "MADDE 242"},
        {"title": "213 sayılı Vergi Usul Kanunu"},
        {"title": "4459 sayılı Gümrük Kanunu"},
        {"source": "00000000-0000-0000-0000-000000000242"},
    ],
)
def test_full_anchor_identity_keeps_distinct_navigation_and_originals(
    different: dict[str, str],
) -> None:
    items = [
        _item("first-opening"),
        _item("first-exception"),
        _item("second-opening", **different),
        _item("second-exception", **different),
    ]
    baseline, old_view, old_navigation, _ = _adapter(items)
    adapter, view, navigation, _ = _adapter(items, fast=True)
    assert _wire(adapter._fit_native_decision(view)) == _wire(
        baseline._fit_native_decision(old_view)
    )
    assert len(old_navigation.calls) == 4
    assert len(navigation.calls) == 2
    assert len(set(navigation.calls)) == 2
    assert navigation.records == old_navigation.records


@pytest.mark.parametrize("status", [None, "denied", "unavailable", "error"])
def test_unsuccessful_navigation_does_not_suppress_later_original_retry(
    status: str | None,
) -> None:
    adapter, view, navigation, _ = _adapter(
        [_item("opening"), _item("exception"), _item("procedure")], fast=True
    )
    attempts: list[str] = []

    def acquire(item: EvidenceItem, context: RunContext) -> dict[str, JsonValue] | None:
        attempts.append(item.chunk_id or "")
        if len(attempts) == 1:
            return None if status is None else {"status": status}
        return navigation(item, context)

    adapter.context.services["legal_source_navigation_acquire"] = acquire
    adapter._fit_native_decision(view)
    assert attempts == ["opening", "exception"]
    assert len(navigation.calls) == 1


@pytest.mark.parametrize(
    "field",
    [
        "anchor_source_id",
        "instrument_name",
        "instrument_number",
        "article_no",
        "qualifier",
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_available_status_requires_every_exact_anchor_field_before_sharing(
    field: str, missing: bool
) -> None:
    adapter, view, navigation, _ = _adapter(
        [_item("opening"), _item("exception"), _item("procedure")], fast=True
    )
    attempts = 0

    def acquire(item: EvidenceItem, context: RunContext) -> dict[str, JsonValue] | None:
        nonlocal attempts
        attempts += 1
        result = navigation(item, context)
        assert result is not None
        if attempts == 1:
            if missing:
                del result[field]
            else:
                result[field] = "different identity"
        return result

    adapter.context.services["legal_source_navigation_acquire"] = acquire
    adapter._fit_native_decision(view)
    assert attempts == 2


def test_error_propagates_and_next_packet_reacquires_without_cross_call_cache() -> None:
    adapter, view, navigation, _ = _adapter(
        [_item("opening"), _item("exception")], fast=True
    )

    def fail(_item: EvidenceItem, _context: RunContext) -> dict[str, JsonValue] | None:
        raise PermissionError("navigation scope revoked")

    adapter.context.services["legal_source_navigation_acquire"] = fail
    with pytest.raises(PermissionError, match="scope revoked"):
        adapter._fit_native_decision(view)
    adapter.context.services["legal_source_navigation_acquire"] = navigation
    adapter._fit_native_decision(view)
    adapter.context.scope = {"document_set_id": 15, "as_of_date": "2026-10-30"}
    adapter._fit_native_decision(view)
    assert len(navigation.calls) == 2
    assert navigation.scopes == [{}, adapter.context.scope]


@pytest.mark.parametrize("fast", [False, True])
def test_cancellation_after_first_success_is_checked_before_duplicate_reuse(
    fast: bool,
) -> None:
    adapter, view, navigation, _ = _adapter(
        [_item("opening"), _item("exception")], fast=fast
    )

    def acquire(item: EvidenceItem, context: RunContext) -> dict[str, JsonValue] | None:
        result = navigation(item, context)
        context.cancel()
        return result

    adapter.context.services["legal_source_navigation_acquire"] = acquire
    with pytest.raises(RunStopped, match="Research cancelled"):
        adapter._fit_native_decision(view)
    assert len(navigation.calls) == 1


@pytest.mark.parametrize("defect", ["text", "source_id", "chunk_id", "range"])
def test_duplicate_navigation_keeps_each_originals_validation_behavior(
    defect: str,
) -> None:
    items = [_item("opening"), _item("exception")]
    packets: list[str] = []
    errors: list[tuple[type[BaseException], str]] = []
    for fast in (False, True):
        adapter, view, _, ledger = _adapter(items, fast=fast)
        before = ledger.export()
        damaged = copy.deepcopy(view.original_evidence)
        if defect == "text":
            damaged[1]["text"] = str(damaged[1]["text"]) + "forged text"
        elif defect == "range":
            damaged[1]["start_char"] = 1
        else:
            damaged[1][defect] = "different original"
        view = view.model_copy(update={"original_evidence": damaged})
        try:
            packets.append(_wire(adapter._fit_native_decision(view)))
        except ValueError as error:
            errors.append((type(error), str(error)))
        assert ledger.export() == before
    assert packets[0:1] == packets[1:2]
    assert errors[0:1] == errors[1:2]
