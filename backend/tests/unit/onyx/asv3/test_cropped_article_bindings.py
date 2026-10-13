import copy
import hashlib
import json
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import cited_lower_statute_gap
from onyx.asv3.cheap_authority import (
    canonical_identity_ledger,
    canonical_named_authority_gap,
)
from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.cropped_article_bindings import (
    CroppedArticleBindings,
    cropped_identity_ledger,
    cropped_named_authority_gap,
    ordered_source_read_data,
    without_ordered_source_read_data,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker
from tests.unit.onyx.asv3.test_native_authority import original

FIXTURE = Path(__file__).with_name("fixtures") / "cropped_article_binding_5172.json"
OPTIONS = {
    "strict_reference_boundaries": True,
    "syntactic_reference_binding": True,
    "strict_numeric_paragraphs": True,
}
ANSWER = "5326 sayılı Kabahatler Kanunu m.15/3 uyarınca bir fiil hem kabahat hem de suç ise sadece suçtan yaptırım uygulanır [3]."


def context() -> RunContext:
    return RunContext(
        run_id="binding-run",
        scope={"owner": "owner", "corpus": 15},
        services={"asv3_exp_cropped": True},
    )


def originals() -> list[EvidenceItem]:
    data = json.loads(FIXTURE.read_text())
    result = []
    for position, row in enumerate(data["rows"], 30):
        assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_hash"]
        result.append(
            EvidenceItem(
                source_id=row["source_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                text_hash=row["text_hash"],
                metadata={"position": position, "canonical_metadata": row["metadata"]},
                search_doc=SearchDoc(
                    document_id=row["source_id"],
                    chunk_ind=position,
                    semantic_identifier=row["metadata"]["title"],
                    blurb=row["text"],
                    source_type=DocumentSource.FILE,
                    boost=0,
                    hidden=False,
                    metadata={"regulatory_chunk_id": row["chunk_id"]},
                    match_highlights=[],
                ),
            )
        )
    return result


def setup(
    items: list[EvidenceItem] | None = None,
) -> tuple[RunContext, EvidenceLedger, CroppedArticleBindings, ToolReceipt]:
    current = context()
    evidence = originals() if items is None else items
    ledger = EvidenceLedger()
    numbers = ledger.add(evidence, current)
    state = CroppedArticleBindings(current, "Question")
    current.services["cropped_article_bindings"] = state
    receipt = ToolReceipt(
        call=CapabilityCall(
            name="read_source_range",
            arguments={"source_id": evidence[0].source_id, "start": 30, "limit": 3},
            call_id="ordered-read",
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Canonical ordered range",
            evidence=evidence,
            data={"has_more": True, **ordered_source_read_data(evidence, current)},
        ),
        evidence_ids=numbers,
        elapsed_seconds=0,
    )
    return current, ledger, state, receipt


def test_captured_414_requires_actual_contiguous_host_read() -> None:
    current, ledger, state, receipt = setup()
    before = ledger.export()
    assert canonical_named_authority_gap(ANSWER, ledger, **OPTIONS)
    assert cropped_named_authority_gap(ANSWER, ledger, current, **OPTIONS)
    assert state.record_receipt(receipt, current, ledger) == 2
    assert cropped_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
    assert ledger.export() == before
    retained = ledger.get(3)
    assert retained is not None
    canonical = retained.metadata["canonical_metadata"]
    assert isinstance(canonical, dict) and canonical["article_no"] is None
    # Correct attribution is still an identity witness, never a legal-merits approval.
    assert cropped_named_authority_gap(
        ANSWER.replace("[3]", "[1]"), ledger, current, **OPTIONS
    )
    assert cropped_named_authority_gap(
        ANSWER.replace("15/3", "15/4"), ledger, current, **OPTIONS
    )
    assert cropped_named_authority_gap(
        ANSWER.replace("5326", "6455"), ledger, current, **OPTIONS
    )
    assert cropped_named_authority_gap(
        ANSWER.replace("m.15/3", "geçici m.15/3"), ledger, current, **OPTIONS
    )


def combined_original(count: int, *, article_no: str | None = None) -> EvidenceItem:
    items = originals()
    raw = items[0].model_dump()
    raw["text"] = "\n".join(item.text for item in items[:count])
    raw["text_hash"] = ""
    metadata = raw["metadata"]["canonical_metadata"]
    metadata["paragraph_no"] = None
    metadata["article_no"] = article_no
    return EvidenceItem.model_validate(raw)


@pytest.mark.parametrize("article_no", [None, "15"])
@pytest.mark.parametrize("paragraph", ["1", "2", "3"])
def test_own_heading_keeps_every_previously_supported_literal_paragraph(
    article_no: str | None, paragraph: str
) -> None:
    item = combined_original(3, article_no=article_no)
    current, ledger, state, receipt = setup([item])
    claim = ANSWER.replace("15/3", f"15/{paragraph}").replace("[3]", "[1]")
    before = ledger.export()
    assert canonical_named_authority_gap(claim, ledger, **OPTIONS) is None
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.identity_bindings(current, ledger) == {}
    assert cropped_named_authority_gap(claim, ledger, current, **OPTIONS) is None
    assert ledger.export() == before


@pytest.mark.parametrize("paragraph", ["1", "2", "3"])
def test_own_heading_keeps_delivered_lower_statute_dependency_behavior(
    paragraph: str,
) -> None:
    current, ledger, state, receipt = setup([combined_original(3)])
    lower = original(
        "Uygulama Tebliği",
        "1",
        kind="tebliğ",
        text=f"5326 sayılı Kabahatler Kanunu m.15/{paragraph} kapsamında işlem yapılır.",
    )
    assert ledger.add([lower], current) == [2]
    claim = "Başvuru koşulu ayrıca incelenmiştir [2]."
    ledger.record_delivery(
        "writer", "asv3_coordinator", [{"citation": 2, "text": lower.text}]
    )
    baseline = canonical_identity_ledger(ledger)
    assert cited_lower_statute_gap(
        claim, baseline, ledger.completely_delivered("writer")
    )
    statute = ledger.get(1)
    assert statute is not None
    ledger.record_delivery(
        "writer", "asv3_coordinator", [{"citation": 1, "text": statute.text}]
    )
    delivered = ledger.completely_delivered("writer")
    assert delivered == {1, 2}
    assert cited_lower_statute_gap(claim, baseline, delivered) is None
    before = ledger.export()
    state.record_receipt(receipt, current, ledger)
    assert (
        cited_lower_statute_gap(
            claim, cropped_identity_ledger(ledger, current), delivered
        )
        is None
    )
    assert ledger.export() == before


def test_multi_paragraph_anchor_advances_only_to_its_last_literal_paragraph() -> None:
    anchor = combined_original(2)
    continuation = originals()[2]
    continuation.metadata["position"] = 31
    current, ledger, state, receipt = setup([anchor, continuation])
    assert state.record_receipt(receipt, current, ledger) == 1
    assert {
        number: binding.paragraph_no
        for number, binding in state.identity_bindings(current, ledger).items()
    } == {2: "3"}
    assert (
        cropped_named_authority_gap(
            ANSWER.replace("[3]", "[2]"), ledger, current, **OPTIONS
        )
        is None
    )
    for paragraph in ("1", "2"):
        claim = ANSWER.replace("15/3", f"15/{paragraph}").replace("[3]", "[1]")
        assert canonical_named_authority_gap(claim, ledger, **OPTIONS) is None
        assert cropped_named_authority_gap(claim, ledger, current, **OPTIONS) is None


@pytest.mark.parametrize("opening", ["(1)", "1."])
def test_plain_numbered_list_does_not_advance_an_article_paragraph(
    opening: str,
) -> None:
    raw = originals()[0].model_dump()
    raw["text"] = (
        raw["text"].replace("(1)", opening, 1)
        + "\n2. İşlem listesi ve başvuru aşamaları."
    )
    raw["text_hash"] = ""
    anchor = EvidenceItem.model_validate(raw)
    continuation = originals()[2]
    continuation.metadata["position"] = 31
    current, ledger, state, receipt = setup([anchor, continuation])
    claim = ANSWER.replace("[3]", "[2]")
    assert canonical_named_authority_gap(claim, ledger, **OPTIONS)
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.identity_bindings(current, ledger) == {}
    assert cropped_named_authority_gap(claim, ledger, current, **OPTIONS)
    initial = ANSWER.replace("15/3", "15/1").replace("[3]", "[1]")
    assert canonical_named_authority_gap(initial, ledger, **OPTIONS) is None
    assert cropped_named_authority_gap(initial, ledger, current, **OPTIONS) is None


@pytest.mark.parametrize("sequence", ["jump", "duplicate"])
def test_ambiguous_multi_paragraph_anchor_does_not_bind_a_following_chunk(
    sequence: str,
) -> None:
    anchor = combined_original(2)
    raw = anchor.model_dump()
    raw["text"] = anchor.text.replace(
        "\n(2)", "\n(4)" if sequence == "jump" else "\n(1)", 1
    )
    raw["text_hash"] = ""
    anchor = EvidenceItem.model_validate(raw)
    continuation = originals()[2]
    continuation.metadata["position"] = 31
    current, ledger, state, receipt = setup([anchor, continuation])
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.identity_bindings(current, ledger) == {}
    assert cropped_named_authority_gap(
        ANSWER.replace("[3]", "[2]"), ledger, current, **OPTIONS
    )


def test_existing_canonical_article_continuation_is_not_overridden() -> None:
    items = originals()
    raw = items[1].model_dump()
    raw["text"] = "\n".join(item.text for item in items[1:])
    raw["text_hash"] = ""
    metadata = raw["metadata"]["canonical_metadata"]
    metadata["article_no"] = "15"
    metadata["paragraph_no"] = None
    items = [items[0], EvidenceItem.model_validate(raw)]
    current, ledger, state, receipt = setup(items)
    baseline = canonical_identity_ledger(ledger).authority_metadata()
    state.record_receipt(receipt, current, ledger)
    assert state.identity_bindings(current, ledger) == {}
    assert cropped_identity_ledger(ledger, current).authority_metadata() == baseline


@pytest.mark.parametrize(
    "case",
    [
        "missing_middle",
        "position_gap",
        "reverse",
        "only_paragraph",
        "conflicting_article",
        "conflicting_paragraph",
        "another_heading",
        "amending",
        "quoted",
        "wrong_identifier",
    ],
)
@pytest.mark.parametrize("method", ["read_source_range", "read_chunk_context"])
def test_no_neighbor_or_prose_inference(case: str, method: str) -> None:
    items = originals()
    if case == "missing_middle":
        items = [items[0], items[2]]
    elif case == "position_gap":
        items[2].metadata["position"] = 34
    elif case == "reverse":
        items = list(reversed(items))
    elif case == "only_paragraph":
        items = items[2:]
    else:
        metadata = items[2].metadata["canonical_metadata"]
        assert isinstance(metadata, dict)
        if case == "conflicting_article":
            metadata["article_no"] = "16"
        if case == "conflicting_paragraph":
            metadata["paragraph_no"] = "4"
        if case == "wrong_identifier":
            metadata["provision_identifiers"] = ["madde 16"]
        if case in {"another_heading", "amending", "quoted"}:
            text = {
                "another_heading": "**MADDE 16** .- (3) Başka madde.",
                "amending": "(3) 5326 sayılı Kanunun 15 inci maddesi aşağıdaki şekilde değiştirilmiştir.",
                "quoted": "“(3) Bir fiil hem kabahat hem suç.”",
            }[case]
            items[2] = EvidenceItem.model_validate(
                {**items[2].model_dump(), "text": text, "text_hash": ""}
            )
    current, ledger, state, receipt = setup(items)
    receipt.call.name = method
    state.record_receipt(receipt, current, ledger)
    assert not any(
        binding.paragraph_no == "3"
        for binding in state.identity_bindings(current, ledger).values()
    )


@pytest.mark.parametrize(
    "case",
    [
        "wrong_scope",
        "wrong_run",
        "wrong_source",
        "hash",
        "bool_position",
        "partial_body",
        "wrong_chunk",
        "truncated",
        "external",
    ],
)
@pytest.mark.parametrize("method", ["read_source_range", "read_chunk_context"])
def test_tampered_receipts_never_project(case: str, method: str) -> None:
    current, ledger, state, receipt = setup()
    raw = receipt.outcome.data["cropped_ordered_source_read"]
    assert isinstance(raw, dict)
    if case == "wrong_scope":
        current.scope["owner"] = "foreign"
    elif case == "wrong_run":
        current.run_id = "foreign"
    elif case == "wrong_source":
        receipt.call.arguments["source_id"] = "foreign"
    elif case in {"truncated", "external"}:
        items = originals()
        items[2].metadata[case] = True
        current, ledger, state, receipt = setup(items)
    else:
        members = raw["members"]
        assert isinstance(members, list)
        member = members[-1]
        assert isinstance(member, dict)
        member[
            {
                "hash": "text_hash",
                "bool_position": "position",
                "partial_body": "end_char",
                "wrong_chunk": "chunk_id",
            }[case]
        ] = {
            "hash": "0" * 64,
            "bool_position": True,
            "partial_body": 5,
            "wrong_chunk": "foreign",
        }[case]
    receipt.call.name = method
    with pytest.raises(ValueError):
        state.record_receipt(receipt, current, ledger)
    assert state.export()["reads"] == []


def test_nested_provision_data_is_not_mistaken_for_a_direct_receipt() -> None:
    current, ledger, state, receipt = setup()
    receipt.call.name = "follow_reference"
    assert state.record_receipt(receipt, current, ledger) == 0
    receipt.call.name = "search_corpus"
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.export()["reads"] == []


def test_checkpoint_roundtrip_and_current_scope_fence() -> None:
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    saved = state.export()
    restored = CroppedArticleBindings(current, "Question")
    restored.restore(saved, current, ledger)
    assert restored.export() == saved
    current.services["cropped_article_bindings"] = restored
    assert cropped_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
    for field, value in [
        ("version", True),
        ("run_id", "foreign"),
        ("request_hash", "foreign"),
        ("scope_hash", "foreign"),
    ]:
        changed = copy.deepcopy(saved)
        changed[field] = value
        with pytest.raises(ValueError):
            restored.restore(changed, current, ledger)
    other = CroppedArticleBindings(current, "Different question")
    with pytest.raises(ValueError):
        other.restore(saved, current, ledger)


def test_checkpoint_current_original_integrity_and_repeated_receipt() -> None:
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    assert state.record_receipt(receipt, current.child(), ledger) == 2
    assert len(cast(list[JsonValue], state.export()["reads"])) == 1
    saved = state.export()
    altered = EvidenceLedger()
    items = originals()
    items[2].metadata["canonical_metadata"] = {"article_no": "16"}
    altered.add(items, current)
    restored = CroppedArticleBindings(current, "Question")
    with pytest.raises(ValueError):
        restored.restore(saved, current, altered)


def test_same_length_late_literal_change_invalidates_identity_and_checkpoint() -> None:
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    saved = state.export()
    item = ledger._items[3]
    old_text = item.text
    item.text = item.text.replace("fiil", "olay", 1)
    assert item.text != old_text and len(item.text) == len(old_text)
    assert hashlib.sha256(item.text.encode()).hexdigest() != item.text_hash
    with pytest.raises(ValueError, match="canonical original"):
        state.identity_bindings(current, ledger)
    restored = CroppedArticleBindings(current, "Question")
    with pytest.raises(ValueError, match="canonical original"):
        restored.restore(saved, current, ledger)


def test_exact_search_original_can_be_reread_with_canonical_ordered_witness() -> None:
    fresh = originals()
    acquired = [item.model_copy(deep=True) for item in fresh]
    # CorpusBroker.hydrate_search_centers annotates evidence_for_chunk this way.
    for item in acquired:
        item.metadata.update(
            retrieval_method="established_search_exact_original",
            article_closure_complete=False,
            section_context="not_inferred",
            additional_context="harness_controlled",
            follow_context_tool="read_provision",
            retrieved_projection_ordinal=item.metadata["position"],
            retrieved_center=True,
        )
    current, ledger, state, receipt = setup(acquired)
    initial_metadata = ledger.get(3)
    assert initial_metadata is not None
    receipt.outcome.evidence = fresh
    receipt.outcome.data = {
        "has_more": True,
        **ordered_source_read_data(fresh, current),
    }
    assert ledger.add(fresh, current) == [1, 2, 3]
    assert state.record_receipt(receipt, current, ledger) == 2
    retained = ledger.get(3)
    assert retained is not None and retained.metadata == initial_metadata.metadata
    assert cropped_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
    assert ledger.completely_delivered("next-writer") == set()


@pytest.mark.parametrize(
    "field",
    [
        "canonical",
        "read_as_of_date",
        "validity_end",
        "source_sha256",
        "publication_payload_sha256",
    ],
)
def test_reacquisition_never_ignores_source_provenance_or_canonical_changes(
    field: str,
) -> None:
    fresh = originals()
    acquired = [item.model_copy(deep=True) for item in fresh]
    if field == "canonical":
        canonical = acquired[2].metadata["canonical_metadata"]
        assert isinstance(canonical, dict)
        canonical["paragraph_no"] = "4"
    else:
        acquired[2].metadata[field] = "a different source snapshot"
    current, ledger, state, receipt = setup(acquired)
    receipt.outcome.evidence = fresh
    receipt.outcome.data = {
        "has_more": True,
        **ordered_source_read_data(fresh, current),
    }
    assert ledger.add(fresh, current) == [1, 2, 3]
    with pytest.raises(ValueError, match="canonical original"):
        state.record_receipt(receipt, current, ledger)
    assert state.export()["reads"] == []


def test_default_modes_and_cheap_projection_remain_unchanged() -> None:
    items = originals()
    controls: list[dict[str, object]] = [{}, {"asv3_exp_cheap": True}]
    for services in controls:
        current = RunContext(services=services)
        assert ordered_source_read_data(items, current) == {}
        with pytest.raises(ValueError):
            CroppedArticleBindings(current, "Question")
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    assert canonical_named_authority_gap(ANSWER, ledger, **OPTIONS)
    assert (
        cropped_identity_ledger(ledger, current).authority_metadata()[2]["paragraph_no"]
        == "3"
    )


@pytest.mark.parametrize("cropped", [True, False])
@pytest.mark.parametrize("prior_search", [True, False])
@pytest.mark.parametrize("method", ["read_source_range", "read_chunk_context"])
def test_actual_corpus_range_emits_only_cropped_positional_witness(
    cropped: bool,
    prior_search: bool,
    method: str,
) -> None:
    items = originals()
    chunks = []
    for item in items:
        assert item.chunk_id is not None
        canonical = item.metadata["canonical_metadata"]
        assert isinstance(canonical, dict)
        headings = canonical["heading_path"]
        assert isinstance(headings, list)
        position = item.metadata["position"]
        assert type(position) is int
        chunks.append(
            CorpusChunk(
                id=item.chunk_id,
                source_id=UUID(item.source_id),
                text=item.text,
                position=position,
                projection_ordinal=position,
                heading_path=tuple(str(h) for h in headings),
                metadata=canonical,
                validity_start=None,
                validity_end=None,
                status="active",
            )
        )
    broker = MemoryBroker(chunks)
    broker.item = CorpusSource(
        UUID(items[0].source_id), "Diğer Kanunlar/5326_sayili_kanun.md", "captured-file"
    )
    current = context() if cropped else RunContext()
    state = CroppedArticleBindings(current, "Question") if cropped else None
    if state is not None:
        current.services["cropped_article_bindings"] = state
    tool = next(s for s in build_corpus_specs(broker) if s.name == method)
    args: dict[str, JsonValue] = {
        "source_id": items[0].source_id,
        "start": 30,
        "limit": 3,
    }
    if method == "read_chunk_context":
        args = {"source_id": items[0].source_id, "chunk_id": items[2].chunk_id}
    outcome = tool.handler(args, current)
    assert outcome.status == OutcomeStatus.FOUND
    assert [item.identity for item in outcome.evidence] == [
        item.identity for item in items
    ]
    assert ("cropped_ordered_source_read" in outcome.data) is cropped
    if cropped:
        ledger = EvidenceLedger()
        if prior_search:
            previous = [item.model_copy(deep=True) for item in outcome.evidence]
            for item in previous:
                item.metadata.update(
                    retrieval_method="established_search_exact_original",
                    article_closure_complete=False,
                    section_context="not_inferred",
                    additional_context="harness_controlled",
                    follow_context_tool="read_provision",
                    retrieved_projection_ordinal=item.metadata["position"],
                    retrieved_center=True,
                )
            ledger.add(previous, current)
        numbers = ledger.add(outcome.evidence, current)
        assert state is not None
        receipt = ToolReceipt(
            call=CapabilityCall(name=tool.name, arguments=args),
            outcome=outcome,
            evidence_ids=numbers,
            elapsed_seconds=0,
        )
        assert state.record_receipt(receipt, current, ledger) == 2
        assert cropped_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None


def test_positional_witness_requires_valid_installed_runtime_state() -> None:
    current = context()
    assert ordered_source_read_data(originals(), current) == {}
    state = CroppedArticleBindings(current, "Question")
    current.services["cropped_article_bindings"] = state
    current.services["asv3_exp_cheap"] = True
    assert ordered_source_read_data(originals(), current) == {}
    current.services["asv3_exp_cheap"] = False
    current.run_id = "foreign"
    with pytest.raises(ValueError):
        ordered_source_read_data(originals(), current)


def test_host_only_proof_preserves_source_facts_and_persisted_identity() -> None:
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    raw = copy.deepcopy(receipt.outcome.data)
    receipt.outcome.data = without_ordered_source_read_data(receipt.outcome.data)
    assert "cropped_ordered_source_read" not in receipt.outcome.data
    assert receipt.outcome.data == {
        k: v for k, v in raw.items() if k != "cropped_ordered_source_read"
    }
    assert raw["cropped_ordered_source_read"]
    assert state.export()["reads"]
    assert cropped_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
    actual_source: dict[str, JsonValue] = {
        "text": originals()[2].text,
        "metadata": originals()[2].metadata,
    }
    nested: dict[str, JsonValue] = {
        "sources": [actual_source],
        "steps": {
            "read": {"status": "partial", "summary": "Read actual source", "data": raw}
        },
        "results": [
            {
                "outcome": {
                    "status": "partial",
                    "summary": "Read actual source",
                    "data": raw,
                }
            }
        ],
    }
    cleaned = without_ordered_source_read_data(nested)
    assert cleaned["sources"] == nested["sources"]
    steps = cleaned["steps"]
    assert isinstance(steps, dict)
    step = steps["read"]
    assert isinstance(step, dict) and step["data"] == receipt.outcome.data
    assert "cropped_ordered_source_read" not in json.dumps(cleaned, ensure_ascii=False)
    assert "cropped_ordered_source_read" in json.dumps(nested, ensure_ascii=False)


def test_actual_harness_consumes_host_proof_before_model_tool_history() -> None:
    current, ledger, state, receipt = setup()
    raw = copy.deepcopy(receipt.outcome.data)
    harness = Harness(
        request="Question",
        context=current,
        registry=CapabilityRegistry([]),
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    harness._commit_receipt(receipt)
    result = harness._model_tool_result(receipt)
    payload = json.loads(result.content)
    assert "cropped_ordered_source_read" not in result.content
    assert payload["outcome"]["data"] == {
        k: v for k, v in raw.items() if k != "cropped_ordered_source_read"
    }
    assert [row["text"] for row in payload["original_evidence"]] == [
        item.text for item in originals()
    ]
    assert state.export()["reads"]
    assert cropped_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
    assert "cropped_ordered_source_read" not in json.dumps(
        harness.snapshot()["receipts"]
    )
