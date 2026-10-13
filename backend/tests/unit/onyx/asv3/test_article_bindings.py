import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.article_bindings import (
    ArticleBindings,
    bound_identity_ledger,
    bound_named_authority_gap,
    ordered_source_read_data,
    without_ordered_source_read_data,
)
from onyx.asv3.canonical_authority import (
    canonical_cited_lower_statute_gap,
    canonical_named_authority_gap,
)
from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.governing_acquisitions import (
    GoverningOriginalAcquisitions,
    GoverningOriginalTarget,
)
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolReceipt,
)
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker
from tests.unit.onyx.asv3.test_native_authority import original
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals,
    response,
    setup_run,
)
from tests.unit.onyx.asv3.test_spawned_researcher_terminal import (
    capture_harnesses,
    scripted_coordinator,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")

FIXTURE = Path(__file__).with_name("fixtures") / "article_binding_5172.json"
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
        services={"asv3_workflow_variant": "asv3_tuned"},
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
) -> tuple[RunContext, EvidenceLedger, ArticleBindings, ToolReceipt]:
    current = context()
    evidence = originals() if items is None else items
    ledger = EvidenceLedger()
    numbers = ledger.add(evidence, current)
    state = ArticleBindings(current, "Question")
    current.services["tuned_article_bindings"] = state
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
    assert bound_named_authority_gap(ANSWER, ledger, current, **OPTIONS)
    assert state.record_receipt(receipt, current, ledger) == 2
    assert bound_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
    assert ledger.export() == before
    retained = ledger.get(3)
    assert retained is not None
    canonical = retained.metadata["canonical_metadata"]
    assert isinstance(canonical, dict) and canonical["article_no"] is None
    # Correct attribution is still an identity witness, never a legal-merits approval.
    assert bound_named_authority_gap(
        ANSWER.replace("[3]", "[1]"), ledger, current, **OPTIONS
    )
    assert bound_named_authority_gap(
        ANSWER.replace("15/3", "15/4"), ledger, current, **OPTIONS
    )
    assert bound_named_authority_gap(
        ANSWER.replace("5326", "6455"), ledger, current, **OPTIONS
    )
    assert bound_named_authority_gap(
        ANSWER.replace("m.15/3", "geçici m.15/3"), ledger, current, **OPTIONS
    )


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
def test_no_neighbor_or_prose_inference(case: str) -> None:
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
def test_tampered_receipts_never_project(case: str) -> None:
    current, ledger, state, receipt = setup()
    raw = receipt.outcome.data["asv3_ordered_source_read"]
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
    restored = ArticleBindings(current, "Question")
    restored.restore(saved, current, ledger)
    assert restored.export() == saved
    current.services["tuned_article_bindings"] = restored
    assert bound_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
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
    other = ArticleBindings(current, "Different question")
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
    restored = ArticleBindings(current, "Question")
    with pytest.raises(ValueError):
        restored.restore(saved, current, altered)


def test_literal_mutation_cannot_reuse_a_stored_hash() -> None:
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    item = ledger._items[3]
    item.text = item.text.replace("fiil", "olay")
    with pytest.raises(ValueError, match="canonical original"):
        state.identity_bindings(current, ledger)


def test_acquisition_match_and_lower_rule_keep_exact_paragraph_delivery() -> None:
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    acquisitions = GoverningOriginalAcquisitions(current, "Question")
    target = GoverningOriginalTarget(
        instrument_number="5326", article="15", paragraph="3"
    )
    assert acquisitions.matching_originals(target, ledger) == [3]
    assert (
        acquisitions.matching_originals(
            target.model_copy(update={"paragraph": "4"}), ledger
        )
        == []
    )
    lower = original(
        "Uygulama Tebliği",
        "1",
        kind="tebliğ",
        text="5326 sayılı Kabahatler Kanunu m.15/3 kapsamında başvuru alınır.",
    )
    assert ledger.add([lower], current) == [4]
    identity = bound_identity_ledger(ledger, current)
    body = "Başvuru koşulu ayrıca incelenmiştir [4]."
    assert canonical_cited_lower_statute_gap(body, identity, {1, 2, 4}) is not None
    assert canonical_cited_lower_statute_gap(body, identity, {3, 4}) is None
    assert identity.completely_delivered("unread-writer") == set()


def test_reacquisition_metadata_is_not_source_identity() -> None:
    current = context()
    state = ArticleBindings(current, "Question")
    current.services["tuned_article_bindings"] = state
    ledger = EvidenceLedger()
    searched = originals()
    for item in searched:
        item.metadata.update(
            retrieval_method="established_search_exact_original",
            article_closure_complete=False,
            section_context="not_inferred",
            additional_context="harness_controlled",
            follow_context_tool="read_provision",
            retrieved_projection_ordinal=item.metadata["position"],
            retrieved_center=True,
        )
    ledger.add(searched, current)
    direct = originals()
    receipt = ToolReceipt(
        call=CapabilityCall(
            name="read_source_range", arguments={"source_id": direct[0].source_id}
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual direct read",
            evidence=direct,
            data=ordered_source_read_data(direct, current),
        ),
        evidence_ids=ledger.add(direct, current),
        elapsed_seconds=0,
    )
    assert state.record_receipt(receipt, current, ledger) == 2
    assert bound_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None
    retained = ledger._items[3]
    retained.metadata["read_as_of_date"] = "2099-01-01"
    with pytest.raises(ValueError, match="canonical original"):
        state.identity_bindings(current, ledger)


def test_host_proof_stripping_preserves_source_metadata_and_nested_status() -> None:
    data: dict[str, JsonValue] = {
        "asv3_ordered_source_read": {"digest": "host"},
        "steps": {
            "read": {
                "status": "partial",
                "data": {
                    "asv3_ordered_source_read": {"digest": "nested"},
                    "has_more": True,
                    "next_position": 33,
                    "metadata": {"asv3_ordered_source_read": "actual metadata"},
                },
            }
        },
    }
    without_ordered_source_read_data(data)
    assert data == {
        "steps": {
            "read": {
                "status": "partial",
                "data": {
                    "has_more": True,
                    "next_position": 33,
                    "metadata": {"asv3_ordered_source_read": "actual metadata"},
                },
            }
        }
    }


def test_default_modes_and_unbound_projection_remain_unchanged() -> None:
    items = originals()
    controls: list[dict[str, object]] = [
        {},
        {"asv3_workflow_variant": "standard"},
        {"asv3_workflow_variant": "standard", "research_profile": "deep"},
    ]
    for services in controls:
        current = RunContext(services=services)
        assert ordered_source_read_data(items, current) == {}
        with pytest.raises(ValueError):
            ArticleBindings(current, "Question")
    current, ledger, state, receipt = setup()
    state.record_receipt(receipt, current, ledger)
    assert canonical_named_authority_gap(ANSWER, ledger, **OPTIONS)
    assert (
        bound_identity_ledger(ledger, current).authority_metadata()[2]["paragraph_no"]
        == "3"
    )


@pytest.mark.parametrize("tuned", [True, False])
def test_actual_corpus_range_emits_only_active_tuned_positional_witness(
    tuned: bool,
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
    current = context() if tuned else RunContext()
    if tuned:
        current.services["tuned_article_bindings"] = ArticleBindings(
            current, "Question"
        )
    tool = next(s for s in build_corpus_specs(broker) if s.name == "read_source_range")
    args: dict[str, JsonValue] = {
        "source_id": items[0].source_id,
        "start": 30,
        "limit": 3,
    }
    outcome = tool.handler(args, current)
    assert outcome.status == OutcomeStatus.FOUND
    assert [item.identity for item in outcome.evidence] == [
        item.identity for item in items
    ]
    assert ("asv3_ordered_source_read" in outcome.data) is tuned
    if tuned:
        ledger = EvidenceLedger()
        numbers = ledger.add(outcome.evidence, current)
        state = ArticleBindings(current, "Question")
        current.services["tuned_article_bindings"] = state
        receipt = ToolReceipt(
            call=CapabilityCall(name=tool.name, arguments=args),
            outcome=outcome,
            evidence_ids=numbers,
            elapsed_seconds=0,
        )
        assert state.record_receipt(receipt, current, ledger) == 2
        assert bound_named_authority_gap(ANSWER, ledger, current, **OPTIONS) is None


def _runtime_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Any, Any, list[dict[str, Any]]]:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant="asv3_tuned")
    kwargs["simple_chat_history"][
        -1
    ].message = "5326 sayılı Kabahatler Kanunu m.15/3 özgün hükmünü açıkla."
    items = originals()
    source = CorpusSource(
        UUID(items[0].source_id), "Diğer Kanunlar/5326_sayili_kanun.md", "capture"
    )
    chunks = [
        CorpusChunk(
            id=str(item.chunk_id),
            source_id=source.id,
            text=item.text,
            position=30 + index,
            projection_ordinal=30 + index,
            heading_path=tuple(
                cast(
                    list[str],
                    cast(dict[str, JsonValue], item.metadata["canonical_metadata"])[
                        "heading_path"
                    ],
                )
            ),
            metadata=copy.deepcopy(
                cast(dict[str, JsonValue], item.metadata["canonical_metadata"])
            ),
            validity_start=None,
            validity_end=None,
            status="active",
        )
        for index, item in enumerate(items)
    ]
    broker.sources = [source]
    broker.chunks = {str(source.id): chunks[0]}

    def page(
        source_id: str, current: RunContext, **arguments: Any
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        current.check_active()
        assert source_id == str(source.id)
        selected = [c for c in chunks if c.position >= arguments.get("start", 0)]
        limit = arguments.get("limit", 30)
        return source, selected[:limit], len(selected) > limit

    def revalidate(evidence: list[EvidenceItem], current: RunContext) -> None:
        current.check_active()
        by_id = {chunk.id: chunk for chunk in chunks}
        assert all(
            item.source_id == str(source.id)
            and item.text == by_id[str(item.chunk_id)].text
            for item in evidence
        )

    monkeypatch.setattr(broker, "page", page)
    monkeypatch.setattr(broker, "revalidate_evidence", revalidate)
    return kwargs, broker, llm, checkpoints


@pytest.mark.parametrize("variant", ["asv3_tuned", "standard"])
def test_actual_runtime_contiguous_read_is_tuned_only_and_host_only(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, broker, llm, checkpoints = _runtime_triplet(monkeypatch)
    kwargs["workflow_variant"] = variant
    harnesses = capture_harnesses(monkeypatch)
    count = 0

    def invoke(**arguments: Any) -> Any:
        nonlocal count
        count += 1
        assert "asv3_ordered_source_read" not in json.dumps(
            [m.model_dump(mode="json") for m in arguments["prompt"]]
        )
        if count == 1:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {
                            "source_id": str(broker.sources[0].id),
                            "start": 30,
                            "limit": 3,
                        },
                    )
                ]
            )
        if count == 2:
            assert {row["text"] for row in delivered_originals(arguments)} == {
                item.text for item in originals()
            }
            return response(ANSWER)
        assert variant == "standard"
        broker.cancelled.set()
        return response(ANSWER)

    llm.invoke.side_effect = invoke
    if variant == "asv3_tuned":
        runtime.run_asv3_loop(**kwargs)
        assert count == 2
        assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
        assert checkpoints[-1]["article_bindings"]["reads"][0]["citations"] == [1, 2, 3]
        assert (
            bound_identity_ledger(
                harnesses[0].evidence, harnesses[0].context
            ).authority_metadata()[2]["paragraph_no"]
            == "3"
        )
    else:
        with pytest.raises(RunStopped, match="Research cancelled"):
            runtime.run_asv3_loop(**kwargs)
        assert count == 3
        assert all("article_bindings" not in snapshot for snapshot in checkpoints)
    assert all(
        "asv3_ordered_source_read" not in receipt.outcome.data
        for h in harnesses
        for receipt in h.receipts
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_actual_runtime_resume_validates_optional_bindings(
    monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    kwargs, broker, llm, checkpoints = _runtime_triplet(monkeypatch)
    calls = 0

    def interrupted(**_arguments: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {
                            "source_id": str(broker.sources[0].id),
                            "start": 30,
                            "limit": 3,
                        },
                    )
                ]
            )
        broker.cancelled.set()
        return response(ANSWER)

    llm.invoke.side_effect = interrupted
    with pytest.raises(RunStopped, match="Research cancelled"):
        runtime.run_asv3_loop(**kwargs)
    saved = copy.deepcopy(checkpoints[-1])
    assert saved["article_bindings"]["reads"]
    if legacy:
        saved.pop("article_bindings")
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: saved)
    broker.cancelled.clear()
    calls = 0

    def resumed(**_arguments: Any) -> Any:
        nonlocal calls
        calls += 1
        if legacy and calls == 1:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {
                            "source_id": str(broker.sources[0].id),
                            "start": 30,
                            "limit": 3,
                        },
                    )
                ]
            )
        return response(ANSWER)

    llm.invoke.side_effect = resumed
    kwargs["resume_message_id"] = 2
    runtime.run_asv3_loop(**kwargs)
    assert calls == (2 if legacy else 1)
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"


def test_child_read_identity_is_shared_without_forging_writer_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints = _runtime_triplet(monkeypatch)
    harnesses = capture_harnesses(monkeypatch)
    task = "Kabahatler hükmünün fıkralarını özgün kaynaktan araştır."
    child_calls = 0

    def child(**arguments: Any) -> Any:
        nonlocal child_calls
        child_calls += 1
        if child_calls == 1:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {
                            "source_id": str(broker.sources[0].id),
                            "start": 30,
                            "limit": 3,
                        },
                    )
                ]
            )
        assert child_calls == 2
        assert {row["citation"] for row in delivered_originals(arguments)} == {1, 2, 3}
        return response(
            calls=[("submit_answer", {"answer": ANSWER, "basis": "originals"})]
        )

    llm.invoke.side_effect = scripted_coordinator([task], child)
    runtime.run_asv3_loop(**kwargs)
    assert child_calls == 2
    root, worker = harnesses
    assert (
        root.context.services["tuned_article_bindings"]
        is worker.context.services["tuned_article_bindings"]
    )
    assert root.evidence.completely_delivered("never-called-writer") == set()
    assert (
        bound_named_authority_gap(ANSWER, root.evidence, root.context, **OPTIONS)
        is None
    )
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"


@pytest.mark.parametrize("article_metadata", [None, "15"])
def test_complete_own_article_keeps_every_existing_literal_paragraph(
    article_metadata: str | None,
) -> None:
    items = originals()
    own = items[0].model_copy(deep=True)
    own.text = "\n\n".join(item.text for item in items)
    own.text_hash = hashlib.sha256(own.text.encode()).hexdigest()
    canonical = own.metadata["canonical_metadata"]
    assert isinstance(canonical, dict)
    canonical["article_no"] = article_metadata
    canonical["paragraph_no"] = None
    current, ledger, state, receipt = setup([own])
    body = ANSWER.replace("[3]", "[1]")
    before = canonical_named_authority_gap(body, ledger, **OPTIONS)
    assert before is None
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.identity_bindings(current, ledger) == {}
    assert bound_named_authority_gap(body, ledger, current, **OPTIONS) == before
    acquisitions = GoverningOriginalAcquisitions(current, "Question")
    target = GoverningOriginalTarget(
        instrument_number="5326", article="15", paragraph="3"
    )
    assert acquisitions.matching_originals(target, ledger) == [1]
    assert (
        acquisitions.matching_originals(
            target.model_copy(update={"paragraph": "4"}), ledger
        )
        == []
    )
    lower = original(
        "Uygulama Tebliği",
        "1",
        kind="tebliğ",
        text="5326 sayılı Kabahatler Kanunu m.15/3 kapsamında başvuru alınır.",
    )
    assert ledger.add([lower], current) == [2]
    projection = bound_identity_ledger(ledger, current)
    assert (
        canonical_cited_lower_statute_gap("Başvuru incelenmiştir [2].", projection, {2})
        is not None
    )
    assert (
        canonical_cited_lower_statute_gap(
            "Başvuru incelenmiştir [2].", projection, {1, 2}
        )
        is None
    )


def test_continuation_after_a_complete_two_paragraph_anchor() -> None:
    first, second, third = originals()
    first.text += "\n\n" + second.text
    first.text_hash = hashlib.sha256(first.text.encode()).hexdigest()
    metadata = first.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    metadata["paragraph_no"] = None
    third.metadata["position"] = 31
    current, ledger, state, receipt = setup([first, third])
    assert state.record_receipt(receipt, current, ledger) == 1
    assert list(state.identity_bindings(current, ledger)) == [2]
    assert (
        bound_named_authority_gap(
            ANSWER.replace("[3]", "[2]"), ledger, current, **OPTIONS
        )
        is None
    )
    assert (
        bound_named_authority_gap(
            ANSWER.replace("[3]", "[1]"), ledger, current, **OPTIONS
        )
        is not None
    )


def test_already_known_article_metadata_is_never_narrowed() -> None:
    items = originals()
    for item in items:
        metadata = item.metadata["canonical_metadata"]
        assert isinstance(metadata, dict)
        metadata["article_no"] = "15"
    current, ledger, state, receipt = setup(items)
    before = ledger.authority_metadata()
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.identity_bindings(current, ledger) == {}
    assert bound_identity_ledger(ledger, current).authority_metadata() == before


def test_plain_numbered_list_is_not_a_second_paragraph_anchor() -> None:
    first, _second, third = originals()
    first.text += "\n\n2. İşlem listesi maddesidir."
    first.text_hash = hashlib.sha256(first.text.encode()).hexdigest()
    third.metadata["position"] = 31
    current, ledger, state, receipt = setup([first, third])
    body = ANSWER.replace("[3]", "[2]")
    assert canonical_named_authority_gap(body, ledger, **OPTIONS) is not None
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.identity_bindings(current, ledger) == {}
    assert bound_named_authority_gap(body, ledger, current, **OPTIONS) is not None
    acquisitions = GoverningOriginalAcquisitions(current, "Question")
    target = GoverningOriginalTarget(
        instrument_number="5326", article="15", paragraph="3"
    )
    assert acquisitions.matching_originals(target, ledger) == []
