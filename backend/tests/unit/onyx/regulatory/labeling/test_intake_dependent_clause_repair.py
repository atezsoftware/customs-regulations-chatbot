"""Actual handling alternatives permit reconsideration, never inherited labels."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling_semantic_repair as repository
from onyx.db.models import RegulatoryLabelingItem, RegulatoryLabelingRun
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepair,
    build_abstention_repair,
    build_abstention_repair_audit,
    legacy_annex_caption_shape,
    parse_abstention_repair_output,
    validate_abstention_repair,
    validate_stored_abstention_repair_audit,
)
from onyx.regulatory.labeling.intake_dependent_clause_repair import (
    DependentClauseRepairSource,
    legacy_dependent_clause_repair_candidate,
    legacy_dependent_clause_shape,
)
from onyx.regulatory.labeling.intake_structural_repair import (
    legacy_structural_repair_candidate,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import line


@pytest.fixture
def frozen() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures/legacy_manual_handling_context.json"
    return json.loads(path.read_text())


def _sources(frozen: dict[str, Any]) -> list[DependentClauseRepairSource]:
    return [DependentClauseRepairSource(**row) for row in frozen["sources"]]


def _index(sources: list[DependentClauseRepairSource], position: int) -> int:
    return next(
        i for i, row in enumerate(sources) if row.snapshot["position"] == position
    )


def _changed(
    source: DependentClauseRepairSource,
    *,
    text: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> DependentClauseRepairSource:
    text = source.text if text is None else text
    snapshot = deepcopy(source.snapshot) if snapshot is None else snapshot
    return replace(
        source,
        text=text,
        canonical_text_sha256=context_hash(text),
        snapshot=snapshot,
        current_text=text,
        current_snapshot=deepcopy(snapshot),
    )


def _repair(frozen: dict[str, Any]) -> AbstentionRepair:
    target = next(
        row for row in _sources(frozen) if row.chunk_id == frozen["target_chunk_id"]
    )
    return build_abstention_repair(
        base_request=VertexBatchRequest.model_validate(frozen["base_request"]),
        rejected_response=frozen["base_response"],
        text=target.text,
        chunk_id=target.chunk_id,
        source_hash=target.canonical_text_sha256,
        taxonomy=TaxonomyDefinition.model_validate(frozen["taxonomy"]),
        provider_binding=frozen["provider_binding"],
    )


def test_actual_frozen_27_sources_bind_the_new_intro_and_closing_sibling(
    frozen: dict[str, Any],
) -> None:
    sources = _sources(frozen)
    before = deepcopy(sources)
    target = sources[_index(sources, 87)]
    assert len(sources) == 27 and legacy_dependent_clause_shape(target.text)
    assert legacy_dependent_clause_repair_candidate(
        chunk_id=target.chunk_id, sources=sources
    )
    assert not legacy_annex_caption_shape(target.text)
    assert not legacy_structural_repair_candidate(
        chunk_id=target.chunk_id, text=target.text, source_snapshot=target.snapshot
    )
    assert sources == before
    repair = _repair(frozen)
    original = json.loads(repair.base_request.prompt)
    prompt = json.loads(repair.request.prompt)
    context = original["interpretation_context"]
    assert len(context) == 15779
    for position in (81, 86, 87, 88, 89):
        assert sources[_index(sources, position)].text in context
    assert prompt["interpretation_context"] == context
    assert prompt["target"] == original["target"]
    assert repair.request.request_hash == frozen["prospective_repair_request_hash"]
    assert repair.request.generation_config == repair.base_request.generation_config


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "duplicate-position",
        "members",
        "foreign-file",
        "inactive",
        "current-text",
        "current-metadata",
        "hash",
        "wrong-intro",
        "extra-intro",
        "wrong-b",
        "negative-b",
        "wrong-clause",
        "parent-depth",
        "parent-metadata",
        "wrong-appendix",
        "article",
        "audit-missing",
        "audit-stale",
        "audit-owner",
        "body-audit",
        "fake-body-quote",
        "wrong-section",
        "wrong-section-number",
        "missing-scope-member",
        "intervening-article",
        "intervening-heading",
        "mixed-locator",
        "version",
    ],
)
def test_equal_current_snapshots_cannot_waive_list_or_scope_proof(
    frozen: dict[str, Any], change: str
) -> None:
    sources = _sources(frozen)
    index = _index(sources, 87)
    target = sources[index]
    if change in {"missing", "missing-scope-member"}:
        sources.pop(_index(sources, 88 if change == "missing" else 85))
        if change == "missing-scope-member":
            target.snapshot["context_member_ids"] = [row.chunk_id for row in sources]
    elif change == "duplicate":
        sources.append(target)
    elif change == "members":
        target.snapshot["context_member_ids"].append("absent")
    elif change == "duplicate-position":
        snapshot = deepcopy(sources[0].snapshot)
        snapshot["position"] = 87
        sources[0] = _changed(sources[0], snapshot=snapshot)
    elif change in {
        "foreign-file",
        "inactive",
        "current-text",
        "current-metadata",
        "hash",
    }:
        source = sources[0]
        if change == "foreign-file":
            sources[0] = replace(source, file_id=str(uuid4()))
        elif change == "inactive":
            sources[0] = replace(source, active=False)
        elif change == "current-text":
            sources[0] = replace(source, current_text=source.text + " changed")
        elif change == "current-metadata":
            snapshot = deepcopy(source.current_snapshot)
            snapshot["chunk_metadata"]["paragraph_no"] = "9"
            sources[0] = replace(source, current_snapshot=snapshot)
        else:
            sources[0] = replace(source, canonical_text_sha256="f" * 64)
    elif change in {"wrong-intro", "extra-intro"}:
        index = _index(sources, 86)
        text = sources[index].text
        text = (
            text.replace("elle taşınacağı", "elle taşınmayacağı")
            if change == "wrong-intro"
            else text + "\nVergi ödenir."
        )
        sources[index] = _changed(sources[index], text=text)
    elif change in {"wrong-b", "negative-b", "wrong-section-number"}:
        index = _index(sources, 88)
        old, new = {
            "wrong-b": ("(b)", "(c)"),
            "negative-b": ("donatılır.", "donatılmaz."),
            "wrong-section-number": ("1.1.6.", "1.2.6."),
        }[change]
        sources[index] = _changed(
            sources[index], text=sources[index].text.replace(old, new)
        )
    elif change in {"wrong-section", "intervening-article"}:
        index = _index(sources, 81 if change == "wrong-section" else 85)
        text = (
            sources[index].text.replace(
                "taşınmasını kolaylaştıracak şekilde tasarımı", "vergilendirilmesi"
            )
            if change == "wrong-section"
            else sources[index].text + "\nMADDE 9 – Başka hüküm."
        )
        sources[index] = _changed(sources[index], text=text)
    else:
        index = _index(
            sources,
            88
            if change in {"body-audit", "fake-body-quote"}
            else 85
            if change == "intervening-heading"
            else 87,
        )
        source = sources[index]
        snapshot = deepcopy(source.snapshot)
        metadata = snapshot["chunk_metadata"]
        if change in {"parent-depth", "intervening-heading"}:
            snapshot["heading_path"].insert(-1, "MADDE 9")
            metadata["heading_path"] = deepcopy(snapshot["heading_path"])
        elif change == "parent-metadata":
            metadata["heading_path"] = ["Another scope"]
        elif change == "wrong-appendix":
            metadata["appendix_label"] = "EK 4"
        elif change in {"article", "mixed-locator"}:
            metadata["article_no" if change == "article" else "paragraph_no"] = "9"
        elif change == "version":
            metadata["intake_chunking_version"] = "4"
        elif change == "wrong-clause":
            metadata["clause_label"] = "c"
        elif change == "audit-missing":
            metadata.pop("intake_legal_metadata")
        elif change == "audit-stale":
            metadata["intake_legal_metadata"]["input_stale"] = True
        else:
            fact = metadata["intake_legal_metadata"]["fields"]["clause_label"]
            if change == "audit-owner":
                fact["source_chunk_id"] = "another-source"
            elif change == "body-audit":
                fact["evidence_scope"] = "heading"
            else:
                fact["evidence_quote"] = "(b)"
        sources[index] = _changed(source, snapshot=snapshot)
    assert not legacy_dependent_clause_repair_candidate(
        chunk_id=target.chunk_id, sources=sources
    )


def test_real_abstention_remains_failed_and_quotes_stay_target_only(
    frozen: dict[str, Any],
) -> None:
    repair = _repair(frozen)
    text = json.loads(repair.base_request.prompt)["target"]["text"]
    taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
    assert validate_abstention_repair(
        frozen["base_response"], repair=repair, text=text, taxonomy=taxonomy
    ).abstained
    output = parse_abstention_repair_output(
        [line(repair, frozen["base_response"])], repair=repair, on_progress=lambda: None
    )
    audit = build_abstention_repair_audit(
        output=output,
        repair=repair,
        output_uri="fixture://continued",
        text=text,
        taxonomy=taxonomy,
    )
    with pytest.raises(ValueError, match="accepted assignments"):
        validate_stored_abstention_repair_audit(
            audit,
            base_request=repair.base_request,
            text=text,
            chunk_id=repair.chunk_id,
            source_hash=repair.canonical_text_sha256,
            taxonomy=taxonomy,
            provider_binding=repair.provider_binding,
            labels=[],
            assignments=[],
        )
    for label, quote in (
        ("not-in-taxonomy", text),
        (taxonomy.labels[0].id, "donatılır."),
    ):
        response = json.dumps(
            {
                "labels": [{"label_id": label, "evidence_quote": quote}],
                "abstained": False,
            }
        )
        with pytest.raises(ValueError):
            validate_abstention_repair(
                response, repair=repair, text=text, taxonomy=taxonomy
            )


@pytest.mark.parametrize("drift", [False, True])
def test_initial_and_stored_receipt_checks_reload_the_full_current_context(
    frozen: dict[str, Any], monkeypatch: pytest.MonkeyPatch, drift: bool
) -> None:
    sources = _sources(frozen)
    rows = []
    for source in sources:
        current = source.current_snapshot
        row = SimpleNamespace(
            id=source.chunk_id,
            user_file_id=UUID(source.file_id),
            text=source.current_text,
            status="active",
            superseded_by_chunk_id=None,
            **current,
        )
        saved = SimpleNamespace(
            text_snapshot=source.text,
            source_snapshot=source.snapshot,
            canonical_text_sha256=source.canonical_text_sha256,
        )
        rows.append((row, saved))
    row, saved = rows[_index(sources, 87)]
    run = SimpleNamespace(
        id=uuid4(),
        file_ids=[str(row.user_file_id)],
        document_set_id=15,
        taxonomy_id=uuid4(),
        provider_binding=frozen["provider_binding"],
        model=frozen["provider_binding"]["model"],
    )
    item = SimpleNamespace(
        run_id=run.id,
        user_file_id=row.user_file_id,
        regulatory_chunk_id=row.id,
        text_snapshot=saved.text_snapshot,
        canonical_text_sha256=saved.canonical_text_sha256,
        source_snapshot=deepcopy(saved.source_snapshot),
    )
    session = Mock(spec=Session)
    session.scalar.return_value, session.execute.return_value.all.return_value = (
        row,
        rows,
    )
    owned = Mock()
    monkeypatch.setattr(repository, "require_owned_intake_source", owned)
    if drift:
        rows[_index(sources, 86)][0].text += " changed"
        with pytest.raises(ValueError, match="verified structural source"):
            repository.require_current_intake_repair_item(
                session,
                item=cast(RegulatoryLabelingItem, item),
                run=cast(RegulatoryLabelingRun, run),
            )
        owned.assert_not_called()
        return
    repository.require_current_intake_repair_item(
        session,
        item=cast(RegulatoryLabelingItem, item),
        run=cast(RegulatoryLabelingRun, run),
    )
    owned.assert_called_once_with(
        session, user_file_id=row.user_file_id, document_set_id=15
    )
    repair = _repair(frozen)
    taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
    item.request_payload, item.request_hash = (
        frozen["base_request"],
        repair.base_request.request_hash,
    )
    item.context_snapshot = json.loads(repair.base_request.prompt)[
        "interpretation_context"
    ]
    item.context_sha256, item.labels, item.assignments = (
        context_hash(item.context_snapshot),
        [],
        [],
    )
    session.get_one.return_value = SimpleNamespace(
        definition=frozen["taxonomy"], version_hash=taxonomy.version_hash
    )
    output = parse_abstention_repair_output(
        [line(repair, '{"labels":[],"abstained":false}')],
        repair=repair,
        on_progress=lambda: None,
    )
    accepted = build_abstention_repair_audit(
        output=output,
        repair=repair,
        output_uri="fixture://model-decision",
        text=item.text_snapshot,
        taxonomy=taxonomy,
    )
    item.source_snapshot[repository.SEMANTIC_REPAIR_KEY] = accepted.model_dump(
        mode="json"
    )
    repository.validate_item_semantic_repair_audit(
        session,
        item=cast(RegulatoryLabelingItem, item),
        run=cast(RegulatoryLabelingRun, run),
    )
    rows[_index(sources, 88)][0].text += " changed"
    with pytest.raises(ValueError, match="verified structural source"):
        repository.validate_item_semantic_repair_audit(
            session,
            item=cast(RegulatoryLabelingItem, item),
            run=cast(RegulatoryLabelingRun, run),
        )


@pytest.fixture
def requirement_lists() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures/legacy_requirement_lists_context.json"
    return json.loads(path.read_text())


def _requirement_case(frozen: dict[str, Any], position: int) -> dict[str, Any]:
    case = next(row for row in frozen["cases"] if row["position"] == position)
    target = next(
        row for row in frozen["sources"] if row["chunk_id"] == case["target_chunk_id"]
    )
    members = set(target["snapshot"]["context_member_ids"])
    return {
        **frozen,
        **case,
        "sources": [row for row in frozen["sources"] if row["chunk_id"] in members],
    }


@pytest.mark.parametrize("position", [362, 364, 369, 371])
def test_actual_requirement_lists_keep_body_scope_and_request_authority(
    requirement_lists: dict[str, Any], position: int
) -> None:
    frozen = _requirement_case(requirement_lists, position)
    sources = _sources(frozen)
    target = sources[_index(sources, position)]
    assert len(sources) == 27
    assert target.snapshot["heading_path"][2] == "1.2.4"
    assert (
        "section_title"
        not in target.snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"]
    )
    assert legacy_dependent_clause_repair_candidate(
        chunk_id=target.chunk_id, sources=sources
    )
    assert not legacy_annex_caption_shape(target.text)
    repair = _repair(frozen)
    context = json.loads(repair.base_request.prompt)["interpretation_context"]
    assert all(source.text in context for source in sources)
    assert repair.request.request_hash == frozen["prospective_repair_request_hash"]


@pytest.mark.parametrize("position", [362, 369])
@pytest.mark.parametrize(
    "change",
    [
        "missing-sibling",
        "sequence",
        "foreign",
        "depth",
        "article",
        "no-intro",
        "negative-intro",
        "extra-intro",
        "no-body-section",
        "wrong-title",
        "wrong-boundary",
        "no-boundary",
        "intervening-article",
        "no-audit",
        "heading-audit",
        "different-owner",
        "short-quote",
        "stale",
        "hash",
        "current-text",
        "inactive",
    ],
)
def test_requirement_lists_need_complete_current_own_body_evidence(
    requirement_lists: dict[str, Any], position: int, change: str
) -> None:
    frozen = _requirement_case(requirement_lists, position)
    sources = _sources(frozen)
    target = sources[_index(sources, position)]
    intro_position = position - 1
    intro_index = _index(sources, intro_position)
    last_index = _index(sources, position + 3)
    if change == "missing-sibling":
        sources.pop(_index(sources, position + 1))
        target.snapshot["context_member_ids"] = [source.chunk_id for source in sources]
    elif change in {
        "no-intro",
        "negative-intro",
        "extra-intro",
        "no-body-section",
        "wrong-title",
        "intervening-article",
    }:
        source = sources[intro_index]
        text = source.text
        if change == "no-intro":
            text = text.rsplit(":", 1)[0] + "."
        elif change == "negative-intro":
            text = text.replace("gösterir:", "göstermez:").replace(
                "eşlik eder:", "eşlik etmez:"
            )
        elif change == "extra-intro":
            text += "\nBaşka bir yükümlülük uygulanır."
        elif change == "no-body-section":
            text = text.replace(
                "4.3.1. Zincirler, halatlar ve kayışlar", "1.2.4"
            ).replace("4.4.1. Kaldırma aksesuarları", "1.2.4")
        elif change == "wrong-title":
            text = text.replace(
                "4.3.1. Zincirler, halatlar ve kayışlar", "4.3.1. Vergilendirme"
            ).replace("4.4.1. Kaldırma aksesuarları", "4.4.1. Vergilendirme")
        else:
            text = text.replace(
                "4.3.1. Zincirler, halatlar ve kayışlar",
                "4.3.1. Zincirler, halatlar ve kayışlar\nMADDE 9 – Başka hüküm.",
            ).replace(
                "4.4.1. Kaldırma aksesuarları",
                "4.4.1. Kaldırma aksesuarları\nMADDE 9 – Başka hüküm.",
            )
        sources[intro_index] = _changed(source, text=text)
    elif change in {"wrong-boundary", "no-boundary"}:
        source = sources[last_index]
        text = (
            source.text.replace("4.3.2.", "4.3.3.").replace("4.4.2.", "4.4.3.")
            if change == "wrong-boundary"
            else source.text.splitlines()[0]
        )
        sources[last_index] = _changed(source, text=text)
    elif change in {"foreign", "hash", "current-text", "inactive"}:
        source = sources[last_index]
        key, value = {
            "foreign": ("file_id", str(uuid4())),
            "hash": ("canonical_text_sha256", "f" * 64),
            "current-text": ("current_text", source.text + " changed"),
            "inactive": ("active", False),
        }[change]
        sources[last_index] = replace(source, **{key: value})
    else:
        index = _index(sources, position + 1 if change == "sequence" else position)
        source = sources[index]
        snapshot = deepcopy(source.snapshot)
        metadata = snapshot["chunk_metadata"]
        fact = metadata["intake_legal_metadata"]["fields"]["clause_label"]
        if change == "sequence":
            metadata["clause_label"] = "c"
        elif change == "depth":
            snapshot["heading_path"].insert(-1, "MADDE 9")
            metadata["heading_path"] = deepcopy(snapshot["heading_path"])
        elif change == "article":
            metadata["article_no"] = "9"
        elif change == "no-audit":
            metadata.pop("intake_legal_metadata")
        elif change == "heading-audit":
            fact["evidence_scope"] = "heading"
        elif change == "different-owner":
            fact["source_chunk_id"] = sources[intro_index].chunk_id
        elif change == "short-quote":
            fact["evidence_quote"] = "(a)"
        else:
            fact["stale"] = True
        sources[index] = _changed(source, snapshot=snapshot)
    assert not legacy_dependent_clause_repair_candidate(
        chunk_id=target.chunk_id, sources=sources
    )
