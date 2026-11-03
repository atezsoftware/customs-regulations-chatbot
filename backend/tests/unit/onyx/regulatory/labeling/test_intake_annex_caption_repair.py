"""Exact legacy annex caption formats retain current context and strict model proof."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling_semantic_repair as repository
from onyx.db.models import RegulatoryLabelingItem, RegulatoryLabelingRun
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling.intake_abstention_repair import (
    CaptionRepairSource,
    legacy_annex_caption_repair_candidate,
    legacy_annex_caption_shape,
)
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import (
    BINDING,
    REJECTED,
    audit,
    build,
    response,
    taxonomy,
    validate_audit,
)

# Exact source11 diagram body and following rule; fixture IDs only bind this cohort.
DIAGRAM = "2. Yoğuşturuculu çamaşır kurutma makineleri için etiket tasarımı\nŞekil 2\n96 mm\n2\n21 mm 2 mm 14 mm 3 mm\n3 mm 3 mm\n14 mm ENERG 14 mm 3\n1\n2 mm\n4 8 mm TRADEMARK MODEL IDENTIFIER 5\n4 mm 7\n8.5 mm A 22 mm\n1.5 mm\n8.5 mm B B 12 mm 8\n1.5 mm\n8.5 mm C\n1.5 mm\n6 D 77 mm\n8.5 mm\n1.5 mm\n8.5 mm E\n1.5 mm\n8.5 mm F\n1.5 mm\n8.5 mm G\n4.5 mm 192 mm\n7\n14 mm XYZ kWh / 100 14 mm 20 mm\n9\n6 mm 7\n10 mm 15 mm 22 mm\nA A\nB B\n16 mm C XYdB 10\n10 D XY% D\n5 mm\n7 mm 14 mm\n14 mm 14 mm 10\n10\nXY,Z kg X:YZ 11\n7 mm 2023/2534\n6 mm\n3 mm 3 mm\n27 mm 38 mm 27 mm\n24 mm\n28 mm\n32 mm\n36 mm\n40 mm\n44 mm"
DIMENSION_RULE = "a) Etiket en az 96 mm genişliğinde ve 192 mm yüksekliğinde olmalıdır. Etiketin daha büyük bir formatta basıldığı durumlarda, içeriği yine de Şekil 2'deki özelliklerle orantılı kalmalıdır."


def _source(
    identifier: str,
    text: str,
    position: int,
    section: str,
    number: str,
    *,
    clause: bool = False,
    body_title: bool = False,
) -> CaptionRepairSource:
    headings = ["Tebliğ", "EK iii", section]
    fields: dict[str, Any] = {}

    def fact(name: str, quote: str, scope: str) -> None:
        fields[name] = {
            "source": "llm",
            "basis": "explicit",
            "stale": False,
            "evidence_scope": scope,
            "evidence_quote": quote,
            "source_chunk_id": identifier,
        }

    fact(
        "paragraph_no",
        text if identifier == "target" and number == "1" else section,
        "body" if body_title else "heading",
    )
    fact("section_title", section, "body" if body_title else "heading")
    if clause:
        fact("clause_label", text.split(". ")[0], "body")
        headings.append(text[:30])
    metadata = {
        "heading_path": headings,
        "intake_chunking_version": "2",
        "chunk_variant": "atomic",
        "appendix_label": "EK iii",
        "section_title": section,
        "paragraph_no": number,
        "clause_label": "a" if clause else None,
        "intake_legal_metadata": {"input_stale": False, "fields": fields},
    }
    snapshot = {
        "position": position,
        "chunk_type": "clause" if clause else "paragraph",
        "heading_path": headings,
        "chunk_metadata": metadata,
        "validity_start_date": None,
        "validity_end_date": None,
    }
    return CaptionRepairSource(
        identifier,
        "owned-file",
        text,
        context_hash(text),
        snapshot,
        text,
        deepcopy(snapshot),
        True,
    )


def _sources(dimension: bool) -> list[CaptionRepairSource]:
    if dimension:
        section = "2. Yoğuşturuculu çamaşır kurutma makineleri için etiket tasarımı"
        sources = [
            _source("previous", DIAGRAM, 44, section, "2", body_title=True),
            _source("target", "48 mm\nEtiket:", 45, section, "2"),
            _source("following", DIMENSION_RULE, 46, section, "2", clause=True),
        ]
    else:
        section = "(1) Yedek parçaların bulunabilirliği"
        sources = [
            _source("target", section + ":", 42, section, "1", body_title=True),
            _source(
                "following",
                "a) Birimleri 1/7/2025 tarihinden itibaren piyasaya arz edilen tüm modeller için, ev tipi çamaşır kurutma makinesi imalatçıları, ithalatçıları veya yetkili temsilcileri, profesyonel tamirciler için en azından aşağıdaki yedek parçaları hazır bulundurur:",
                43,
                section,
                "1",
                clause=True,
            ),
        ]
    target = next(source for source in sources if source.chunk_id == "target")
    target.snapshot["context_member_ids"] = [source.chunk_id for source in sources]
    return sources


def _changed(
    source: CaptionRepairSource,
    *,
    text: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> CaptionRepairSource:
    body = source.text if text is None else text
    frozen = deepcopy(source.snapshot) if snapshot is None else snapshot
    return replace(
        source,
        text=body,
        canonical_text_sha256=context_hash(body),
        snapshot=frozen,
        current_text=body,
        current_snapshot=deepcopy(frozen),
    )


@pytest.mark.parametrize("dimension", [False, True])
def test_frozen_caption_bodies_with_positive_own_and_adjacent_witnesses(
    dimension: bool,
) -> None:
    sources = _sources(dimension)
    before = deepcopy(sources)
    assert legacy_annex_caption_shape(
        next(source.text for source in sources if source.chunk_id == "target")
    )
    assert legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)
    assert sources == before


@pytest.mark.parametrize("dimension", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "other-file",
        "other-heading",
        "other-article",
        "no-audit",
        "stale",
        "wrong-number",
        "own-quote",
        "clause-proof",
        "current-text",
        "current-metadata",
        "hash",
        "inactive",
        "multiscope",
        "operative",
        "citation",
    ],
)
def test_caption_requires_exact_current_owned_scope_and_positive_proof(
    dimension: bool, change: str
) -> None:
    sources = _sources(dimension)
    index = next(n for n, source in enumerate(sources) if source.chunk_id == "target")
    target, following = sources[index], sources[-1]
    snapshot = deepcopy(target.snapshot)
    metadata = snapshot["chunk_metadata"]
    if change == "missing":
        sources.pop()
    elif change == "other-file":
        sources[-1] = replace(following, file_id="different-owned-file")
    elif change == "other-heading":
        wrong = deepcopy(following.snapshot)
        wrong["heading_path"] = ["Another heading"]
        sources[-1] = _changed(following, snapshot=wrong)
    elif change == "other-article":
        wrong = deepcopy(following.snapshot)
        wrong["chunk_metadata"]["article_no"] = "9"
        sources[-1] = _changed(following, snapshot=wrong)
    elif change == "clause-proof":
        wrong = deepcopy(following.snapshot)
        wrong["chunk_metadata"].pop("intake_legal_metadata")
        sources[-1] = _changed(following, snapshot=wrong)
    elif change == "current-text":
        sources[-1] = replace(following, current_text=following.text + " changed")
    elif change == "current-metadata":
        wrong = deepcopy(following.current_snapshot)
        wrong["chunk_metadata"]["paragraph_no"] = "9"
        sources[-1] = replace(following, current_snapshot=wrong)
    elif change == "hash":
        sources[-1] = replace(following, canonical_text_sha256="f" * 64)
    elif change == "inactive":
        sources[-1] = replace(following, active=False)
    elif change in ("multiscope", "operative", "citation"):
        text = (
            target.text + "\n(2) Diğer hükümler"
            if change == "multiscope"
            else "(1) İthalatçı hazır bulundurulur:"
            if change == "operative"
            else "(1) Maddeye göre yedek parçalar:"
        )
        sources[index] = _changed(target, text=text)
    else:
        if change == "no-audit":
            metadata.pop("intake_legal_metadata")
        elif change == "stale":
            metadata["intake_legal_metadata"]["input_stale"] = True
        elif change == "wrong-number":
            metadata["paragraph_no"] = "9"
        else:
            metadata["intake_legal_metadata"]["fields"]["paragraph_no"][
                "source_chunk_id"
            ] = "following"
        sources[index] = _changed(target, snapshot=snapshot)
    assert not legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


@pytest.mark.parametrize(
    "change",
    [
        "figure",
        "ticks",
        "missing-previous",
        "arbitrary-number",
        "body-provenance",
        "figure-reference",
        "operative-previous",
    ],
)
def test_dimension_caption_requires_the_actual_adjacent_layout(change: str) -> None:
    sources = _sources(True)
    if change == "figure":
        sources[0] = _changed(sources[0], text=DIAGRAM.replace("Şekil 2", "Bir örnek"))
    elif change == "ticks":
        sources[0] = _changed(sources[0], text=DIAGRAM.removesuffix("44 mm") + "43 mm")
    elif change == "missing-previous":
        sources.pop(0)
    elif change == "arbitrary-number":
        sources[1] = _changed(sources[1], text="49 mm\nEtiket:")
    elif change == "figure-reference":
        sources[-1] = _changed(
            sources[-1], text=DIMENSION_RULE.replace("Şekil 2", "Şekil 3")
        )
    elif change == "operative-previous":
        sources[0] = _changed(
            sources[0], text=DIAGRAM.replace("Şekil 2\n", "Şekil 2\nVergi uygulanır\n")
        )
    else:
        wrong = deepcopy(sources[0].snapshot)
        wrong["chunk_metadata"]["intake_legal_metadata"]["fields"]["section_title"][
            "evidence_scope"
        ] = "heading"
        sources[0] = _changed(sources[0], snapshot=wrong)
    assert not legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


@pytest.mark.parametrize(
    "title",
    [
        "Yedek parçalar bulundurulur",
        "Maddeye göre yedek parçalar",
        "İthalatçı sağlar",
        "Vergi uygulanmaz",
        "Ücret ödemeden hizmet alınamaz",
        "İmalatçı yedek parçaları hazır bulundurur",
        "Başvuru sahibi belgeleri ibraz eder",
        "İthalatçı belgeleri saklar",
    ],
)
def test_own_audits_cannot_turn_prose_or_citations_into_a_caption(title: str) -> None:
    section = f"(1) {title}"
    sources = [
        _source("target", section + ":", 42, section, "1", body_title=True),
        _source(
            "following",
            "a) Yedek parçalar hazır bulundurulur.",
            43,
            section,
            "1",
            clause=True,
        ),
    ]
    sources[0].snapshot["context_member_ids"] = [source.chunk_id for source in sources]
    assert not legacy_annex_caption_shape(sources[0].text)
    assert not legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


@pytest.mark.parametrize(
    "prose",
    [
        "Üretici vergiyi öder",
        "ÜRETİCİ VERGİYİ ÖDER",
        "14 mm Üretici vergiyi öder 14 mm",
        "Vergi uygulanmaz",
        "İthalatçı belgeleri saklar",
        "A MODEL VERGİYİ ÖDER",
    ],
)
def test_diagram_witness_cannot_contain_prose_with_valid_own_audits(prose: str) -> None:
    sources = _sources(True)
    # Current and frozen source/hash move together; valid caption audits remain.
    sources[0] = _changed(
        sources[0], text=DIAGRAM.replace("Şekil 2\n", f"Şekil 2\n{prose}\n")
    )
    assert (
        sources[0].snapshot["chunk_metadata"]["intake_legal_metadata"]
        == _sources(True)[0].snapshot["chunk_metadata"]["intake_legal_metadata"]
    )
    assert not legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


def test_numbered_domain_caption_is_not_bound_to_one_number() -> None:
    section = "(7) Yedek parçaların bulunabilirliği"
    sources = [
        _source("target", section + ":", 42, section, "7", body_title=True),
        _source(
            "following",
            "a) Yedek parçalar hazır bulundurulur.",
            43,
            section,
            "7",
            clause=True,
        ),
    ]
    fields = sources[0].snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"]
    fields["paragraph_no"]["evidence_quote"] = section + ":"
    fields["paragraph_no"]["evidence_scope"] = "body"
    sources[0] = _changed(sources[0])
    sources[0].snapshot["context_member_ids"] = [source.chunk_id for source in sources]
    assert legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


@pytest.mark.parametrize("dimension", [False, True])
@pytest.mark.parametrize("intervening_heading", ["MADDE 9", "EK iv", "(2) Başka bölüm"])
def test_following_clause_must_be_a_direct_child_in_the_same_scope(
    dimension: bool, intervening_heading: str
) -> None:
    sources = _sources(dimension)
    following = sources[-1]
    snapshot = deepcopy(following.snapshot)
    headings = snapshot["heading_path"]
    nested_headings = [*headings[:-1], intervening_heading, headings[-1]]
    snapshot["heading_path"] = nested_headings
    snapshot["chunk_metadata"]["heading_path"] = nested_headings
    sources[-1] = _changed(following, snapshot=snapshot)
    assert sources[-1].snapshot == sources[-1].current_snapshot
    assert not legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


@pytest.mark.parametrize("dimension", [False, True])
@pytest.mark.parametrize(
    "child_heading", ["MADDE 9", "b) Başka bölüm", "a) Başka bir hüküm"]
)
def test_direct_child_heading_must_identify_the_actual_following_body(
    dimension: bool, child_heading: str
) -> None:
    sources = _sources(dimension)
    following = sources[-1]
    snapshot = deepcopy(following.snapshot)
    snapshot["heading_path"][-1] = child_heading
    snapshot["chunk_metadata"]["heading_path"] = snapshot["heading_path"]
    sources[-1] = _changed(following, snapshot=snapshot)
    assert sources[-1].snapshot == sources[-1].current_snapshot
    assert not legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


@pytest.mark.parametrize("dimension", [False, True])
@pytest.mark.parametrize("ellipsis", ["...", "…"])
def test_only_terminal_ellipsis_can_shorten_the_actual_child_heading(
    dimension: bool, ellipsis: str
) -> None:
    sources = _sources(dimension)
    following = sources[-1]
    snapshot = deepcopy(following.snapshot)
    snapshot["heading_path"][-1] = following.text[:40] + ellipsis
    snapshot["chunk_metadata"]["heading_path"] = snapshot["heading_path"]
    sources[-1] = _changed(following, snapshot=snapshot)
    assert legacy_annex_caption_repair_candidate(chunk_id="target", sources=sources)


@pytest.mark.parametrize("dimension", [False, True])
@pytest.mark.parametrize("drift", [False, True])
def test_initial_and_durable_acceptance_reload_the_frozen_context(
    monkeypatch: pytest.MonkeyPatch, dimension: bool, drift: bool
) -> None:
    file_id, run_id = uuid4(), uuid4()
    sources = [replace(source, file_id=str(file_id)) for source in _sources(dimension)]
    rows = []
    for source in sources:
        current = source.current_snapshot
        row = SimpleNamespace(
            id=source.chunk_id,
            user_file_id=file_id,
            text=source.current_text,
            status="active",
            superseded_by_chunk_id=None,
            position=current["position"],
            heading_path=current["heading_path"],
            chunk_type=current["chunk_type"],
            chunk_metadata=current["chunk_metadata"],
            validity_start_date=None,
            validity_end_date=None,
        )
        frozen = SimpleNamespace(
            text_snapshot=source.text,
            source_snapshot=source.snapshot,
            canonical_text_sha256=source.canonical_text_sha256,
        )
        rows.append((row, frozen))
    target_row, target_frozen = next(pair for pair in rows if pair[0].id == "target")
    item = SimpleNamespace(
        run_id=run_id,
        user_file_id=file_id,
        regulatory_chunk_id="target",
        text_snapshot=target_frozen.text_snapshot,
        canonical_text_sha256=target_frozen.canonical_text_sha256,
        source_snapshot=deepcopy(target_frozen.source_snapshot),
    )
    run = SimpleNamespace(id=run_id, file_ids=[str(file_id)], document_set_id=15)
    session = Mock(spec=Session)
    session.scalar.return_value = target_row
    session.execute.return_value.all.return_value = rows
    owned = Mock()
    monkeypatch.setattr(repository, "require_owned_intake_source", owned)
    if drift:
        rows[-1][0].text += " changed"
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
    owned.assert_called_once()
    repair = build(text=item.text_snapshot)
    item.request_payload, item.request_hash = (
        repair.base_request.model_dump(mode="json", exclude_computed_fields=True),
        repair.base_request.request_hash,
    )
    item.context_snapshot = json.loads(repair.base_request.prompt)[
        "interpretation_context"
    ]
    item.context_sha256 = context_hash(item.context_snapshot)
    item.labels, item.assignments = [], []
    run.taxonomy_id, run.provider_binding, run.model = (
        uuid4(),
        BINDING,
        BINDING["model"],
    )
    session.get_one.return_value = SimpleNamespace(
        definition=taxonomy().model_dump(mode="json"),
        version_hash=taxonomy().version_hash,
    )
    accepted = audit(repair, response(empty=True), text=item.text_snapshot)
    item.source_snapshot[repository.SEMANTIC_REPAIR_KEY] = accepted.model_dump(
        mode="json"
    )
    repository.validate_item_semantic_repair_audit(
        session,
        item=cast(RegulatoryLabelingItem, item),
        run=cast(RegulatoryLabelingRun, run),
    )
    rows[-1][0].text += " changed"
    with pytest.raises(ValueError, match="verified structural source"):
        repository.validate_item_semantic_repair_audit(
            session,
            item=cast(RegulatoryLabelingItem, item),
            run=cast(RegulatoryLabelingRun, run),
        )
