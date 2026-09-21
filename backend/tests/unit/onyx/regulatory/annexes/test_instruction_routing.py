from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.db import regulatory_annex_changes, regulatory_annexes
from onyx.db.models import AmendmentBatch, RegulatoryChunk
from onyx.regulatory.amendments.annexes.analysis import group_annex_instructions
from onyx.regulatory.amendments.models import AmendmentInstruction


def test_explicit_edit_does_not_become_document_comparison_when_target_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_target(*_args: object, **_kwargs: object) -> object:
        raise ValueError("annex_file_missing")

    monkeypatch.setattr(
        regulatory_annex_changes, "resolve_annex_instruction_file", missing_target
    )
    group = group_annex_instructions(
        [
            AmendmentInstruction(
                instruction_text="Ek-2 listesinin 26 ncı sırası yürürlükten kaldırılmıştır.",
                article_reference="Ek-2",
                annex_change_basis="explicit_amendment",
            )
        ]
    )[0]
    assert regulatory_annex_changes.can_draft_annex_from_instructions(
        MagicMock(),
        batch=AmendmentBatch(document_set_id=7, source_package_id=uuid4()),
        group=group,
        reference_date=date(2027, 1, 1),
    )


def test_annex_file_resolution_scopes_only_matching_source_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated_id, target_id = uuid4(), uuid4()
    session = MagicMock()
    session.execute.return_value = [
        (unrelated_id, "unrelated.md"),
        (target_id, "2026-02_makinalar.md"),
    ]
    scoped: list[object] = []

    def require_scope(_session: object, _set_id: int, file_id: object) -> object:
        assert file_id == target_id, (
            "Do not issue a scope query for every unrelated file"
        )
        scoped.append(file_id)
        return MagicMock(name="2026-02_makinalar.md")

    monkeypatch.setattr(
        regulatory_annex_changes, "require_annex_file_scope", require_scope
    )
    monkeypatch.setattr(
        regulatory_annexes,
        "load_legacy_annex_chunks",
        lambda *_args, **_kwargs: [object()],
    )
    result = regulatory_annex_changes.resolve_annex_instruction_file(
        session,
        batch=AmendmentBatch(
            document_set_id=7, user_file_ids=[str(unrelated_id), str(target_id)]
        ),
        annex_label="ek:2",
        target_sources=["Makinalar 2026/2"],
        effective_date=date(2027, 1, 1),
    )
    assert result == target_id
    assert scoped == [target_id]


@pytest.mark.parametrize("has_package", [False, True])
@pytest.mark.parametrize("chunk_count", [1, 14])
def test_explicit_annex_edits_use_chunks_regardless_of_input_format(
    monkeypatch: pytest.MonkeyPatch, has_package: bool, chunk_count: int
) -> None:
    file_id = uuid4()
    batch = AmendmentBatch(
        id=96,
        document_set_id=7,
        source_package_id=uuid4() if has_package else None,
        user_file_ids=[str(file_id)],
    )
    rows = [
        RegulatoryChunk(
            id=f"row-{index}",
            user_file_id=file_id,
            text=f"Row {index}",
            chunk_type="table",
            chunk_metadata={"appendix_label": "EK 2"},
        )
        for index in range(chunk_count)
    ]
    monkeypatch.setattr(
        regulatory_annex_changes,
        "resolve_annex_instruction_file",
        lambda *_args, **_kwargs: file_id,
    )
    monkeypatch.setattr(
        regulatory_annexes, "load_legacy_annex_chunks", lambda *_args, **_kwargs: rows
    )
    instructions = [
        AmendmentInstruction(instruction_text=text, article_reference="Ek-2")
        for text in [
            "MADDE 16- Aynı Tebliğin Ek-2’sinde yer alan listenin 26 ncı sırası yürürlükten kaldırılmıştır.",
            "MADDE 17- Aynı Tebliğin Ek-2’sinde yer alan listeye aşağıdaki sıra eklenmiştir.\n26. 8429.11.00.00.00 Paletli olanlar Makina, Emisyon, Gürültü",
        ]
    ]
    group = group_annex_instructions(instructions)[0]
    assert regulatory_annex_changes.can_draft_annex_from_instructions(
        MagicMock(), batch=batch, group=group, reference_date=date(2027, 1, 1)
    )


def test_attached_replacement_still_requires_new_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_id = uuid4()
    monkeypatch.setattr(
        regulatory_annex_changes,
        "resolve_annex_instruction_file",
        lambda *_args, **_kwargs: file_id,
    )
    monkeypatch.setattr(
        regulatory_annexes,
        "load_legacy_annex_chunks",
        lambda *_args, **_kwargs: [
            RegulatoryChunk(
                id="old",
                text="Existing annex",
                chunk_type="table",
                chunk_metadata={"appendix_label": "EK 2"},
            )
        ],
    )
    group = group_annex_instructions(
        [
            AmendmentInstruction(
                instruction_text="MADDE 1- Aynı Tebliğin Ek-2’si ekteki şekilde değiştirilmiştir.",
                article_reference="Ek-2",
            )
        ]
    )[0]
    assert not regulatory_annex_changes.can_draft_annex_from_instructions(
        MagicMock(),
        batch=AmendmentBatch(document_set_id=7, source_package_id=uuid4()),
        group=group,
        reference_date=date(2027, 1, 1),
    )


@pytest.mark.parametrize("basis", [None, "replacement_document"])
@pytest.mark.parametrize(
    "replacement_text",
    [
        "Ek-2 tablosunun yeni hali aşağıdadır; farkları çıkar.\nEk-2\n| 1 | Yeni ürün |",
        "Ek-2 aşağıdaki şekilde değiştirilmiştir.\nEk-2\n| 1 | Yeni ürün |",
    ],
)
def test_supplied_new_annex_is_compared_even_when_its_text_is_inline(
    monkeypatch: pytest.MonkeyPatch,
    basis: str | None,
    replacement_text: str,
) -> None:
    file_id = uuid4()
    monkeypatch.setattr(
        regulatory_annex_changes,
        "resolve_annex_instruction_file",
        lambda *_args, **_kwargs: file_id,
    )
    monkeypatch.setattr(
        regulatory_annexes,
        "load_legacy_annex_chunks",
        lambda *_args, **_kwargs: [
            RegulatoryChunk(
                id="old",
                text="Old rows",
                chunk_type="table",
                chunk_metadata={"appendix_label": "EK 2"},
            )
        ],
    )
    instruction = AmendmentInstruction.model_validate(
        {
            "instruction_text": replacement_text,
            "article_reference": "Ek-2",
            "annex_change_basis": basis,
        }
    )
    group = group_annex_instructions([instruction])[0]
    assert not regulatory_annex_changes.can_draft_annex_from_instructions(
        MagicMock(),
        batch=AmendmentBatch(document_set_id=7, source_package_id=None),
        group=group,
        reference_date=date(2027, 1, 1),
    )


def test_qualified_articles_and_quoted_annex_references_are_not_annex_groups() -> None:
    from onyx.regulatory.amendments.annexes.analysis import group_annex_instructions
    from onyx.regulatory.amendments.models import AmendmentInstruction

    instructions = [
        AmendmentInstruction(
            instruction_text="Kanunun ek 3 üncü maddesi değiştirilmiştir.",
            article_reference="Ek 3 üncü madde",
        ),
        AmendmentInstruction(
            instruction_text="Kanuna aşağıdaki geçici madde eklenmiştir.\n“GEÇİCİ MADDE 20- EK-3 kapsamındakiler.”",
            article_reference="Geçici Madde 20",
        ),
        AmendmentInstruction(
            instruction_text="Tebliğin EK-IV/A eki değiştirilmiştir.",
            article_reference="EK-IV/A",
        ),
    ]
    groups = group_annex_instructions(instructions)
    assert len(groups) == 1
    assert groups[0].instruction_indices == [2]
