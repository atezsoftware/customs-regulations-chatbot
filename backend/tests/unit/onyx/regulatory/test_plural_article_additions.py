from unittest.mock import MagicMock

import pytest

from onyx.db import regulatory_amendment_order
from onyx.regulatory.amendments import pipeline
from onyx.regulatory.amendments.draft_integrity import (
    DraftIntegrityError,
    explicit_added_article_identity,
    explicit_added_body,
    validate_added_article_draft,
)
from onyx.regulatory.amendments.insertion_order import OrderMember
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    ChunkFieldsDraft,
    DateResolution,
    DraftResult,
    MatchResult,
)
from onyx.regulatory.amendments.new_provision_policy import (
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.provision_identity import article_identity


def test_unclosed_quoted_added_article_preserves_its_own_identity() -> None:
    body = (
        "Destek ödemesi işlemleri ile ilgili inceleme yapılması\n"
        "MADDE 6/A- (1) Merkez Bankası bilgi ve belge talep edebilir.\n"
        "(2) İnceleme süresince destek ödenmez."
    )
    instruction = (
        "MADDE 5- Aynı Tebliğe 6 ncı maddesinden sonra gelmek üzere "
        "aşağıdaki maddeler eklenmiştir.\n“" + body
    )
    assert explicitly_adds_top_level_provision(instruction)
    assert explicit_added_body(instruction) == body
    assert explicit_added_article_identity(instruction) == "6/A"


@pytest.mark.parametrize(
    ("command", "heading", "identity"),
    [
        (
            "6 ncı maddesinden sonra gelmek üzere aşağıdaki maddeler eklenmiştir.",
            "MADDE 6/A",
            "6/A",
        ),
        (
            "6 ncı maddesinden sonra gelmek üzere aşağıdaki maddeler eklenmiştir.",
            "MADDE 6/B",
            "6/B",
        ),
        ("aşağıdaki geçici maddeler eklenmiştir.", "GEÇİCİ MADDE 2", "GEÇİCİ 2"),
        ("aşağıdaki geçici maddeler eklenmiştir.", "GEÇİCİ MADDE 3", "GEÇİCİ 3"),
    ],
)
def test_plural_addition_reaches_draft_with_its_own_identity(
    monkeypatch: pytest.MonkeyPatch, command: str, heading: str, identity: str
) -> None:
    body = f"Uygulama esasları {heading}- (1) Bu Tebliğin 4/A maddesi uygulanır."
    instruction = AmendmentInstruction(
        instruction_text=f"MADDE 5- Aynı Tebliğe {command} “{body}”",
        article_reference="Madde 6",  # A neighbour must not become the new identity.
        target_source="Örnek Tebliğ (Sayı: 2023/5)",
    )
    assert instruction.target_source is not None
    candidate = CandidateChunk(
        chunk_id="neighbour",
        user_file_id="00000000-0000-0000-0000-000000000123",
        source_name=instruction.target_source,
        source_verified=True,
        text="MADDE 6- Eski hüküm.",
        metadata={"article_no": "6", "heading_path": ["Örnek Tebliğ", "MADDE 6"]},
    )
    match = MatchResult(
        outcome="new_provision",
        old_chunk_id=None,
        confidence=1,
        rationale="New article",
    )
    monkeypatch.setattr(pipeline, "confirm_match", lambda *_args, **_kwargs: match)
    monkeypatch.setattr(pipeline, "get_next_chunk_position", lambda *_args: 12)
    monkeypatch.setattr(
        regulatory_amendment_order,
        "load_amendment_order",
        lambda *_args: [
            OrderMember(id="neighbour", position=4, article_no="6"),
            OrderMember(id="next", position=5, article_no="7"),
            OrderMember(id="temp", position=6, article_no="GEÇİCİ 1"),
        ],
    )
    confirmed = pipeline.confirm_instruction_match(
        MagicMock(), instruction=instruction, candidates=[candidate]
    )
    assert confirmed == match
    context = pipeline.load_instruction_draft_context(
        MagicMock(), instruction=instruction, candidates=[candidate], match=match
    )
    assert context is not None
    assert context.expected_new_article_no == identity
    assert context.old_chunk_snapshot == {}
    assert context.sibling_reference is not None
    assert context.sibling_reference["metadata"]["article_no"] == "6"
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text=body,
            chunk_type="article",
            heading_path=["Örnek Tebliğ", heading],
            metadata_changes={"article_no": identity},
        ),
        dates=DateResolution(rationale="No effective date supplied"),
    )
    proposal = pipeline._build_proposal_draft(
        instruction_indices=[0],
        instructions=[instruction],
        matches=[match],
        context=context,
        draft=draft,
    )
    assert proposal.old_chunk_id is None
    assert proposal.new_chunk_draft["metadata"]["article_no"] == identity
    assert proposal.new_chunk_draft["text"] == body
    assert proposal.new_chunk_draft["insertion_order"]["article_no"] == identity
    assert proposal.new_chunk_draft["position"] == (
        5 if identity.startswith("6/") else 7
    )
    wrong = draft.model_copy(
        update={
            "new_chunk": draft.new_chunk.model_copy(
                update={
                    "metadata_changes": {"article_no": "6"},
                    "heading_path": ["MADDE 6"],
                }
            )
        }
    )
    with pytest.raises(DraftIntegrityError, match="identity"):
        pipeline._build_proposal_draft(
            instruction_indices=[0],
            instructions=[instruction],
            matches=[match],
            context=context,
            draft=wrong,
        )


@pytest.mark.parametrize(
    "text",
    [
        "MADDE 5- Aynı Tebliğin 6 ncı maddesine aşağıdaki fıkralar eklenmiştir.",
        "MADDE 5- Aynı Tebliğin aşağıdaki maddeleri yürürlükten kaldırılmıştır.",
        "MADDE 5- Aynı Tebliğin 6 ncı maddesinde yer alan ‘maddeler’ ibaresi değiştirilmiştir.",
        "MADDE 5- Aynı Tebliğin 6 ncı maddesi aşağıdaki şekilde değiştirilmiştir. “Aşağıdaki maddeler eklenmiştir.”",
    ],
)
def test_non_article_additions_are_not_promoted(text: str) -> None:
    assert not explicitly_adds_top_level_provision(text)


@pytest.mark.parametrize(
    ("heading", "expected"),
    [("MADDE 6/A", "6/A"), ("MADDE 6/B", "6/B"), ("GEÇİCİ MADDE 6/A", "GEÇİCİ 6/A")],
)
def test_slash_identity_is_not_collapsed_to_neighbour(
    heading: str, expected: str
) -> None:
    assert article_identity(heading) == expected


@pytest.mark.parametrize(
    ("text", "identity"),
    [("6/A maddesinin", "6/A"), ("geçici 6/A maddesinin", "GEÇİCİ 6/A")],
)
def test_existing_slash_article_reference(text: str, identity: str) -> None:
    assert article_identity(text) == identity


@pytest.mark.parametrize(
    ("identity", "heading", "target", "valid"),
    [
        ("4", "MADDE 4", None, False),
        ("4/A", "MADDE 4", None, False),
        ("4/A", "MADDE 4/A", "existing", False),
        ("4/A", "MADDE 4/A", None, True),
    ],
)
def test_persisted_new_article_is_revalidated_at_approval(
    identity: str, heading: str, target: str | None, valid: bool
) -> None:
    instructions = [
        "MADDE 3- Aynı Tebliğe 4 üncü maddesinden sonra gelmek üzere aşağıdaki madde eklenmiştir. “MADDE 4/A- (1) Yeni hüküm.”"
    ]
    if valid:
        validate_added_article_draft(
            instructions,
            metadata={"article_no": identity},
            heading_path=[heading],
            old_chunk_id=target,
            insertion_order={"article_no": "4/A"},
        )
    else:
        with pytest.raises(DraftIntegrityError):
            validate_added_article_draft(
                instructions,
                metadata={"article_no": identity},
                heading_path=[heading],
                old_chunk_id=target,
            )


@pytest.mark.parametrize("identity", ["4", "4/A"])
def test_queue_rejects_legacy_wrong_identity_or_missing_order_before_db_writes(
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    from sqlalchemy.orm import Session

    from onyx.db import regulatory_amendments
    from onyx.db.models import AmendmentProposal

    proposal = AmendmentProposal(
        id=415,
        status="pending",
        old_chunk_id=None,
        old_chunk_snapshot={},
        instruction_index=2,
        instruction_text="MADDE 3- Aynı Tebliğe aşağıdaki madde eklenmiştir. “MADDE 4/A- Yeni hüküm.”",
        new_chunk_draft={
            "user_file_id": "00000000-0000-0000-0000-000000000123",
            "position": 12,
            "text": "MADDE 4/A- Yeni hüküm.",
            "chunk_type": "article",
            "metadata": {"article_no": identity},
            "heading_path": [f"MADDE {identity}"],
        },
    )
    monkeypatch.setattr(
        regulatory_amendments, "_lock_proposal_for_transition", lambda *_args: proposal
    )
    session = MagicMock(spec=Session)
    with pytest.raises(DraftIntegrityError, match="4/A|insertion order"):
        regulatory_amendments.queue_amendment_proposal_approval(
            session, proposal, decided_by=None
        )
    assert proposal.status == "pending"
    session.add.assert_not_called()
    session.flush.assert_not_called()
