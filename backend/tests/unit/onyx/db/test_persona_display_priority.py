from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import MultipleResultsFound

from onyx.auth.schemas import UserRole
from onyx.db.models import User
from onyx.db.persona import (
    get_best_persona_id_for_user,
    update_personas_display_priority,
)


def _persona(persona_id: int, display_priority: int | None) -> SimpleNamespace:
    return SimpleNamespace(id=persona_id, display_priority=display_priority)


class _AccessiblePersonaResult:
    def __init__(self, statement: object, personas: list[SimpleNamespace]) -> None:
        self._statement = statement
        self._personas = personas

    def one_or_none(self) -> SimpleNamespace | None:
        limit_clause = getattr(self._statement, "_limit_clause", None)
        limit = getattr(limit_clause, "value", None)
        order_by = [
            str(clause) for clause in getattr(self._statement, "_order_by_clauses", ())
        ]
        priority_descending = bool(order_by) and order_by[0].startswith(
            "persona.display_priority DESC"
        )
        nulls_last = bool(order_by) and order_by[0].endswith("NULLS LAST")
        lower_id_tiebreaker = len(order_by) > 1 and order_by[1] == "persona.id ASC"

        def sort_key(persona: SimpleNamespace) -> tuple[int, int, int]:
            priority = persona.display_priority
            null_rank = (
                int(priority is None) if nulls_last else int(priority is not None)
            )
            priority_rank = (
                -priority
                if priority_descending and priority is not None
                else priority or 0
            )
            id_rank = persona.id if lower_id_tiebreaker else 0
            return null_rank, priority_rank, id_rank

        personas = sorted(
            self._personas,
            key=sort_key,
        )
        selected = personas[:limit] if isinstance(limit, int) else personas
        if len(selected) > 1:
            raise MultipleResultsFound()
        if not selected:
            return None
        return selected[0]


def test_best_persona_selects_highest_priority_when_multiple_are_accessible() -> None:
    lower_priority = _persona(11, 10)
    higher_priority = _persona(12, 20)
    db_session = MagicMock()
    db_session.scalars.side_effect = lambda statement: _AccessiblePersonaResult(
        statement, [lower_priority, higher_priority]
    )
    user = SimpleNamespace(
        id=uuid4(),
        is_anonymous=False,
        role=UserRole.BASIC,
    )

    selected_id = get_best_persona_id_for_user(db_session, cast(User, user))

    assert selected_id == higher_priority.id


def test_best_persona_places_null_priority_last_and_breaks_ties_by_lower_id() -> None:
    null_priority = _persona(1, None)
    later_equal_priority = _persona(12, 20)
    earlier_equal_priority = _persona(11, 20)
    db_session = MagicMock()
    db_session.scalars.side_effect = lambda statement: _AccessiblePersonaResult(
        statement,
        [null_priority, later_equal_priority, earlier_equal_priority],
    )
    user = SimpleNamespace(
        id=uuid4(),
        is_anonymous=False,
        role=UserRole.BASIC,
    )

    selected_id = get_best_persona_id_for_user(db_session, cast(User, user))

    assert selected_id == earlier_equal_priority.id


def test_best_persona_for_document_set_excludes_other_scoped_personas() -> None:
    captured_statements: list[object] = []
    db_session = MagicMock()

    def capture_statement(statement: object) -> _AccessiblePersonaResult:
        captured_statements.append(statement)
        return _AccessiblePersonaResult(statement, [])

    db_session.scalars.side_effect = capture_statement
    user = SimpleNamespace(
        id=uuid4(),
        is_anonymous=False,
        role=UserRole.BASIC,
    )

    get_best_persona_id_for_user(
        db_session,
        cast(User, user),
        document_set_id=23,
    )

    assert len(captured_statements) == 2
    scoped_sql, unscoped_sql = map(str, captured_statements)
    assert "persona__document_set.document_set_id" in scoped_sql
    assert "persona__document_set.document_set_id = :document_set_id_1" in scoped_sql
    assert "NOT (EXISTS" in unscoped_sql
    for sql in (scoped_sql, unscoped_sql):
        assert "persona.display_priority DESC NULLS LAST" in sql
        assert "persona.id ASC" in sql


def test_exact_document_set_persona_precedes_higher_priority_unscoped_fallback() -> (
    None
):
    exact_persona = _persona(11, 10)
    unscoped_persona = _persona(12, 100)
    db_session = MagicMock()

    def select_by_scope(statement: object) -> _AccessiblePersonaResult:
        if "persona__document_set.document_set_id" in str(statement):
            return _AccessiblePersonaResult(statement, [exact_persona])
        return _AccessiblePersonaResult(statement, [unscoped_persona])

    db_session.scalars.side_effect = select_by_scope
    user = SimpleNamespace(
        id=uuid4(),
        is_anonymous=False,
        role=UserRole.BASIC,
    )

    selected_id = get_best_persona_id_for_user(
        db_session,
        cast(User, user),
        document_set_id=23,
    )

    assert selected_id == exact_persona.id
    assert db_session.scalars.call_count == 1


def test_update_display_priority_updates_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Precondition
    persona_a = _persona(1, 5)
    persona_b = _persona(2, 6)
    db_session = MagicMock()
    user = MagicMock()
    monkeypatch.setattr(
        "onyx.db.persona.get_raw_personas_for_user",
        lambda user, db_session, **kwargs: [persona_a, persona_b],  # noqa: ARG005
    )

    # Under test
    update_personas_display_priority(
        {persona_a.id: 0}, db_session, user, commit_db_txn=True
    )

    # Postcondition
    assert persona_a.display_priority == 0
    assert persona_b.display_priority == 6
    db_session.commit.assert_called_once_with()


def test_update_display_priority_invalid_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    # Precondition
    persona_a = _persona(1, 5)
    db_session = MagicMock()
    user = MagicMock()
    monkeypatch.setattr(
        "onyx.db.persona.get_raw_personas_for_user",
        lambda user, db_session, **kwargs: [persona_a],  # noqa: ARG005
    )

    # Under test
    with pytest.raises(ValueError):
        update_personas_display_priority(
            {persona_a.id: 0, 99: 1},
            db_session,
            user,
            commit_db_txn=True,
        )

    # Postcondition
    db_session.commit.assert_not_called()
