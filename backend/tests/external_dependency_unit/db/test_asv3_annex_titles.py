"""Legacy annex filenames remain discoverable by their qualified natural titles."""

import pytest
from sqlalchemy import Text, column, select, values
from sqlalchemy.orm import Session
from sqlalchemy.sql import visitors
from sqlalchemy.sql.elements import ColumnElement

from onyx.db.asv3_corpus import _source_title_query
from onyx.db.models import UserFile


@pytest.mark.parametrize("number", [4, 6])
@pytest.mark.parametrize("spelling", ["Gümrük Yöneteliği", "Gümrük Yönetmeliği"])
@pytest.mark.parametrize("label", ["ek {number}", "EK-{number}", "Ek_{number:02d}"])
def test_qualified_customs_annex_title_matches_only_its_own_number(
    db_session: Session, number: int, spelling: str, label: str
) -> None:
    expected = f"{spelling} Ekleri/ek_{number:02d}.md"
    names = values(column("name", Text), name="annex_names").data(
        [
            (name,)
            for name in (
                expected,
                f"{spelling} Ekleri/ek_14_ek_{number:02d}.md",
                f"{spelling} Ekleri/ek_{number}0.md",
                f"{spelling} Ekleri/ek_{number + 1:02d}.md",
                f"Başka Yönetmelik Ekleri/ek_{number:02d}.md",
                f"Başka Belgeler/ek_{number:02d}.md",
                f"{spelling} Ekleri/ek_{number:02d}_eski.md",
            )
        ]
    )
    conditions, _ = _source_title_query(
        "Gümrük Yönetmeliği " + label.format(number=number)
    )

    def replace_name(element: ColumnElement[object]) -> ColumnElement[object] | None:
        return names.c.name if element.compare(UserFile.name.expression) else None

    conditions = [
        visitors.replacement_traverse(condition, {}, replace_name)
        for condition in conditions
    ]
    assert list(db_session.scalars(select(names.c.name).where(*conditions))) == [
        expected
    ]
