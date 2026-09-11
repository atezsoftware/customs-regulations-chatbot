"""Private logical configuration fingerprints for regulatory preparation guards."""

from sqlalchemy import inspect

from onyx.db.models import SearchSettings, UserFile
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.utils.sensitive import SensitiveValue


def configuration_fingerprint(
    value: UserFile | SearchSettings | list[object],
) -> str:
    """Hash complete mapped inputs; logical credentials never leave this boundary."""

    def normalize(item: object) -> object:
        if isinstance(item, (UserFile, SearchSettings)):
            columns: dict[str, object] = {}
            for column in inspect(type(item)).columns:
                if column.key in ("created_at", "updated_at", "last_accessed_at"):
                    continue
                field = getattr(item, column.key)
                columns[column.key] = (
                    field.get_value(apply_mask=False)
                    if isinstance(field, SensitiveValue)
                    else field
                )
            return columns
        if isinstance(item, list):
            return [normalize(child) for child in item]
        return item

    return context_hash(normalize(value))
