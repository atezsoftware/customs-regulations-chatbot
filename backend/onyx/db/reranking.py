from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, GetPydanticSchema
from pydantic_core import CoreSchema, core_schema
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.enums import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.utils.sensitive import SensitiveValue
from shared_configs.enums import RerankerProvider


def _sensitive_string_schema(
    source_type: Any,  # noqa: ARG001
    handler: GetCoreSchemaHandler,  # noqa: ARG001
) -> CoreSchema:
    return core_schema.is_instance_schema(SensitiveValue)


SensitiveString = Annotated[
    SensitiveValue[str], GetPydanticSchema(_sensitive_string_schema)
]


class RerankerRuntimeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool
    provider_type: RerankerProvider | None
    model_name: str | None
    api_key: SensitiveString | None
    configuration_generation: str


def _runtime_config(settings: SearchSettings) -> RerankerRuntimeConfig:
    return RerankerRuntimeConfig(
        enabled=settings.rerank_enabled,
        provider_type=settings.rerank_provider_type,
        model_name=settings.rerank_model_name,
        api_key=settings.rerank_api_key,
        configuration_generation=settings.rerank_configuration_generation,
    )


def _get_present_settings(db_session: Session, *, for_update: bool) -> SearchSettings:
    statement = (
        select(SearchSettings)
        .where(SearchSettings.status == IndexModelStatus.PRESENT)
        .order_by(SearchSettings.id.desc())
    )
    if for_update:
        statement = statement.with_for_update()
    settings = db_session.scalar(statement)
    if settings is None:
        raise RuntimeError("No PRESENT search settings row exists.")
    return settings


def get_reranker_configuration(db_session: Session) -> RerankerRuntimeConfig:
    return _runtime_config(_get_present_settings(db_session, for_update=False))


def get_reranker_configuration_for_update(
    db_session: Session,
) -> RerankerRuntimeConfig:
    """Return the live configuration while holding its row lock."""
    return _runtime_config(_get_present_settings(db_session, for_update=True))


def upsert_reranker_configuration(
    db_session: Session,
    *,
    enabled: bool,
    provider_type: RerankerProvider | None,
    model_name: str | None,
    api_key: str | None,
    updated_by_user_id: UUID | None = None,
    commit: bool = True,
) -> RerankerRuntimeConfig:
    settings = _get_present_settings(db_session, for_update=True)

    if enabled:
        if provider_type is None or not model_name:
            raise ValueError(
                "Enabled reranking requires a provider type and model name."
            )
        if api_key is None and settings.rerank_api_key is None:
            raise ValueError("Enabled reranking requires an API key.")
    settings.rerank_enabled = enabled
    settings.rerank_provider_type = provider_type
    settings.rerank_model_name = model_name
    if api_key is not None:
        settings.rerank_api_key = api_key  # ty: ignore[invalid-assignment]

    settings.rerank_updated_at = datetime.now(timezone.utc)
    settings.rerank_configuration_generation = uuid4().hex
    settings.rerank_updated_by_user_id = updated_by_user_id
    if commit:
        db_session.commit()
        db_session.refresh(settings)
    else:
        db_session.flush()
    return _runtime_config(settings)


def delete_reranker_configuration(
    db_session: Session,
    *,
    updated_by_user_id: UUID | None = None,
    commit: bool = True,
) -> None:
    settings_rows = list(
        db_session.scalars(select(SearchSettings).with_for_update()).all()
    )
    updated_at = datetime.now(timezone.utc)
    for settings in settings_rows:
        settings.rerank_enabled = False
        settings.rerank_provider_type = None
        settings.rerank_model_name = None
        settings.rerank_api_key = None
        settings.rerank_updated_at = updated_at
        settings.rerank_configuration_generation = uuid4().hex
        if settings.status == IndexModelStatus.PRESENT:
            settings.rerank_updated_by_user_id = updated_by_user_id
    if commit:
        db_session.commit()
    else:
        db_session.flush()


def transfer_reranker_configuration__no_commit(
    db_session: Session,
    *,
    source_search_settings_id: int,
    target_search_settings_id: int,
) -> None:
    rows = list(
        db_session.scalars(
            select(SearchSettings)
            .where(
                SearchSettings.id.in_(
                    [source_search_settings_id, target_search_settings_id]
                )
            )
            .order_by(SearchSettings.id)
            .with_for_update()
        ).all()
    )
    rows_by_id = {row.id: row for row in rows}
    try:
        source = rows_by_id[source_search_settings_id]
        target = rows_by_id[target_search_settings_id]
    except KeyError as error:
        raise RuntimeError(
            "Cannot transfer reranker configuration: row not found."
        ) from error

    target.rerank_enabled = source.rerank_enabled
    target.rerank_provider_type = source.rerank_provider_type
    target.rerank_model_name = source.rerank_model_name
    target.rerank_api_key = source.rerank_api_key
    target.rerank_updated_at = source.rerank_updated_at
    target.rerank_updated_by_user_id = source.rerank_updated_by_user_id
