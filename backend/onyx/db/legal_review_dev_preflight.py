"""Read-only DEV identity and configured legal-review model inspection."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast
from uuid import UUID

from pydantic import JsonValue

_DEV_DATABASE = "customs-regulations-dev"
_REDIS_DATABASES = (4, 5, 6)
_REDIS_OWNER_KEY = "onyx:deployment-owner"
_MODEL = "gemini-3.8-flash"
_PROVIDER_TYPE = "vertex_ai"


class PreflightRefusal(RuntimeError):
    """Only fixed operational codes may be printed."""


@contextmanager
def _inspection_engine() -> Iterator[None]:
    from onyx.db.engine.sql_engine import SqlEngine

    try:
        # The model resolver opens a second session while inspection retains one.
        with SqlEngine.scoped_engine(
            pool_size=2,
            max_overflow=0,
            pool_timeout=5,
            connect_args={"connect_timeout": 5, "options": os.environ["PGOPTIONS"]},
        ):
            yield
    except PreflightRefusal:
        raise
    except Exception as error:
        raise PreflightRefusal(
            f"database_inspection_failed_{type(error).__name__}"
        ) from None


def inspect_dev() -> dict[str, JsonValue]:
    if (
        os.environ.get("POSTGRES_DB") != _DEV_DATABASE
        or os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev"
    ):
        raise PreflightRefusal("explicit_dev_environment_required")
    # Every session opened by the normal model resolver inherits the same fence.
    os.environ["PGOPTIONS"] = (
        "-c default_transaction_read_only=on -c statement_timeout=15000 "
        "-c application_name=legal_review_dev_preflight"
    )

    from redis import Redis
    from sqlalchemy import select, text
    from sqlalchemy.sql.elements import ColumnElement

    from onyx.auth.schemas import UserRole
    from onyx.configs import app_configs
    from onyx.db.engine.sql_engine import get_session_with_current_tenant
    from onyx.db.legal_review_providers import resolve_legal_review_decision
    from onyx.db.llm import fetch_llm_provider_for_model_selection
    from onyx.db.models import LLMProvider, User
    from onyx.db.persona import get_default_behavior_persona
    from onyx.llm.factory import get_llm_for_persona
    from onyx.llm.override_models import LLMOverride
    from onyx.redis.redis_pool import RedisPool
    from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

    # Standalone inspection must use the API's configured credential codec.
    set_is_ee_based_on_env_variable()

    if (
        app_configs.POSTGRES_DB != _DEV_DATABASE
        or app_configs.REGULATORY_ANNEX_ENVIRONMENT != "dev"
        or (
            app_configs.REDIS_DB_NUMBER,
            app_configs.REDIS_DB_NUMBER_CELERY,
            app_configs.REDIS_DB_NUMBER_CELERY_RESULT_BACKEND,
        )
        != _REDIS_DATABASES
    ):
        raise PreflightRefusal("dev_runtime_isolation_mismatch")
    owner = (
        _DEV_DATABASE
        + ":"
        + hashlib.sha256(
            f"{app_configs.POSTGRES_HOST}:{app_configs.POSTGRES_PORT}/{_DEV_DATABASE}".encode()
        ).hexdigest()
    )
    for database in _REDIS_DATABASES:
        pool = RedisPool.create_pool(
            db=database, ssl=app_configs.REDIS_SSL, max_connections=1
        )
        pool.connection_kwargs.update(socket_timeout=3, socket_connect_timeout=3)
        try:
            with Redis(connection_pool=pool) as client:
                if client.get(_REDIS_OWNER_KEY) != owner.encode():
                    raise PreflightRefusal("dev_redis_owner_marker_mismatch")
        finally:
            pool.disconnect()

    report: dict[str, JsonValue] = {
        "operation": "inspect",
        "environment": "dev",
        "redis_isolation_verified": True,
        "typesafe_credential_configured": bool(
            (app_configs.TYPESAFE_API_KEY or "").strip()
        ),
    }
    with _inspection_engine(), get_session_with_current_tenant() as session:
        if session.scalar(text("SHOW default_transaction_read_only")) != "on":
            raise PreflightRefusal("database_read_only_fence_missing")
        if session.scalar(text("SELECT current_database()")) != _DEV_DATABASE:
            raise PreflightRefusal("connected_database_is_not_dev")
        report["database_revision"] = [
            str(revision)
            for revision in session.scalars(
                text("SELECT version_num FROM alembic_version ORDER BY version_num")
            ).all()
        ]
        providers: list[JsonValue] = []
        for provider in session.scalars(select(LLMProvider).order_by(LLMProvider.id)):
            providers.append(
                {
                    "name": provider.name,
                    "provider_type": provider.provider,
                    "model_names": [
                        model.name
                        for model in provider.model_configurations
                        if model.is_visible
                    ],
                    "api_key_configured": provider.api_key is not None,
                    "custom_config_configured": bool(provider.custom_config),
                }
            )
        report["configured_providers"] = providers
        admin = (
            session.scalars(
                select(User)
                .where(
                    User.role == UserRole.ADMIN,
                    cast(ColumnElement[bool], User.is_active).is_(True),
                )
                .order_by(cast(ColumnElement[UUID], User.id))
                .limit(1)
            )
            .unique()
            .first()
        )
        persona = get_default_behavior_persona(session)
        report["active_admin_available"] = admin is not None
        report["default_persona_available"] = persona is not None
        report["decision_resolution"] = {"available": False}
        if admin is None or persona is None:
            report["gemini_resolution"] = {
                "available": False,
                "failure_code": "existing_admin_or_default_persona_missing",
            }
        else:
            decision = resolve_legal_review_decision(
                admin, persona=persona, db_session=session
            )
            report["decision_resolution"] = {
                "available": decision is not None,
                "route": decision.route if decision else None,
                "provider_name": decision.provider_name if decision else None,
                "model_name": "gpt-6-luna" if decision else None,
            }
            provider = fetch_llm_provider_for_model_selection(
                None, _PROVIDER_TYPE, _MODEL, session
            )
            try:
                llm = get_llm_for_persona(
                    persona,
                    admin,
                    llm_override=LLMOverride(
                        model_provider_type=_PROVIDER_TYPE,
                        model_version=_MODEL,
                        temperature=0,
                    ),
                )
            except Exception as error:
                # Exception messages may contain provider configuration or credentials.
                report["gemini_resolution"] = {
                    "available": False,
                    "failure_code": type(error).__name__,
                }
            else:
                report["gemini_resolution"] = {
                    "available": (
                        llm.config.model_provider == _PROVIDER_TYPE
                        and llm.config.model_name == _MODEL
                        and provider is not None
                    ),
                    "provider_name": provider.name if provider else None,
                    "provider_type": llm.config.model_provider,
                    "model_name": llm.config.model_name,
                }
    resolution = report["gemini_resolution"]
    decision_resolution = report["decision_resolution"]
    report["ready"] = (
        isinstance(resolution, dict)
        and resolution.get("available") is True
        and isinstance(decision_resolution, dict)
        and decision_resolution.get("available") is True
    )
    return report


def main() -> None:
    logging.disable(logging.CRITICAL)

    def expired(_signum: int, _frame: object) -> None:
        raise PreflightRefusal("preflight_timeout")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    try:
        report = inspect_dev()
        print(json.dumps(report, sort_keys=True), flush=True)
        if not report["ready"]:
            raise SystemExit(1)
    except Exception as error:
        print(
            json.dumps(
                {
                    "operation": "inspect",
                    "ready": False,
                    "failure_code": str(error)
                    if isinstance(error, PreflightRefusal)
                    else type(error).__name__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(1) from None
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
