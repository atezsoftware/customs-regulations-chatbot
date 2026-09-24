from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.db.models import SearchSettings
from onyx.db.regulatory_public_reads import resolve_public_query_index


def test_query_authority_requires_nonempty_file_scope() -> None:
    with pytest.raises(ValueError, match="file scope"):
        resolve_public_query_index("index", "physical", file_ids=())


def test_query_authority_does_not_read_unrelated_projection_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.engine import sql_engine
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.publication_baseline import observed_index_snapshot
    from shared_configs.configs import MULTI_TENANT

    file_id = uuid4()
    setting = MagicMock(
        spec=SearchSettings,
        id=11,
        index_name="index",
        provider_type="openai",
        model_name="text-embedding-3-small",
        final_embedding_dim=1024,
        api_url=None,
        deployment_name=None,
        api_version=None,
        normalize=True,
        passage_prefix=None,
        query_prefix=None,
    )
    observed = observed_index_snapshot(setting, "physical")
    session = MagicMock()
    session.scalars.side_effect = [
        MagicMock(one_or_none=lambda: setting),
        [observed.model_dump(mode="json")],
    ]
    context = MagicMock()
    context.__enter__.return_value = session
    monkeypatch.setattr(sql_engine, "get_session_with_current_tenant", lambda: context)
    result = resolve_public_query_index("index", "physical", file_ids=(file_id,))
    statement = session.scalars.call_args_list[1].args[0]
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "regulatory_temporal_projection.user_file_id IN" in sql
    assert str(file_id).replace("-", "") in sql.replace("-", "")
    assert result == observed
    assert result.multitenant == MULTI_TENANT
    assert result.embedding_config_sha256 != context_hash("invented authority")
