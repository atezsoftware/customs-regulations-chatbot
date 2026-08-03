from typing import Any, cast

import pytest

from onyx.server.documents import cc_pair, connector, targeted_reindex
from onyx.server.features.projects import api as projects_api
from onyx.server.manage import search_settings
from onyx.server.onyx_api import ingestion


class ImportBoundaryReached(RuntimeError):
    pass


def _reject_import() -> None:
    raise ImportBoundaryReached


def test_project_upload_checks_document_import_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        projects_api, "ensure_document_import_available", _reject_import
    )

    with pytest.raises(ImportBoundaryReached):
        projects_api.upload_user_files(
            bg_tasks=cast(Any, None),
            files=[],
            project_id=None,
            temp_id_map=None,
            user=cast(Any, None),
            db_session=cast(Any, None),
        )


def test_connector_upload_checks_document_import_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connector, "ensure_document_import_available", _reject_import)

    with pytest.raises(ImportBoundaryReached):
        connector.upload_files_api(files=[], unzip=True, _=cast(Any, None))


def test_ingestion_upsert_checks_document_import_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingestion, "ensure_document_import_available", _reject_import)

    with pytest.raises(ImportBoundaryReached):
        ingestion.upsert_ingestion_doc(
            doc_info=cast(Any, None),
            _=cast(Any, None),
            db_session=cast(Any, None),
        )


@pytest.mark.parametrize(
    ("module", "call"),
    [
        (
            connector,
            lambda: connector.create_connector_from_model(
                connector_data=cast(Any, None),
                user=cast(Any, None),
                db_session=cast(Any, None),
            ),
        ),
        (
            cc_pair,
            lambda: cc_pair.prune_cc_pair(
                cc_pair_id=1,
                user=cast(Any, None),
                db_session=cast(Any, None),
            ),
        ),
        (
            targeted_reindex,
            lambda: targeted_reindex.submit_targeted_reindex(
                request=cast(Any, None),
                user=cast(Any, None),
                db_session=cast(Any, None),
            ),
        ),
        (
            search_settings,
            lambda: search_settings.set_new_search_settings(
                search_settings_new=cast(Any, None),
                db_session=cast(Any, None),
            ),
        ),
    ],
)
def test_indexing_mutations_check_document_import_capability(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    call: Any,
) -> None:
    monkeypatch.setattr(module, "ensure_document_import_available", _reject_import)

    with pytest.raises(ImportBoundaryReached):
        call()
