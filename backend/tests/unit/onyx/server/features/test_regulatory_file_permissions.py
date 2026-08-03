import inspect
from collections.abc import Callable

import pytest

from onyx.db.enums import Permission
from onyx.server.features.projects.api import (
    delete_user_file,
    get_files_in_project,
    get_user_file,
    upload_user_files,
)
from onyx.server.features.regulatory.api import (
    list_chunks_for_file,
    patch_chunk,
    rename_user_file,
)


@pytest.mark.parametrize(
    "endpoint",
    [
        upload_user_files,
        get_files_in_project,
        get_user_file,
        delete_user_file,
        list_chunks_for_file,
        patch_chunk,
        rename_user_file,
    ],
)
def test_regulatory_file_endpoints_require_admin_permission(
    endpoint: Callable[..., object],
) -> None:
    user_parameter = inspect.signature(endpoint).parameters["user"]
    dependency = user_parameter.default.dependency

    assert getattr(dependency, "_is_require_permission", False)
    assert (
        getattr(dependency, "_required_permission", None)
        is Permission.FULL_ADMIN_PANEL_ACCESS
    )
