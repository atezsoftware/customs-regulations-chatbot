"""Import smoke tests for lightweight indexer entry points."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType


def _import_with_forbidden_modules(
    monkeypatch, module_name: str, forbidden: tuple[str, ...]
) -> ModuleType:
    """Import a fresh module while failing if it imports forbidden heavy deps."""
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            del sys.modules[loaded_name]

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN002
        if any(
            name == blocked or name.startswith(f"{blocked}.") for blocked in forbidden
        ):
            raise AssertionError(f"{module_name} imported heavy dependency {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return importlib.import_module(module_name)


def test_indexer_cli_import_does_not_require_heavy_runtime_dependencies(
    monkeypatch,
) -> None:
    module = _import_with_forbidden_modules(
        monkeypatch,
        "fs_explorer_indexer.main",
        ("docling", "google.genai", "psycopg", "rich"),
    )

    assert module.app is not None


def test_indexer_server_import_does_not_require_heavy_runtime_dependencies(
    monkeypatch,
) -> None:
    module = _import_with_forbidden_modules(
        monkeypatch,
        "fs_explorer_indexer.indexer_server",
        ("docling", "google.genai", "psycopg"),
    )

    assert module.app.title == "FsExplorer Indexer"
