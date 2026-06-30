"""
CLI entry point for the FsExplorer indexer.

Builds and manages the Postgres+pgvector index (Docling parsing, regulatory
chunking, optional langextract metadata extraction). This is the only CLI
that touches Docling/langextract — querying an existing index from the
command line is the `fs-explorer-api` service's `explore` CLI, not this one.
"""

import json
import os
from pathlib import Path

from typer import Typer, Option, Argument, BadParameter, Exit
from typing import Annotated, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fs_explorer_shared.embeddings import EmbeddingProvider
from fs_explorer_shared.index_config import resolve_database_url
from fs_explorer_shared.storage import PostgresStorage
from .indexing import IndexingPipeline, SchemaDiscovery

app = Typer()
schema_app = Typer(help="Manage metadata schemas for indexed corpora.")
app.add_typer(schema_app, name="schema")


def _load_metadata_profile(path_value: str | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    resolved = Path(path_value).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise BadParameter(f"Metadata profile file not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise BadParameter(
            f"Metadata profile file is not valid JSON: {resolved}"
        ) from exc
    if not isinstance(payload, dict):
        raise BadParameter("Metadata profile JSON must be an object.")
    return payload


@app.command("index")
def index_command(
    folder: Annotated[
        str,
        Argument(help="Folder to index recursively."),
    ] = ".",
    database_url: Annotated[
        str | None,
        Option(
            "--database-url", help="Postgres connection string (or set DATABASE_URL)."
        ),
    ] = None,
    discover_schema: Annotated[
        bool,
        Option(
            "--discover-schema",
            help="Auto-discover metadata schema and set it active for this corpus.",
        ),
    ] = False,
    schema_name: Annotated[
        str | None,
        Option("--schema-name", help="Use an existing stored schema by name."),
    ] = None,
    with_metadata: Annotated[
        bool,
        Option(
            "--with-metadata",
            help=(
                "Enable langextract metadata extraction (requires API key). "
                "Also enables schema discovery if not explicitly requested."
            ),
        ),
    ] = False,
    metadata_profile_path: Annotated[
        str | None,
        Option(
            "--metadata-profile",
            help=(
                "Path to JSON profile defining dynamic langextract metadata fields "
                "and prompt. Implies --with-metadata."
            ),
        ),
    ] = None,
    with_embeddings: Annotated[
        bool,
        Option(
            "--with-embeddings",
            help="Generate vector embeddings for indexed chunks (requires GOOGLE_API_KEY).",
        ),
    ] = False,
) -> None:
    """Build or refresh an index for a folder."""
    console = Console()
    resolved_database_url = resolve_database_url(database_url)
    storage = PostgresStorage(resolved_database_url)

    embedding_provider: EmbeddingProvider | None = None
    if with_embeddings:
        try:
            embedding_provider = EmbeddingProvider()
        except ValueError as exc:
            raise BadParameter(str(exc)) from exc

    pipeline = IndexingPipeline(
        storage=storage,
        embedding_provider=embedding_provider,
    )
    metadata_profile = _load_metadata_profile(metadata_profile_path)
    effective_with_metadata = with_metadata or metadata_profile is not None

    if effective_with_metadata and metadata_profile is None:
        console.print(
            "[bold cyan]🔍 Analyzing corpus to generate metadata profile...[/]"
        )

    try:
        effective_discover_schema = discover_schema or effective_with_metadata
        result = pipeline.index_folder(
            folder,
            discover_schema=effective_discover_schema,
            schema_name=schema_name,
            with_metadata=effective_with_metadata,
            metadata_profile=metadata_profile,
        )
    except ValueError as exc:
        raise BadParameter(str(exc)) from exc

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold", justify="right")
    summary.add_column()
    summary.add_row("Database URL:", resolved_database_url)
    summary.add_row("Corpus ID:", result.corpus_id)
    summary.add_row("Indexed Files:", str(result.indexed_files))
    summary.add_row("Skipped Files:", str(result.skipped_files))
    summary.add_row("Deleted Files:", str(result.deleted_files))
    summary.add_row("Chunks Written:", str(result.chunks_written))
    summary.add_row("Active Documents:", str(result.active_documents))
    summary.add_row("Embeddings Written:", str(result.embeddings_written))
    summary.add_row("Schema Used:", result.schema_used or "<none>")
    summary.add_row(
        "Metadata Mode:",
        "langextract" if effective_with_metadata else "heuristic",
    )
    if metadata_profile_path:
        profile_label = str(Path(metadata_profile_path).expanduser().resolve())
    elif effective_with_metadata:
        profile_label = "<auto-discovered>"
    else:
        profile_label = "<none>"
    summary.add_row("Metadata Profile:", profile_label)

    console.print(Panel(summary, title="📦 Index Complete", border_style="bold green"))


@schema_app.command("discover")
def schema_discover_command(
    folder: Annotated[
        str,
        Argument(help="Folder to inspect for schema discovery."),
    ] = ".",
    database_url: Annotated[
        str | None,
        Option(
            "--database-url", help="Postgres connection string (or set DATABASE_URL)."
        ),
    ] = None,
    name: Annotated[
        str | None,
        Option("--name", help="Override discovered schema name."),
    ] = None,
    activate: Annotated[
        bool,
        Option(
            "--activate/--no-activate",
            help="Set schema as active for the corpus.",
        ),
    ] = True,
    with_metadata: Annotated[
        bool,
        Option(
            "--with-metadata",
            help="Include langextract metadata fields in discovered schema.",
        ),
    ] = False,
    metadata_profile_path: Annotated[
        str | None,
        Option(
            "--metadata-profile",
            help=(
                "Path to JSON profile defining dynamic langextract metadata fields "
                "and prompt. Implies --with-metadata."
            ),
        ),
    ] = None,
) -> None:
    """Auto-discover and store a metadata schema for a folder."""
    console = Console()
    resolved_folder = str(os.path.abspath(folder))
    if not os.path.isdir(resolved_folder):
        raise BadParameter(f"No such directory: {resolved_folder}")

    resolved_database_url = resolve_database_url(database_url)
    storage = PostgresStorage(resolved_database_url)
    corpus_id = storage.get_or_create_corpus(resolved_folder)
    metadata_profile = _load_metadata_profile(metadata_profile_path)
    effective_with_metadata = with_metadata or metadata_profile is not None

    if effective_with_metadata and metadata_profile is None:
        console.print(
            "[bold cyan]🔍 Analyzing corpus to generate metadata profile...[/]"
        )

    discovery = SchemaDiscovery()
    discovered = discovery.discover_from_folder(
        resolved_folder,
        with_langextract=effective_with_metadata,
        metadata_profile=metadata_profile,
    )
    schema_name = name or str(
        discovered.get("name", f"auto_{os.path.basename(resolved_folder)}")
    )
    discovered["name"] = schema_name
    schema_id = storage.save_schema(
        corpus_id=corpus_id,
        name=schema_name,
        schema_def=discovered,
        is_active=activate,
    )

    output = Table.grid(padding=(0, 2))
    output.add_column(style="bold", justify="right")
    output.add_column()
    output.add_row("Database URL:", resolved_database_url)
    output.add_row("Corpus ID:", corpus_id)
    output.add_row("Schema ID:", schema_id)
    output.add_row("Schema Name:", schema_name)
    output.add_row("Active:", str(activate))
    output.add_row("Field Count:", str(len(discovered.get("fields", []))))
    output.add_row(
        "Metadata Mode:", "langextract" if effective_with_metadata else "heuristic"
    )
    if metadata_profile_path:
        profile_label = str(Path(metadata_profile_path).expanduser().resolve())
    elif effective_with_metadata:
        profile_label = "<auto-discovered>"
    else:
        profile_label = "<none>"
    output.add_row("Metadata Profile:", profile_label)

    console.print(Panel(output, title="🧩 Schema Saved", border_style="bold cyan"))
    console.print_json(json.dumps(discovered, indent=2))


@schema_app.command("show")
def schema_show_command(
    folder: Annotated[
        str,
        Argument(help="Folder whose schemas should be listed."),
    ] = ".",
    database_url: Annotated[
        str | None,
        Option(
            "--database-url", help="Postgres connection string (or set DATABASE_URL)."
        ),
    ] = None,
) -> None:
    """Show saved schemas for a folder's corpus."""
    console = Console()
    resolved_folder = str(os.path.abspath(folder))
    resolved_database_url = resolve_database_url(database_url)
    storage = PostgresStorage(resolved_database_url)

    corpus_id = storage.get_corpus_id(resolved_folder)
    if corpus_id is None:
        console.print(
            Panel(
                f"No corpus found for folder: {resolved_folder}\nRun `explore-index index {resolved_folder}` first.",
                title="⚠️ No Corpus",
                border_style="bold yellow",
            )
        )
        raise Exit(code=1)

    schemas = storage.list_schemas(corpus_id=corpus_id)
    if not schemas:
        console.print(
            Panel(
                f"No schemas saved for corpus: {corpus_id}",
                title="⚠️ No Schemas",
                border_style="bold yellow",
            )
        )
        raise Exit(code=1)

    table = Table(title=f"Schemas for {resolved_folder}")
    table.add_column("Name")
    table.add_column("Active")
    table.add_column("Created At")
    table.add_column("Field Count")

    for schema in schemas:
        table.add_row(
            schema.name,
            "yes" if schema.is_active else "no",
            schema.created_at,
            str(len(schema.schema_def.get("fields", []))),
        )

    console.print(table)
