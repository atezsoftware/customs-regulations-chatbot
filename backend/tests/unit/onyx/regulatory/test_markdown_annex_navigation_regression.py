import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from elasticsearch import Elasticsearch

from onyx.db.enums import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.document_index.publication_models import (
    FileOwnership,
    FrozenPublicationProjection,
)
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory import writer_projection
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.chunking import IntakeRegulatoryChunker
from onyx.regulatory.document_intake.text import prepare_intake_text
from onyx.regulatory.indexing import RegulatoryIndexingChunker
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from shared_configs.enums import EmbeddingProvider
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)
from tests.unit.onyx.regulatory.whole_table_encoder_helpers import _model

FIXTURES = Path(__file__).parent / "fixtures" / "md_annex_navigation"


def _actual_markdown_writer_manifest(
    filename: str,
    *,
    index_name: str = "markdown-regression",
    client: Elasticsearch | MagicMock | None = None,
) -> tuple[FileOwnership, WriterPublicationManifest, list[list[str]]]:
    main = (FIXTURES / "main.md").read_bytes()
    annex = (FIXTURES / "20220618-8-1 (1).md").read_bytes()
    link = SourceLink(
        parent_asset_hash=sha256(main).hexdigest(),
        target_asset_hash=sha256(annex).hexdigest(),
        source_field="markdown-draft:root:annex",
        label="20220618-8-1 (1).md",
        kind="embedded",
    )
    raw = (FIXTURES / filename).read_bytes()
    text = prepare_intake_text(
        raw.decode("utf-8-sig"),
        asset_sha256=sha256(raw).hexdigest(),
        links=[link],
        preserve_markdown_headings=True,
    )
    model = _model()
    file_id = uuid4()
    indexing_chunker = RegulatoryIndexingChunker(
        db_session=MagicMock(), tokenizer=model.tokenizer, enable_contextual_rag=True
    )
    indexing_chunker.chunk_token_limit = 512
    chunked, _ = indexing_chunker._chunk_with_embedding_budget(
        text,
        source_path=str(file_id),
        source_file=filename,
        intake_version="3",
        retain_markdown_headings=True,
    )
    rows = []
    for position, chunk in enumerate(
        c for c in chunked.chunks if c.metadata.chunk_variant == "atomic"
    ):
        row = canonical_row(file_id, position, chunk.text)
        row.chunk_type = chunk.metadata.chunk_type
        row.heading_path = chunk.metadata.heading_path
        row.chunk_metadata = {
            **chunk.metadata.to_storage_dict(),
            "intake_chunking_version": "3",
        }
        rows.append(row)
    authority = OwnedAuthority(file_id, tenant_id="public")
    settings = SearchSettings(
        id=1,
        status=IndexModelStatus.PRESENT,
        index_name=index_name,
        model_name="gemini-embedding-2",
        provider_type=EmbeddingProvider.GOOGLE,
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    inputs = replace(writer_inputs(file_id, rows), settings=[settings])
    if client is None:
        client = MagicMock()
        client.indices.get.return_value = {
            index_name: {"settings": {"index": {"uuid": "physical-md-regression"}}}
        }
    requests = []

    def encode(*, texts: list[str], **_kwargs: object) -> list[list[float]]:
        requests.append(list(texts))
        assert all(len(model.tokenizer.encode(value)) <= 512 for value in texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    with (
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=False,
        ),
        patch.object(
            writer_projection, "PublicationStore", side_effect=authority.for_scope
        ),
        patch.object(
            writer_projection.DefaultIndexingEmbedder,
            "from_db_search_settings",
            return_value=MagicMock(embedding_model=model),
        ),
        patch.object(
            writer_projection, "resolve_review_context_llm", return_value=None
        ),
        patch.object(model, "encode", side_effect=encode),
    ):
        manifest = writer_projection.prepare_owned_correction(
            authority.owner, client, inputs, inputs.canonical, changed_id=None
        )
    frozen = [
        binding.projection
        for binding in manifest.bindings
        if isinstance(binding.projection, FrozenPublicationProjection)
    ]
    assert frozen and requests
    sent = [text for request in requests for text in request]
    for row in rows:
        assert any(row.text in value for value in sent), row.id
    for projection in frozen:
        assert json.loads(projection.source_json)["content_vector"] == [1.0, 0.0, 0.0]
    return authority.owner, manifest, requests


@pytest.mark.parametrize("filename", ["main.md", "20220618-8-1 (1).md"])
def test_actual_markdown_canonical_chunks_survive_writer_and_embedding(
    filename: str,
) -> None:
    _, manifest, requests = _actual_markdown_writer_manifest(filename)
    sent = "\n".join(text for request in requests for text in request)
    assert manifest.bindings
    if filename == "main.md":
        assert "kapsamaz." in sent and "MADDE 45" in sent
        assert "https://www.resmigazete.gov.tr/eskiler/2022/06/20220618-8-1.pdf" in sent
    else:
        assert "TEHLİKELİ MADDE FAALİYET BELGESİ BAŞVURU FORMU" in sent


@pytest.mark.parametrize("remove_external_link", [False, True])
@pytest.mark.parametrize("chunking_version", ["3", "5"])
def test_actual_markdown_main_and_numeric_annex_preserve_all_headings(
    remove_external_link: bool,
    chunking_version: str,
) -> None:
    main = (FIXTURES / "main.md").read_bytes()
    annex = (FIXTURES / "20220618-8-1 (1).md").read_bytes()
    link = SourceLink(
        parent_asset_hash=sha256(main).hexdigest(),
        target_asset_hash=sha256(annex).hexdigest(),
        source_field="markdown-draft:00000000-0000-0000-0000-000000000001:00000000-0000-0000-0000-000000000002",
        label="20220618-8-1 (1).md",
        kind="embedded",
    )
    frozen_link = link.model_dump()
    for original in (main, annex):
        source = original.decode("utf-8-sig")
        if remove_external_link:
            source = "\n".join(
                line
                for line in source.splitlines()
                if "Ekleri için tıklayınız" not in line
            )
        normalized = prepare_intake_text(
            source,
            asset_sha256=sha256(original).hexdigest(),
            links=[link],
            preserve_markdown_headings=True,
        )
        assert normalized.replace(" ", "").replace("\n", "").replace(
            "\r", ""
        ) == source.replace(" ", "").replace("\n", "").replace("\r", "")
        result = IntakeRegulatoryChunker(
            tokenizer=TiktokenTokenizer("text-embedding-3-large"),
            content_token_limit=505,
            retain_markdown_headings=True,
            chunking_version=chunking_version,
            table_content_token_limit=1848 if chunking_version == "5" else None,
        ).chunk_text(normalized)
        delivered = "\n".join(
            chunk.text + "\n" + "\n".join(chunk.metadata.heading_path)
            for chunk in result.chunks
        )
        for heading in (
            line.lstrip("# ").strip()
            for line in source.splitlines()
            if line.startswith("#")
        ):
            assert heading in delivered or any(
                "".join(
                    character for character in heading if character.isalnum()
                ).casefold()
                == "".join(
                    character for character in label if character.isalnum()
                ).casefold()
                for chunk in result.chunks
                for label in chunk.metadata.heading_path
            )
        if original is main:
            assert "kapsamaz." in delivered
            assert "MADDE 45" in delivered
        assert link.model_dump() == frozen_link
