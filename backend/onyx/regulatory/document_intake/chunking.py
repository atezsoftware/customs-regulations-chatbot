"""Token-bounded splitting for newly admitted documents, before labels are frozen."""

import hashlib
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.chunker import (
    ChunkedDocument,
    ChunkMetadata,
    DocumentMetadata,
    RegulatoryChunk,
    RegulatoryChunker,
    SourceBlock,
    StructureNode,
    _article_rest_from_original,
    _article_unit,
    _blocks_text,
    _ChunkDraft,
    _classify_block,
    _clean_inline_markdown,
    _infer_document_metadata,
    _is_probable_standalone_heading,
    _make_structure_node,
    _markdown_to_blocks,
    _same_label,
)
from onyx.regulatory.document_intake.chunking_profile import (
    INTAKE_CHUNKING_CODE_VERSION as INTAKE_CHUNKING_CODE_VERSION,
)
from onyx.regulatory.document_intake.chunking_profile import (
    SUPPORTED_INTAKE_CHUNKING_VERSIONS,
)
from onyx.regulatory.document_intake.document_profile import intake_document_profile
from onyx.regulatory.document_intake.headings import promote_plain_article_titles
from onyx.regulatory.heading_path import parse_regulatory_article_heading

_SENTENCE_END = re.compile(r"[.!?;:](?:\s+|$)")
_WORD_END = re.compile(r"\s+")
_LINE_END = re.compile(r"\n|$")
_NUMBERING_ONLY = re.compile(r"(?:\d+(?:\.\d+)*[.)]?|\(\d+(?:\.\d+)*\)\.?)")
_TABLE_HEADING_MAX_CHARS = 120
_CELL_BREAK = re.compile(r"(?<!\\)<br\s*/?>", re.IGNORECASE)
_CELL_NUMBERED_BREAK = re.compile(r"<br\s*/?>\s*\d+(?:\.\d+)+[.)]?\s", re.IGNORECASE)
_TABLE_SEPARATOR_CELL = re.compile(r"\s*:?-{3,}:?\s*")


@dataclass(frozen=True)
class IntakeTableRow:
    """Original grid coordinates shared by exact slices of one source cell."""

    version: int
    table_index: int | None
    row_index: int
    cell_index: int
    row_sha256: str
    row_source_start_char: int
    row_source_end_char: int
    fragment_source_start_char: int
    fragment_source_end_char: int
    fragment_index: int
    fragment_count: int


@dataclass(frozen=True)
class IntakeChunkMetadata(ChunkMetadata):
    intake_table_row: IntakeTableRow | None = None

    def to_storage_dict(self) -> dict[str, Any]:
        metadata = super().to_storage_dict()
        if self.intake_table_row is not None:
            metadata["intake_table_row"] = asdict(self.intake_table_row)
        return metadata


class IntakeRegulatoryChunker(RegulatoryChunker):
    """Keep an article's heading with its first body slice within the real budget."""

    def __init__(
        self,
        *,
        tokenizer: BaseTokenizer,
        content_token_limit: int,
        max_chunk_chars: int = 2400,
        min_chunk_chars: int = 120,
        chunking_version: str = INTAKE_CHUNKING_CODE_VERSION,
    ) -> None:
        if content_token_limit < 1:
            raise ValueError("Intake content token limit must be positive")
        if chunking_version not in SUPPORTED_INTAKE_CHUNKING_VERSIONS:
            raise ValueError("The intake chunking version is not supported")
        super().__init__(
            max_chunk_chars=max_chunk_chars, min_chunk_chars=min_chunk_chars
        )
        self.tokenizer = tokenizer
        self.content_token_limit = content_token_limit
        self.chunking_version = chunking_version
        self._source_text = ""
        self._source_starts: dict[SourceBlock, int] = {}
        self._table_row_fragments: dict[tuple[int, int], IntakeTableRow] = {}

    def chunk_text(
        self,
        text: str,
        *,
        source_path: str = "<memory>",
        source_file: str = "<memory>",
        parser: str = "memory",
    ) -> ChunkedDocument:
        self._source_text = text.strip()
        self._source_starts.clear()
        self._table_row_fragments.clear()
        metadata = _infer_document_metadata(
            path=Path(source_file),
            root_path=None,
            content=self._source_text,
            parser=parser,
            source_path_override=source_path,
        )
        blocks = _markdown_to_blocks(self._source_text)
        if self.chunking_version == "3":
            blocks = promote_plain_article_titles(blocks, source_text=self._source_text)
        metadata, ignored = intake_document_profile(blocks=blocks, metadata=metadata)
        chunks, structure = self.chunk_blocks_with_structure(
            [block for block in blocks if block.block_id not in ignored],
            metadata=metadata,
        )
        return ChunkedDocument(
            metadata=metadata,
            content=self._source_text,
            blocks=blocks,
            chunks=chunks,
            structure=structure,
            warnings=[],
        )

    def _fits(self, blocks: list[SourceBlock]) -> bool:
        return (
            len(self.tokenizer.encode(_blocks_text(blocks))) <= self.content_token_limit
        )

    def _source_start(self, block: SourceBlock) -> int:
        if block in self._source_starts:
            return self._source_starts[block]
        original = self._source_text[block.start_char : block.end_char]
        position = original.find(block.text)
        result = block.start_char + max(position, 0)
        self._source_starts[block] = result
        return result

    def _slice(self, block: SourceBlock, start: int, end: int) -> SourceBlock:
        text = block.text[start:end]
        leading = len(text) - len(text.lstrip())
        trailing = len(text.rstrip())
        source_start = self._source_start(block)
        return SourceBlock(
            block_id=block.block_id,
            kind=block.kind,
            text=text.strip(),
            start_char=source_start + start + leading,
            end_char=source_start + start + trailing,
        )

    def _prefix_end(
        self, block: SourceBlock, start: int, prefix: list[SourceBlock]
    ) -> int:
        """Select a source boundary, measuring the final heading and body together."""
        if block.kind == "table":
            end = start
            for match in _LINE_END.finditer(block.text, start):
                if match.end() <= start:
                    continue
                piece = self._slice(block, start, match.end())
                if not piece.text:
                    continue
                if not self._fits([*prefix, piece]):
                    break
                end = match.end()
            if end == start:
                if self.chunking_version == "3":
                    return self._single_cell_row_prefix_end(block, start, prefix)
                raise ValueError(
                    "An intake table row exceeds the embedding content token limit"
                )
            return end
        low = start + 1
        high = min(len(block.text), start + max(64, self.content_token_limit * 8))
        while high < len(block.text) and self._fits(
            [*prefix, self._slice(block, start, high)]
        ):
            low = high
            high = min(len(block.text), start + (high - start) * 2)
        end = start
        while low <= high:
            middle = (low + high) // 2
            piece = self._slice(block, start, middle)
            if piece.text and self._fits([*prefix, piece]):
                end, low = middle, middle + 1
            else:
                high = middle - 1
        if end == start:
            raise ValueError(
                "An intake heading leaves no embedding capacity for its body"
            )
        for pattern in (_SENTENCE_END, _WORD_END):
            boundaries = [
                match.end()
                for match in pattern.finditer(block.text, start, end)
                if block.text[start : match.end()].strip()
                and _NUMBERING_ONLY.fullmatch(block.text[start : match.end()].strip())
                is None
                and self._fits([*prefix, self._slice(block, start, match.end())])
            ]
            if boundaries:
                end = boundaries[-1]
                break
        return end

    def _single_cell_row_prefix_end(
        self, block: SourceBlock, start: int, prefix: list[SourceBlock]
    ) -> int:
        row_end = block.text.find("\n", start)
        if row_end == -1:
            row_end = len(block.text)
        row = block.text[start:row_end]
        pipes: list[int] = []
        escaped = False
        for index, character in enumerate(row):
            if character == "|" and not escaped:
                pipes.append(index)
            escaped = not escaped if character == "\\" else False
        if (
            len(pipes) != 2
            or pipes != [0, len(row) - 1]
            or "`" in row
            or _TABLE_SEPARATOR_CELL.fullmatch(row[1:-1])
        ):
            raise ValueError(
                "An intake table row exceeds the embedding content token limit; "
                "only a single source cell with safe line breaks can continue"
            )
        boundaries = [
            start + match.start()
            for match in _CELL_BREAK.finditer(row)
            if match.start() > 1 and not row[match.start() - 1].isspace()
        ]
        parts: list[tuple[int, int]] = []
        part_start = start
        while part_start < row_end:
            remaining = self._slice(block, part_start, row_end)
            if self._fits([*prefix, remaining]):
                parts.append((part_start, row_end))
                break
            fitting = [
                boundary
                for boundary in boundaries
                if boundary > part_start
                and self._fits([*prefix, self._slice(block, part_start, boundary)])
            ]
            if not fitting:
                raise ValueError(
                    "An intake table row exceeds the embedding content token limit; "
                    "no safe source cell line break fits its final heading and body"
                )
            # Keep numbered subfields together when both resulting slices fit.
            numbered = [
                boundary
                for boundary in fitting
                if _CELL_NUMBERED_BREAK.match(block.text, boundary)
                and self._fits([self._slice(block, boundary, row_end)])
            ]
            part_end = numbered[-1] if numbered else fitting[-1]
            parts.append((part_start, part_end))
            part_start = part_end
            prefix = []

        source_start = self._source_start(block)
        row_index = 0
        for preceding in block.text[:start].splitlines():
            cells = preceding.strip().strip("|").split("|")
            if cells and all(_TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in cells):
                continue
            row_index += 1
        row_hash = hashlib.sha256(row.encode()).hexdigest()
        for index, (part_start, part_end) in enumerate(parts):
            fragment = self._slice(block, part_start, part_end)
            self._table_row_fragments[(block.block_id, fragment.start_char)] = (
                IntakeTableRow(
                    version=1,
                    table_index=None,
                    row_index=row_index,
                    cell_index=0,
                    row_sha256=row_hash,
                    row_source_start_char=source_start + start,
                    row_source_end_char=source_start + row_end,
                    fragment_source_start_char=fragment.start_char,
                    fragment_source_end_char=fragment.end_char,
                    fragment_index=index,
                    fragment_count=len(parts),
                )
            )
        return parts[0][1]

    @staticmethod
    def _is_heading(block: SourceBlock, draft: _ChunkDraft) -> bool:
        if block.kind == "heading":
            return True
        clean = _clean_inline_markdown(block.text)
        if draft.article_title is not None and _same_label(clean, draft.article_title):
            return True
        first = _clean_inline_markdown(draft.blocks[0].text)
        if (
            len(draft.blocks) > 1
            and block is draft.blocks[1]
            and draft.article_no is not None
            and parse_regulatory_article_heading(first) is not None
            and not _article_rest_from_original(first)
            and _is_probable_standalone_heading(clean)
        ):
            return True
        if (
            block is not draft.blocks[0]
            or draft.article_no is None
            or parse_regulatory_article_heading(clean) is None
        ):
            return False
        rest = _article_rest_from_original(clean)
        return (
            not rest
            or _is_probable_standalone_heading(rest)
            or (
                draft.article_title is not None
                and _same_label(rest, draft.article_title)
            )
        )

    def _build_drafts(
        self, blocks: list[SourceBlock], *, metadata: DocumentMetadata
    ) -> tuple[list[_ChunkDraft], list[StructureNode]]:
        content_blocks = blocks
        if (
            len(blocks) > 1
            and blocks[0].text.startswith("#")
            and metadata.title is not None
            and _same_label(blocks[0].text, metadata.title)
            and parse_regulatory_article_heading(_clean_inline_markdown(blocks[0].text))
            is None
        ):
            # The document node already carries this title; it cannot name MADDE 1.
            content_blocks = blocks[1:]
        drafts, structure = super()._build_drafts(content_blocks, metadata=metadata)
        drafts, structure = self._explicit_article_headings(
            drafts, structure, blocks=blocks, metadata=metadata
        )
        retained: list[_ChunkDraft] = []
        for index, draft in enumerate(drafts):
            child = drafts[index + 1] if index + 1 < len(drafts) else None
            if (
                draft.blocks
                and child is not None
                and child.article_no == draft.article_no
                and draft.parent_id is not None
                and draft.parent_id in child.parent_ids
                and (
                    draft.chunk_type == "article"
                    and all(self._is_heading(block, draft) for block in draft.blocks)
                    or draft.chunk_type == "paragraph"
                    and draft.article_no is not None
                    and _blocks_text(draft.blocks).rstrip().endswith((";", ":"))
                    and self._fits([*draft.blocks, *child.blocks])
                )
            ):
                drafts[index + 1] = replace(
                    child, blocks=[*draft.blocks, *child.blocks]
                )
            else:
                retained.append(draft)
        if self.chunking_version in {"2", "3"}:
            retained = self._merge_adjacent_table_headings(retained, blocks=blocks)
        return retained, structure

    def _merge_adjacent_table_headings(
        self, drafts: list[_ChunkDraft], *, blocks: list[SourceBlock]
    ) -> list[_ChunkDraft]:
        positions = {block.block_id: index for index, block in enumerate(blocks)}
        retained: list[_ChunkDraft] = []
        for index, draft in enumerate(drafts):
            following = drafts[index + 1] if index + 1 < len(drafts) else None
            text = _blocks_text(draft.blocks)
            if (
                draft.chunk_type == "free_text"
                and len(draft.blocks) == 1
                and len(text) <= _TABLE_HEADING_MAX_CHARS
                and len(text.splitlines()) == 1
                and text.rstrip().endswith((":", "："))
                and following is not None
                and following.chunk_type == "table"
                and following.blocks
                and positions[draft.blocks[-1].block_id] + 1
                == positions[following.blocks[0].block_id]
                and all(
                    getattr(draft, field) == getattr(following, field)
                    for field in (
                        "parent_id",
                        "parent_ids",
                        "parent_path",
                        "heading_path",
                        "article_no",
                        "article_title",
                        "paragraph_no",
                        "clause_label",
                        "subclause_label",
                        "appendix_label",
                    )
                )
                and self._fits([*draft.blocks, *following.blocks])
            ):
                # Retain both original blocks; the table supplies the heading's body.
                drafts[index + 1] = replace(
                    following, blocks=[*draft.blocks, *following.blocks]
                )
            else:
                retained.append(draft)
        return retained

    @staticmethod
    def _explicit_article_headings(
        drafts: list[_ChunkDraft],
        structure: list[StructureNode],
        *,
        blocks: list[SourceBlock],
        metadata: DocumentMetadata,
    ) -> tuple[list[_ChunkDraft], list[StructureNode]]:
        block_positions = {block.block_id: index for index, block in enumerate(blocks)}
        explicit: dict[str, tuple[str, str | None]] = {}
        for node in structure:
            if node.node_type != "article" or node.source_block not in block_positions:
                continue
            position = block_positions[node.source_block]
            block = blocks[position]
            if re.match(r"^#{1,6}\s+", block.text) is None:
                continue
            classified = _classify_block(block)
            if classified["kind"] != "article":
                continue
            rest = str(classified.get("rest") or "").strip()
            title = rest if rest and _article_unit(rest, block.text) is None else None
            if not rest and position + 1 < len(blocks):
                following = blocks[position + 1]
                clean = _clean_inline_markdown(following.text)
                if (
                    _classify_block(following)["kind"] in {"heading", "paragraph"}
                    and following.kind != "table"
                    and _article_unit(clean, following.text) is None
                    and _is_probable_standalone_heading(clean)
                ):
                    title = clean
            label = str(
                classified.get("article_heading") or f"MADDE {classified['article_no']}"
            )
            explicit[node.node_id] = (f"{label} - {title}" if title else label, title)
        if not explicit:
            return drafts, structure
        corrected: dict[str, StructureNode] = {}
        for node in structure:
            parent = corrected.get(node.parent_id or "")
            corrected[node.node_id] = _make_structure_node(
                metadata=metadata,
                node_type=node.node_type,
                label=explicit.get(node.node_id, (node.label, None))[0],
                parent_id=parent.node_id if parent else node.parent_id,
                level=node.level,
                order=node.order,
                source_block=node.source_block,
            )
        remapped = []
        for draft in drafts:
            title = draft.article_title
            for identifier in reversed(draft.parent_ids):
                if identifier in explicit:
                    title = explicit[identifier][1]
                    break
            path = [corrected[identifier].label for identifier in draft.parent_ids]
            remapped.append(
                replace(
                    draft,
                    parent_id=corrected[draft.parent_id].node_id
                    if draft.parent_id
                    else None,
                    parent_ids=[
                        corrected[identifier].node_id for identifier in draft.parent_ids
                    ],
                    parent_path=path,
                    heading_path=path,
                    article_title=title,
                )
            )
        return remapped, list(corrected.values())

    def _split_oversized_draft(self, draft: _ChunkDraft) -> list[_ChunkDraft]:
        if self._fits(draft.blocks):
            return [draft]
        groups: list[list[SourceBlock]] = []
        current: list[SourceBlock] = []
        for block in draft.blocks:
            start = 0
            while start < len(block.text):
                while start < len(block.text) and block.text[start].isspace():
                    start += 1
                if start == len(block.text):
                    break
                fragment = self._table_row_fragments.get(
                    (block.block_id, self._source_start(block) + start)
                )
                if fragment is not None:
                    end = fragment.fragment_source_end_char - self._source_start(block)
                    piece = self._slice(block, start, end)
                    if not self._fits([*current, piece]):
                        raise ValueError(
                            "An intake table continuation exceeds its budget"
                        )
                    groups.append([*current, piece])
                    current = []
                    start = end
                    continue
                if len(block.text) - start <= max(64, self.content_token_limit * 8):
                    remaining = (
                        block
                        if start == 0
                        else self._slice(block, start, len(block.text))
                    )
                    if self._fits([*current, remaining]):
                        current.append(remaining)
                        break
                if current and not all(
                    self._is_heading(item, draft) for item in current
                ):
                    groups.append(current)
                    current = []
                    continue
                if self._is_heading(block, draft):
                    if not self._fits([*current, block]):
                        raise ValueError(
                            "An intake heading exceeds the embedding content token limit"
                        )
                    current.append(block)
                    break
                end = self._prefix_end(block, start, current)
                current.append(self._slice(block, start, end))
                groups.append(current)
                current = []
                start = end
        if current:
            groups.append(current)
        return [
            self._copy_draft_part(draft, blocks, part)
            for part, blocks in enumerate(groups, start=1)
        ]

    def _finalize_chunk(
        self, draft: _ChunkDraft, metadata: DocumentMetadata, order: int
    ) -> RegulatoryChunk:
        chunk = super()._finalize_chunk(draft, metadata, order)
        fragments = [
            fragment
            for block in draft.blocks
            if (
                fragment := self._table_row_fragments.get(
                    (block.block_id, block.start_char)
                )
            )
            is not None
        ]
        if not fragments:
            return chunk
        if len(fragments) != 1 or draft.table_index is None:
            raise ValueError(
                "An intake table continuation lost its source row identity"
            )
        fragment = replace(fragments[0], table_index=draft.table_index)
        return replace(
            chunk,
            metadata=IntakeChunkMetadata(
                **asdict(chunk.metadata), intake_table_row=fragment
            ),
        )

    def _build_hierarchical_aggregate_chunks(
        self,
        *,
        atomic_chunks: list[RegulatoryChunk],
        structure: list[StructureNode],
        blocks: list[SourceBlock],
        metadata: DocumentMetadata,
    ) -> list[RegulatoryChunk]:
        retained = [
            chunk
            for chunk in super()._build_hierarchical_aggregate_chunks(
                atomic_chunks=atomic_chunks,
                structure=structure,
                blocks=blocks,
                metadata=metadata,
            )
            if len(self.tokenizer.encode(chunk.text)) <= self.content_token_limit
        ]
        aggregates: list[RegulatoryChunk] = []
        for index, chunk in enumerate(retained):
            aggregate_metadata = replace(
                chunk.metadata, chunk_order=len(atomic_chunks) + index
            )
            if isinstance(aggregate_metadata, IntakeChunkMetadata):
                aggregate_metadata = replace(aggregate_metadata, intake_table_row=None)
            aggregates.append(replace(chunk, metadata=aggregate_metadata))
        return aggregates
