"""Retain literal Markdown headings that structural inference would discard."""

import re

from onyx.regulatory.chunker import (
    DocumentMetadata,
    SourceBlock,
    StructureNode,
    _ChunkDraft,
    _same_label,
)


def retain_markdown_headings(
    drafts: list[_ChunkDraft],
    structure: list[StructureNode],
    *,
    blocks: list[SourceBlock],
    metadata: DocumentMetadata,
) -> list[_ChunkDraft]:
    if not structure:
        return drafts
    delivered_blocks = {block.block_id for draft in drafts for block in draft.blocks}
    referenced = {identifier for draft in drafts for identifier in draft.parent_ids}
    delivered_blocks.update(
        node.source_block
        for node in structure
        if node.node_id in referenced and node.source_block is not None
    )
    root = structure[0]
    retained = list(drafts)
    for index, block in enumerate(blocks):
        if (
            block.block_id in delivered_blocks
            or re.match(r"^#{1,6}\s+", block.text) is None
            or index == 0
            and metadata.title is not None
            and _same_label(block.text, metadata.title)
        ):
            continue
        # A source heading can contain operative text (e.g. "kapsamaz.").
        # Keep it verbatim without assigning it to a guessed article.
        retained.append(
            _ChunkDraft(
                chunk_type="paragraph",
                blocks=[block],
                parent_id=root.node_id,
                parent_ids=[root.node_id],
                parent_path=[root.label],
                heading_path=[root.label],
            )
        )
    return sorted(retained, key=lambda draft: draft.blocks[0].start_char)
