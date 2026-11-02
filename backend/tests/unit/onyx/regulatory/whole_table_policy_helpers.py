"""Larger source tables must preserve rows without changing prose or retries."""

from uuid import uuid4

from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.document_intake.models import (
    IntakeSourceContext,
)


def _context() -> IntakeSourceContext:
    return IntakeSourceContext(
        package_id=uuid4(),
        asset_id=uuid4(),
        asset_sha256="a" * 64,
        parent_asset_hashes=[],
        parent_asset_ids=[],
        parent_titles=[],
        source_links=[],
    )


class _CharacterTokenizer(BaseTokenizer):
    def encode(self, string: str) -> list[int]:
        return [ord(character) for character in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)
