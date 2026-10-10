"""A completed model proposal may be corrected once without accepting it."""

from pydantic import JsonValue


class ReadingContractError(ValueError):
    def __init__(self, diagnostics: str, candidate: JsonValue) -> None:
        super().__init__(diagnostics)
        self.diagnostics = diagnostics
        self.candidate = candidate
