"""Allowed Vertex analysis models; persisted choices never inherit chat defaults."""

from enum import StrEnum


class AmendmentAnalysisModel(StrEnum):
    FLASH = "gemini-3.8-flash"
    FLASH_LITE = "gemini-3.5-flash-lite"
