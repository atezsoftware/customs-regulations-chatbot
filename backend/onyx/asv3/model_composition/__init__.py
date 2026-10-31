"""Explicit Fast model compositions; imports perform no model or database work."""

from onyx.asv3.model_composition.models import (
    MODEL_COMPOSITION_SERVICE,
    CompositionProfileId,
    CompositionRole,
    ModelBinding,
    ModelComposition,
    get_model_composition,
)

__all__ = [
    "MODEL_COMPOSITION_SERVICE",
    "CompositionProfileId",
    "CompositionRole",
    "ModelBinding",
    "ModelComposition",
    "get_model_composition",
]
