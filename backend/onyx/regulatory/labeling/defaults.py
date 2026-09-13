"""Bundled label definitions used when starting document-set labeling."""

from importlib.resources import files

from onyx.regulatory.labeling.provider import TaxonomyDefinition


def load_default_taxonomy() -> TaxonomyDefinition:
    definition = (
        files("onyx.regulatory.labeling")
        .joinpath("data", "tariff-regulatory-intelligence-v2.1.json")
        .read_text(encoding="utf-8")
    )
    return TaxonomyDefinition.model_validate_json(definition)
