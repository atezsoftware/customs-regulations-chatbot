"""A reconsideration cannot relax frozen source, vocabulary or quote authority."""

import json
from typing import Any

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepair,
    AbstentionRepairAudit,
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
    validate_stored_abstention_repair_audit,
)
from onyx.regulatory.labeling.provider import (
    LabelDefinition,
    TaxonomyDefinition,
    build_labeling_request,
)

TEXT = "b) Çözücü su dâhil çözücülerin ortadan kaldırılması, veya"
CONTEXT = (
    "Bir kimyasal reaksiyon yeni bir yapıya sahip bir molekül ortaya çıkarır. "
    "Bununla beraber; a) Su veya diğer çözücüler içinde çözdürme,\n"
    + TEXT
    + "\nc) Kristalizasyon suyunun eklenmesi veya ortadan kaldırılması\n"
    "işlemleri menşe kazanımı açısından dikkate alınmaz."
)
REJECTED = '{"labels":[],"abstained":true}'
BINDING: dict[str, object] = {
    "model": "gemini-3.8-flash",
    "transport": "vertex_direct_v1",
    "provider_id": 1,
    "project": "same-project",
}


def taxonomy() -> TaxonomyDefinition:
    return TaxonomyDefinition(
        name="Frozen definitions",
        labels=[
            LabelDefinition(
                id="origin", name="Menşe", description="Menşe ve menşe ispatı"
            ),
            LabelDefinition(
                id="exception",
                name="İstisna",
                description="Genel kuraldan ayrılan durum",
            ),
        ],
    )


def build(text: str = TEXT, context: str = CONTEXT, **changes: Any) -> AbstentionRepair:
    source_hash = context_hash(text)
    base = build_labeling_request(
        chunk_id="target",
        text=text,
        context=context,
        taxonomy=taxonomy(),
        source_hash=source_hash,
    )
    values: dict[str, Any] = {
        "base_request": base,
        "rejected_response": REJECTED,
        "text": text,
        "chunk_id": "target",
        "source_hash": source_hash,
        "taxonomy": taxonomy(),
        "provider_binding": BINDING,
        **changes,
    }
    return build_abstention_repair(**values)


def response(text: str = TEXT, *, abstained: bool = False, empty: bool = False) -> str:
    return json.dumps(
        {
            "labels": [] if empty else [{"label_id": "origin", "evidence_quote": text}],
            "abstained": abstained,
        },
        ensure_ascii=False,
    )


def line(repair: AbstentionRepair, raw: str, **changes: Any) -> str:
    payload = {
        "key": repair.request.request_hash,
        "request": repair.request.to_generate_content_request(),
        "response": {
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": raw}]}}
            ]
        },
        **changes,
    }
    return json.dumps(payload, ensure_ascii=False)


def audit(
    repair: AbstentionRepair, raw: str, *, text: str = TEXT
) -> AbstentionRepairAudit:
    output = parse_abstention_repair_output(
        [line(repair, raw)], repair=repair, on_progress=lambda: None
    )
    return build_abstention_repair_audit(
        output=output,
        repair=repair,
        output_uri="retained-output",
        text=text,
        taxonomy=taxonomy(),
    )


def validate_audit(
    value: AbstentionRepairAudit | dict[str, Any],
    repair: AbstentionRepair,
    **changes: Any,
) -> AbstentionRepairAudit:
    values: dict[str, Any] = {
        "base_request": repair.base_request,
        "text": TEXT,
        "chunk_id": "target",
        "source_hash": context_hash(TEXT),
        "taxonomy": taxonomy(),
        "provider_binding": BINDING,
        "labels": ["origin"],
        "assignments": [{"label_id": "origin", "evidence_quote": TEXT}],
        **changes,
    }
    return validate_stored_abstention_repair_audit(value, **values)
