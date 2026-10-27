"""A selector can change only failed evidence, never labels or canonical words."""

import json
from hashlib import sha256
from typing import Any

import pytest

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling import intake_quote_repair as repair_module
from onyx.regulatory.labeling.intake_quote_repair import (
    QuoteRepair,
    build_quote_repair,
    evidence_catalog,
    parse_quote_repair_output,
    quote_repair_audits,
    validate_quote_repair,
)
from onyx.regulatory.labeling.provider import (
    LabelDefinition,
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)

TEXT = "Başlık.\nDiğer sınıfların harfleri normal 8 punto olmalıdır.\nPiktogramın altında ortalanmış olmalıdır."


def _taxonomy() -> TaxonomyDefinition:
    return TaxonomyDefinition(
        name="Labels",
        labels=[
            LabelDefinition(id=label, name=label, description=label)
            for label in ["exact", "whitespace", "bad"]
        ],
    )


def _raw(labels: list[tuple[str, str]] | None = None, **extra: Any) -> str:
    return json.dumps(
        {
            "labels": [
                {"label_id": label, "evidence_quote": quote}
                for label, quote in (
                    labels
                    if labels is not None
                    else [("bad", "Diğer sınıfların harfi normal 8 punto olmalıdır.")]
                )
            ],
            "abstained": False,
            **extra,
        },
        ensure_ascii=False,
    )


def _build(
    text: str = TEXT,
    raw: str | None = None,
    *,
    base_request: VertexBatchRequest | None = None,
) -> QuoteRepair:
    source_hash = context_hash(text)
    request = base_request or build_labeling_request(
        chunk_id="rc_test",
        text=text,
        context="Full source context: no truncation.",
        taxonomy=_taxonomy(),
        source_hash=source_hash,
    )
    return build_quote_repair(
        base_request=request,
        rejected_response=raw or _raw(),
        text=text,
        chunk_id="rc_test",
        source_hash=source_hash,
        taxonomy=_taxonomy(),
    )


def _selection(
    repair: QuoteRepair, *, label: str = "bad", evidence_id: str | None = None
) -> str:
    entry = next(entry for entry in repair.catalog if "harfleri" in entry.text)
    return json.dumps(
        {
            "labels": [
                {"label_id": label, "evidence_id": evidence_id or entry.evidence_id}
            ],
            "abstained": False,
        }
    )


def _line(repair: QuoteRepair, content: str, **changes: Any) -> str:
    return json.dumps(
        {
            "key": repair.request.request_hash,
            "request": repair.request.to_generate_content_request(),
            "response": {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": content}]}}
                ]
            },
            **changes,
        }
    )


@pytest.mark.parametrize(
    "text", [TEXT, "a" * 3500, " \n\t" + "paragraf\u00a0" * 600, "A." * 800]
)
def test_catalog_has_exact_deterministic_complete_source_coverage(text: str) -> None:
    entries = evidence_catalog(text)
    assert entries == evidence_catalog(text)
    covered = [False] * len(text)
    for entry in entries:
        assert entry.text == text[entry.start : entry.end]
        assert 1 <= len(entry.text) <= 1024
        for position in range(entry.start, entry.end):
            covered[position] = True
    assert all(covered)
    assert {entry.evidence_id for entry in entries}.isdisjoint(
        {entry.evidence_id for entry in evidence_catalog(text + "!")}
    )


def test_catalog_growth_limit_is_explicit_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repair_module, "MAX_CATALOG_ENTRIES", 2)
    with pytest.raises(ValueError, match="too large"):
        evidence_catalog("First. Second. Third. Fourth.")


def test_selector_fixes_actual_grammar_error_and_retains_original_valid_quotes() -> (
    None
):
    raw = _raw(
        [
            ("exact", "Başlık."),
            ("whitespace", "Başlık. Diğer sınıfların"),
            ("bad", "Piktogramın altında ortalanmalıdır."),
        ]
    )
    repair = _build(raw=raw)
    base = json.loads(repair.request.prompt)
    assert (
        base["target"]["text"] == TEXT
        and base["interpretation_context"] == "Full source context: no truncation."
    )
    assert base["quote_repair"]["rejected_response"] == raw
    assert base["quote_repair"]["attempt"] == 1
    assert repair.repair_label_ids == frozenset({"bad"})
    evidence = next(
        entry
        for entry in repair.catalog
        if entry.text.strip() == "Piktogramın altında ortalanmış olmalıdır."
    )
    response = _selection(repair, evidence_id=evidence.evidence_id)
    outcome = validate_quote_repair(
        response, repair=repair, text=TEXT, taxonomy=_taxonomy()
    )
    assert [(entry.label_id, entry.evidence_quote) for entry in outcome.labels] == [
        ("exact", "Başlık."),
        ("whitespace", "Başlık.\nDiğer sınıfların"),
        ("bad", evidence.text),
    ]
    assert (
        validate_labeling_response(
            outcome.model_dump_json(), text=TEXT, taxonomy=_taxonomy()
        )
        == outcome
    )
    assert all(
        set(entry.model_dump()) == {"label_id", "evidence_quote"}
        for entry in outcome.labels
    )
    audits = quote_repair_audits(response, repair=repair)
    assert [entry["method"] for entry in audits] == [
        "unicode_whitespace_unique",
        "canonical_evidence_selector",
    ]
    assert audits[-1]["source_start_char"] == evidence.start
    assert audits[-1]["source_end_char"] == evidence.end
    assert (
        audits[-1]["original_quote_sha256"]
        == sha256("Piktogramın altında ortalanmalıdır.".encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "kind", ["unknown-later", "duplicate", "abstained", "extra", "empty", "blank"]
)
def test_invalid_original_never_permits_quote_repair(kind: str) -> None:
    payload = json.loads(_raw())
    if kind == "unknown-later":
        payload["labels"].append({"label_id": "unknown", "evidence_quote": "Başlık."})
    elif kind == "duplicate":
        payload["labels"].append(payload["labels"][0])
    elif kind == "abstained":
        payload["abstained"] = True
    elif kind == "extra":
        payload["extra"] = True
    elif kind == "empty":
        payload["labels"] = []
    else:
        payload["labels"].append({"label_id": "exact", "evidence_quote": " "})
    with pytest.raises(ValueError):
        _build(raw=json.dumps(payload))


@pytest.mark.parametrize(
    "kind",
    [
        "unknown-id",
        "changed-label",
        "drop-label",
        "duplicate",
        "abstained",
        "extra",
        "quote-text",
        "source-changed",
    ],
)
def test_repair_cannot_change_contract_or_source(kind: str) -> None:
    repair = _build()
    payload = json.loads(_selection(repair))
    text = TEXT
    if kind == "unknown-id":
        payload["labels"][0]["evidence_id"] = "e_foreign"
    elif kind == "changed-label":
        payload["labels"][0]["label_id"] = "exact"
    elif kind == "drop-label":
        payload["labels"] = []
    elif kind == "duplicate":
        payload["labels"].append(payload["labels"][0])
    elif kind == "abstained":
        payload["abstained"] = True
    elif kind == "extra":
        payload["extra"] = True
    elif kind == "quote-text":
        payload["labels"][0]["evidence_quote"] = "harfi"
    else:
        text += "!"
    with pytest.raises(ValueError):
        validate_quote_repair(
            json.dumps(payload), repair=repair, text=text, taxonomy=_taxonomy()
        )


@pytest.mark.parametrize(
    "kind",
    ["target-list", "wrong-text", "wrong-hash", "wrong-taxonomy", "wrong-context"],
)
def test_builder_binds_typed_frozen_request(kind: str) -> None:
    repair = _build()
    original = build_labeling_request(
        chunk_id="rc_test",
        text=TEXT,
        context="Full source context: no truncation.",
        taxonomy=_taxonomy(),
        source_hash=context_hash(TEXT),
    )
    prompt = json.loads(original.prompt)
    if kind == "target-list":
        prompt["target"] = []
    elif kind == "wrong-text":
        prompt["target"]["text"] += "!"
    elif kind == "wrong-hash":
        prompt["target"]["source_hash"] = "a" * 64
    elif kind == "wrong-taxonomy":
        prompt["taxonomy"]["name"] = "Foreign"
    else:
        prompt["interpretation_context"] = None
    with pytest.raises(ValueError, match="frozen"):
        _build(base_request=original.model_copy(update={"prompt": json.dumps(prompt)}))
    assert repair.request.request_hash != original.request_hash
    assert _build().request.request_hash == repair.request.request_hash


def test_optional_system_instruction_and_generation_settings_preserved() -> None:
    base = build_labeling_request(
        chunk_id="rc_test",
        text=TEXT,
        context="Full",
        taxonomy=_taxonomy(),
        source_hash=context_hash(TEXT),
    ).model_copy(update={"system_instruction": None})
    repair = _build(base_request=base)
    assert repair.request.system_instruction
    assert (
        repair.request.generation_config is not None
        and base.generation_config is not None
    )
    assert {
        key: value
        for key, value in repair.request.generation_config.items()
        if key != "responseJsonSchema"
    } == {
        key: value
        for key, value in base.generation_config.items()
        if key != "responseJsonSchema"
    }


@pytest.mark.parametrize(
    "kind",
    [
        "success",
        "missing",
        "duplicate",
        "bad-key",
        "bad-request",
        "multiple-candidates",
        "max-tokens",
        "safety",
        "provider-error",
        "malformed",
    ],
)
def test_repair_envelope_requires_exact_echo_single_complete_stop(kind: str) -> None:
    repair = _build()
    content = _selection(repair)
    payload = json.loads(_line(repair, content))
    lines = [json.dumps(payload)]
    if kind == "missing":
        lines = []
    elif kind == "duplicate":
        lines *= 2
    elif kind == "bad-key":
        payload["key"] = "a" * 64
    elif kind == "bad-request":
        payload["request"]["contents"][0]["parts"][0]["text"] += "!"
    elif kind == "multiple-candidates":
        payload["response"]["candidates"] *= 2
    elif kind == "max-tokens":
        payload["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
    elif kind == "safety":
        payload["response"]["promptFeedback"] = {"blockReason": "SAFETY"}
    elif kind == "provider-error":
        payload["error"] = "Failure"
    elif kind == "malformed":
        payload["response"]["candidates"][0]["content"]["parts"] = [{"text": 123}]
    if kind not in {"missing", "duplicate"}:
        lines = [json.dumps(payload)]
    if kind == "success":
        assert (
            parse_quote_repair_output(
                iter(lines), repair=repair, on_progress=lambda: None
            )
            == content
        )
    else:
        with pytest.raises(ValueError):
            parse_quote_repair_output(
                iter(lines), repair=repair, on_progress=lambda: None
            )
