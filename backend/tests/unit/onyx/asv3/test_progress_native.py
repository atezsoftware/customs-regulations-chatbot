import json

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.progress import (
    ProgressReporter,
    official_corpus_source_name,
    report_source_deliveries,
)
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.tracing.flows import LLMFlow


def test_resume_retains_neutral_start_before_first_action_language() -> None:
    reporter = ProgressReporter("native-run", "und")
    initial = reporter.report("started")
    reporter.language = "tr"
    first_action = reporter.report(
        "tools", title="Kaynaklar inceleniyor", message="İlgili hükmü okuyorum."
    )
    resumed = ProgressReporter("native-run", "tr")
    resumed.restore(reporter.export())
    continued = resumed.report("resume")

    assert [event.language for event in resumed.snapshot()] == ["und", "tr", "tr"]
    assert [event.sequence for event in resumed.snapshot()] == [1, 2, 3]
    assert initial.event_id == resumed.snapshot()[0].event_id
    assert first_action.event_id == resumed.snapshot()[1].event_id
    assert continued.language == "tr"


@pytest.mark.parametrize("foreign_language", ["en", "und"])
def test_resume_rejects_language_drift_after_localized_action(
    foreign_language: str,
) -> None:
    reporter = ProgressReporter("native-run", "tr")
    reporter.report("started")
    reporter.language = foreign_language
    reporter.report("tools")
    payload = reporter.export()
    payload["language"] = "tr"

    with pytest.raises(ValueError, match="Invalid progress event"):
        ProgressReporter("native-run", "tr").restore(payload)


def test_resume_still_rejects_a_different_profile_identity() -> None:
    reporter = ProgressReporter("native-run", "und")
    reporter.report("started")
    reporter.language = "tr"
    reporter.report("tools")

    with pytest.raises(ValueError, match="identity mismatch"):
        ProgressReporter("native-run", "en").restore(reporter.export())


def canonical_item(
    source: str, chunk: str, text: str, metadata: dict[str, JsonValue]
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source,
        chunk_id=chunk,
        text=text,
        metadata=metadata,
        search_doc=SearchDoc(
            document_id=source,
            chunk_ind=0,
            semantic_identifier="Internal catalogue/source_file.md",
            blurb="",
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": chunk},
            match_highlights=[],
        ),
    )


def source_ledger(context: RunContext) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for chunk, article, text in [
        ("permission", "168", "The applicant must request the relief."),
        ("condition", "168", "Institution checks must also be completed."),
        ("exception", "169", "The goods must be returned unchanged."),
    ]:
        ledger.add(
            [
                canonical_item(
                    "customs-law",
                    chunk,
                    text,
                    {
                        "heading_path": [
                            "4458 SAYILI GÜMRÜK KANUNU",
                            f"MADDE {article}",
                        ],
                        "canonical_metadata": {
                            "title": "İhracat Mevzuatı/Kanun/gumruk_kanunu_.md",
                            "article_no": article,
                        },
                    },
                )
            ],
            context,
        )
    return ledger


def delivered(ledger: EvidenceLedger, call_id: str = "coordinator") -> None:
    ledger.record_delivery(
        call_id,
        LLMFlow.ASV3_COORDINATOR.value,
        json.loads(ledger.serialize_records(ledger.citation_numbers())),
    )


@pytest.mark.parametrize(
    "language,message",
    [
        ("tr", "Bu kaynaktaki ilgili özgün hükümler inceleniyor."),
        ("en", "Reviewing the relevant original provisions in this source."),
        ("fr", "J'examine les dispositions originales pertinentes."),
    ],
)
def test_source_progress_groups_full_originals_by_provision_and_resumes_without_repeats(
    language: str, message: str
) -> None:
    context = RunContext(language=language)
    ledger = source_ledger(context)
    reporter = ProgressReporter(context.run_id, language)
    reporter.report("tools", title="Current action", message="Actual work continues.")
    delivered(ledger)
    budget = context.budget.snapshot()
    pair = ["Sources originales", message]
    report_source_deliveries(ledger, "coordinator", context, reporter, pair)
    events = reporter.snapshot()
    assert events[0].task_id is None and events[0].title == "Current action"
    assert len(events) == 3
    assert all(
        event.task_id and event.task_id.startswith("action:source:")
        for event in events[1:]
    )
    assert all(
        event.status == "completed" and event.phase == "tools" for event in events[1:]
    )
    assert all(
        event.message == message and event.language == language for event in events[1:]
    )
    assert all("4458 SAYILI GÜMRÜK KANUNU" in event.title for event in events[1:])
    assert all(
        "/" not in event.title and ".md" not in event.title for event in events[1:]
    )
    assert events[1].title != events[2].title
    assert context.budget.snapshot() == budget
    resumed = ProgressReporter(context.run_id, language)
    resumed.restore(reporter.export())
    delivered(ledger, "next-coordinator")
    report_source_deliveries(ledger, "next-coordinator", context, resumed, pair)
    assert resumed.snapshot() == events


@pytest.mark.parametrize(
    "case", ["partial", "changed", "worker", "review", "missing", "cancelled"]
)
def test_source_progress_excludes_unexecuted_or_noncoordinator_originals(
    case: str,
) -> None:
    context = RunContext(depth=1 if case == "worker" else 0)
    ledger = source_ledger(context)
    records = json.loads(ledger.serialize_records([1]))
    if case == "partial":
        records[0]["text"] = records[0]["text"][:10]
    if case == "changed":
        records[0]["text"] = "Invented text not in the canonical original."
    flow = LLMFlow.ASV3_VERIFICATION if case == "review" else LLMFlow.ASV3_COORDINATOR
    ledger.record_delivery("current", flow.value, records)
    reporter = ProgressReporter(context.run_id, context.language)
    if case == "cancelled":
        context.cancel()
    report_source_deliveries(
        ledger,
        "missing" if case == "missing" else "current",
        context,
        reporter,
        ["Kaynaklar inceleniyor", "İlgili hükümler inceleniyor."],
    )
    assert reporter.snapshot() == []


def test_source_progress_uses_actual_genelge_heading_and_safe_unknown_source_label() -> (
    None
):
    context, ledger = RunContext(), EvidenceLedger()
    ledger.add(
        [
            canonical_item(
                "circular",
                "circular-rule",
                "Operative circular condition.",
                {
                    "heading_path": ["TİCARET BAKANLIĞI", "(2022/4)"],
                    "canonical_metadata": {
                        "document_type": "genelge",
                        "title": "Genelgeler/Gümrükler genel müdürlüğü/genelge_2022-04.md",
                    },
                },
            ),
            canonical_item(
                "unlabelled",
                "unlabelled-rule",
                "Another original condition.",
                {"title": "/tmp/private_source_query.pdf"},
            ),
        ],
        context,
    )
    delivered(ledger)
    reporter = ProgressReporter(context.run_id, context.language)
    report_source_deliveries(
        ledger, "coordinator", context, reporter, ["Kaynak", "Pasaj"]
    )
    assert [event.title for event in reporter.snapshot()] == [
        "2022/4 sayılı Genelge",
        "Özgün kaynak [2]",
    ]
    circular, unknown = ledger.get(1), ledger.get(2)
    assert circular is not None and unknown is not None
    assert official_corpus_source_name(circular) == "2022/4 sayılı Genelge"
    assert official_corpus_source_name(unknown) is None


def test_source_progress_stops_emitting_when_cancelled_between_deliveries() -> None:
    context = RunContext()
    ledger = source_ledger(context)
    delivered(ledger)
    reporter = ProgressReporter(
        context.run_id, context.language, emit=lambda _event: context.cancel()
    )
    report_source_deliveries(
        ledger, "coordinator", context, reporter, ["Kaynak", "Pasaj"]
    )
    assert len(reporter.snapshot()) == 1


@pytest.mark.parametrize(
    "case",
    [
        "derived",
        "external",
        "missing_chunk",
        "navigation",
        "wrong_source",
        "wrong_chunk",
    ],
)
def test_original_provision_notice_requires_aligned_citable_canonical_chunk(
    case: str,
) -> None:
    context, ledger = RunContext(), EvidenceLedger()
    item = canonical_item(
        "law", "article", "Actual source text.", {"title": "Source law"}
    )
    assert item.search_doc is not None
    if case in {"derived", "external"}:
        item.metadata[case] = True
    elif case == "missing_chunk":
        item.chunk_id = None
    elif case == "navigation":
        item.search_doc.metadata = {}
    elif case == "wrong_source":
        item.search_doc.document_id = "different-law"
    else:
        item.search_doc.metadata["regulatory_chunk_id"] = "different-chunk"
    ledger.add([item], context)
    assert official_corpus_source_name(item) is None
    delivered(ledger)
    reporter = ProgressReporter(context.run_id, context.language)
    report_source_deliveries(
        ledger, "coordinator", context, reporter, ["Kaynak", "Pasaj"]
    )
    assert reporter.snapshot() == []


def test_compacted_progress_retains_source_action_identity_across_resume() -> None:
    context, ledger = RunContext(), EvidenceLedger()
    ledger.add(
        [
            canonical_item(
                "law",
                f"article-{number}",
                f"Original provision {number}.",
                {"heading_path": ["GÜMRÜK KANUNU", f"MADDE {number}"]},
            )
            for number in range(1, 226)
        ],
        context,
    )
    delivered(ledger)
    reporter = ProgressReporter(context.run_id, context.language)
    report_source_deliveries(
        ledger, "coordinator", context, reporter, ["Kaynak", "Pasaj"]
    )
    before = reporter.snapshot()
    assert len({event.task_id for event in before}) == 225
    resumed = ProgressReporter(context.run_id, context.language)
    resumed.restore(reporter.export())
    delivered(ledger, "next-coordinator")
    report_source_deliveries(
        ledger, "next-coordinator", context, resumed, ["Kaynak", "Pasaj"]
    )
    assert resumed.snapshot() == before


def test_official_source_name_excludes_article_and_does_not_infer_type_from_number() -> (
    None
):
    context = RunContext()
    ledger = source_ledger(context)
    article = ledger.get(1)
    assert article is not None
    assert official_corpus_source_name(article) == "4458 SAYILI GÜMRÜK KANUNU"
    number_only = canonical_item(
        "number-only",
        "heading",
        "Original text.",
        {
            "heading_path": ["TİCARET BAKANLIĞI", "(2022/4)"],
            "title": "genelge_2022-04.md",
        },
    )
    assert official_corpus_source_name(number_only) is None
