from datetime import datetime, timedelta, timezone
from io import BytesIO

from pypdf import PdfReader

from onyx.db.models import AnswerGraphEdge, AnswerGraphNode, AnswerGraphRun
from onyx.server.manage.answer_graph.pdf_report import build_answer_graph_pdf


def test_pdf_contains_every_ordered_operation_and_its_relations() -> None:
    started = datetime(2026, 9, 30, tzinfo=timezone.utc)
    root = AnswerGraphNode(
        node_id="root",
        parent_node_id=None,
        kind="step",
        operation="chat.input",
        status="COMPLETE",
        started_at=started,
        attributes={"agent": "run_llm_loop"},
    )
    later = [
        AnswerGraphNode(
            node_id=f"node-{number}",
            parent_node_id="root",
            kind="generation",
            operation=f"regulatory_coverage_plan_{number}",
            status="COMPLETE",
            started_at=started + timedelta(seconds=number),
            attributes={},
        )
        for number in range(1, 13)
    ]
    run = AnswerGraphRun(assistant_message_id=2657, status="COMPLETE")
    document = build_answer_graph_pdf(
        run,
        [later[-1], *later[:-1], root],
        [AnswerGraphEdge(from_node_id="root", to_node_id="node-1", kind="data")],
    )
    reader = PdfReader(BytesIO(document))
    extracted = "\n".join(page.extract_text() for page in reader.pages)

    assert len(reader.pages) > 1
    assert extracted.index("#001") < extracted.index("#013")
    assert "run_llm_loop" in extracted
    assert "parent #1" in extracted
    assert "data from #1" in extracted
    assert "regulatory_coverage_plan_12" in extracted


def test_pdf_shows_label_contribution_summary_without_payload() -> None:
    node = AnswerGraphNode(
        node_id="label",
        parent_node_id=None,
        kind="step",
        operation="search.label_evidence_selection",
        status="COMPLETE",
        started_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        attributes={"summary": "2 labeled visible; 1 added"},
    )
    run = AnswerGraphRun(assistant_message_id=42, status="COMPLETE")
    document = build_answer_graph_pdf(run, [node], [])
    extracted = "\n".join(
        page.extract_text() for page in PdfReader(BytesIO(document)).pages
    )

    assert "search.label_evidence_selection" in extracted
    assert "2 labeled visible; 1 added" in extracted
