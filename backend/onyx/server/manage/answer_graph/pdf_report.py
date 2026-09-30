"""A bounded, metadata-only PDF map of an answer's recorded execution."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from onyx.db.models import AnswerGraphEdge, AnswerGraphNode, AnswerGraphRun

_PAGE_WIDTH, _PAGE_HEIGHT = landscape(A4)
_MARGIN = 42
_CARD_HEIGHT = 47
_PHASE_COLORS: dict[str, colors.Color] = {
    "input": colors.HexColor("#64748b"),
    "planning": colors.HexColor("#7c3aed"),
    "retrieval": colors.HexColor("#0891b2"),
    "model": colors.HexColor("#2563eb"),
    "answer": colors.HexColor("#059669"),
    "operation": colors.HexColor("#d97706"),
}


def _phase(node: AnswerGraphNode) -> str:
    operation = node.operation.lower()
    if operation == "chat.input":
        return "input"
    if operation == "answer.delivered":
        return "answer"
    if node.kind == "generation" or operation.startswith("llm."):
        return "planning" if operation.startswith("regulatory_") else "model"
    if any(
        term in operation
        for term in (
            "search",
            "embed",
            "rerank",
            "retriev",
            "bm25",
            "keyword",
            "vector",
        )
    ):
        return "retrieval"
    return "operation"


def _fitted(value: str, width: float, *, font_size: int = 9) -> str:
    if stringWidth(value, "Helvetica", font_size) <= width:
        return value
    while value and stringWidth(value + "...", "Helvetica", font_size) > width:
        value = value[:-1]
    return value + "..."


def build_answer_graph_pdf(
    run: AnswerGraphRun,
    nodes: list[AnswerGraphNode],
    edges: list[AnswerGraphEdge],
) -> bytes:
    """Render dependency stages and agent lanes without decrypting payloads."""
    ordered = sorted(nodes, key=lambda node: (node.started_at, node.node_id))
    by_id = {node.node_id: node for node in ordered}
    sequence = {node.node_id: index + 1 for index, node in enumerate(ordered)}
    incoming: dict[str, list[int]] = defaultdict(list)
    for edge in edges:
        if edge.kind == "data" and edge.from_node_id in sequence:
            incoming[edge.to_node_id].append(sequence[edge.from_node_id])
    root = next((node for node in ordered if node.operation == "chat.input"), None)
    recorded_agent = root.attributes.get("agent") if root else None
    fallback_agent = (
        recorded_agent
        if isinstance(recorded_agent, str) and recorded_agent
        else "Answer workflow - agent unrecorded"
    )
    owners: dict[str, str] = {}

    def owner(node: AnswerGraphNode, seen: set[str]) -> str:
        if node.node_id in owners:
            return owners[node.node_id]
        if node.node_id in seen:
            return fallback_agent
        seen.add(node.node_id)
        name = node.attributes.get(
            "agent" if node.operation == "research_agent" else "name"
        )
        if (
            (node.kind == "agent" or node.operation == "research_agent")
            and isinstance(name, str)
            and name
        ):
            result = name
        elif node.parent_node_id in by_id:
            result = owner(by_id[node.parent_node_id], seen)
        else:
            result = fallback_agent
        owners[node.node_id] = result
        seen.remove(node.node_id)
        return result

    predecessors: dict[str, set[str]] = defaultdict(set)
    for node in ordered:
        if (
            node.parent_node_id in sequence
            and sequence[node.parent_node_id] < sequence[node.node_id]
        ):
            predecessors[node.node_id].add(node.parent_node_id)
    for edge in edges:
        if (
            edge.from_node_id in sequence
            and edge.to_node_id in sequence
            and sequence[edge.from_node_id] < sequence[edge.to_node_id]
        ):
            predecessors[edge.to_node_id].add(edge.from_node_id)
    previous_by_agent: dict[str, list[AnswerGraphNode]] = defaultdict(list)
    order_after: dict[str, int] = {}
    for node in ordered:
        previous = previous_by_agent[owner(node, set())]
        finished = [
            candidate
            for candidate in previous
            if candidate.ended_at and candidate.ended_at <= node.started_at
        ]
        if finished:
            latest = max(finished, key=lambda candidate: candidate.ended_at)
            if latest.node_id not in predecessors[node.node_id]:
                predecessors[node.node_id].add(latest.node_id)
                order_after[node.node_id] = sequence[latest.node_id]
        previous.append(node)
    ranks: dict[str, int] = {}
    stages: dict[int, dict[str, list[AnswerGraphNode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for node in ordered:
        ranks[node.node_id] = max(
            (ranks[parent_id] + 1 for parent_id in predecessors[node.node_id]),
            default=0,
        )
        stages[ranks[node.node_id]][owner(node, set())].append(node)

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle(f"Answer execution - message {run.assistant_message_id}")
    page_number = 0

    def new_page() -> float:
        nonlocal page_number
        if page_number:
            pdf.showPage()
        page_number += 1
        pdf.setFillColor(colors.HexColor("#101827"))
        pdf.rect(0, _PAGE_HEIGHT - 90, _PAGE_WIDTH, 90, stroke=0, fill=1)
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(_MARGIN, _PAGE_HEIGHT - 38, "Answer execution")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(
            _MARGIN,
            _PAGE_HEIGHT - 58,
            f"Assistant message {run.assistant_message_id}  |  {run.status}  |  "
            f"{len(ordered)} operations  |  {len(edges)} recorded links",
        )
        pdf.setFillColor(colors.HexColor("#64748b"))
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(
            _PAGE_WIDTH - _MARGIN,
            22,
            f"Page {page_number}  |  Dependency stages and parallel agent lanes",
        )
        return _PAGE_HEIGHT - 112

    y = new_page()
    if not ordered:
        pdf.setFillColor(colors.HexColor("#334155"))
        pdf.drawString(_MARGIN, y - 20, "No operations have been recorded yet.")
    usable_width = _PAGE_WIDTH - 2 * _MARGIN
    for rank, agents in sorted(stages.items()):
        names = list(agents)
        for first_lane in range(0, len(names), 3):
            lane_names = names[first_lane : first_lane + 3]
            lane_nodes = [agents[name] for name in lane_names]
            max_count = max(map(len, lane_nodes))
            for first_row in range(0, max_count, 5):
                slices = [items[first_row : first_row + 5] for items in lane_nodes]
                row_count = max(map(len, slices))
                panel_height = 34 + row_count * (_CARD_HEIGHT + 5)
                if y - panel_height < 48:
                    y = new_page()
                pdf.setFillColor(colors.HexColor("#e2e8f0"))
                pdf.roundRect(
                    _MARGIN,
                    y - panel_height,
                    usable_width,
                    panel_height,
                    7,
                    stroke=0,
                    fill=1,
                )
                pdf.setFillColor(colors.HexColor("#334155"))
                pdf.setFont("Helvetica-Bold", 9)
                pdf.drawString(
                    _MARGIN + 9,
                    y - 14,
                    f"STAGE {rank + 1}  |  parallel lanes"
                    if len(lane_names) > 1
                    else f"STAGE {rank + 1}",
                )
                column_width = (usable_width - 14) / len(lane_names)
                for lane_index, (name, items) in enumerate(zip(lane_names, slices)):
                    x = _MARGIN + 7 + lane_index * column_width
                    card_width = column_width - 7
                    pdf.setFont("Helvetica-Bold", 8)
                    pdf.setFillColor(colors.HexColor("#475569"))
                    pdf.drawString(
                        x + 4, y - 27, _fitted(name, card_width - 8, font_size=8)
                    )
                    for row_index, node in enumerate(items):
                        top = y - 34 - row_index * (_CARD_HEIGHT + 5)
                        phase = _phase(node)
                        pdf.setFillColor(colors.white)
                        pdf.roundRect(
                            x,
                            top - _CARD_HEIGHT,
                            card_width,
                            _CARD_HEIGHT,
                            4,
                            stroke=0,
                            fill=1,
                        )
                        pdf.setFillColor(_PHASE_COLORS[phase])
                        pdf.roundRect(
                            x, top - _CARD_HEIGHT, 3, _CARD_HEIGHT, 1, stroke=0, fill=1
                        )
                        pdf.setFont("Helvetica-Bold", 9)
                        pdf.drawString(
                            x + 9, top - 11, f"#{sequence[node.node_id]:03d}"
                        )
                        pdf.setFillColor(colors.HexColor("#0f172a"))
                        pdf.drawString(
                            x + 43,
                            top - 11,
                            _fitted(node.operation, card_width - 52, font_size=9),
                        )
                        pdf.setFont("Helvetica", 7)
                        pdf.setFillColor(colors.HexColor("#475569"))
                        pdf.drawString(
                            x + 9,
                            top - 25,
                            _fitted(
                                f"{node.kind}  |  {node.status}  |  {node.started_at:%H:%M:%S} UTC",
                                card_width - 18,
                                font_size=7,
                            ),
                        )
                        parent = (
                            sequence.get(node.parent_node_id)
                            if node.parent_node_id
                            else None
                        )
                        relations = [
                            f"parent #{parent}"
                            if parent is not None
                            else "branch start"
                        ]
                        if incoming[node.node_id]:
                            relations.append(
                                "data from "
                                + ", ".join(
                                    f"#{value}"
                                    for value in sorted(set(incoming[node.node_id]))
                                )
                            )
                        if node.node_id in order_after:
                            relations.append(f"after #{order_after[node.node_id]}")
                        pdf.drawString(
                            x + 9,
                            top - 38,
                            _fitted(
                                "  |  ".join(relations), card_width - 18, font_size=7
                            ),
                        )
                y -= panel_height + 10
    pdf.save()
    return output.getvalue()
