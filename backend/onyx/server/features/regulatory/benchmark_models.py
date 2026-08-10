import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from onyx.configs.app_configs import (
    REGULATORY_BENCHMARK_MAX_CANDIDATES,
    REGULATORY_BENCHMARK_MAX_QUESTIONS,
)
from onyx.db.models import (
    BenchmarkQuestion,
    BenchmarkRun,
    BenchmarkRunItem,
    BenchmarkRunJudgment,
)
from onyx.regulatory.benchmark.models import BenchmarkExpectedCitationInput

BenchmarkFact = Annotated[str, Field(min_length=1, max_length=4000)]
BenchmarkTag = Annotated[str, Field(min_length=1, max_length=200)]


class BenchmarkQuestionCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=20000)
    reference_answer: str | None = Field(default=None, max_length=50000)
    expected_facts: list[BenchmarkFact] = Field(default_factory=list, max_length=100)
    expected_citations: list[BenchmarkExpectedCitationInput] = Field(
        default_factory=list, max_length=200
    )
    as_of_date: datetime.date | None = None
    rubric_notes: str | None = Field(default=None, max_length=10000)
    tags: list[BenchmarkTag] = Field(default_factory=list, max_length=50)
    document_set_id: int


class BenchmarkQuestionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    prompt: str | None = Field(default=None, min_length=1, max_length=20000)
    reference_answer: str | None = Field(default=None, max_length=50000)
    expected_facts: list[BenchmarkFact] | None = Field(default=None, max_length=100)
    expected_citations: list[BenchmarkExpectedCitationInput] | None = Field(
        default=None, max_length=200
    )
    as_of_date: datetime.date | None = None
    rubric_notes: str | None = Field(default=None, max_length=10000)
    tags: list[BenchmarkTag] | None = Field(default=None, max_length=50)
    document_set_id: int | None = None
    is_active: bool | None = None


class BenchmarkQuestionSnapshot(BaseModel):
    id: int
    title: str
    prompt: str
    reference_answer: str | None
    expected_facts: list[str]
    expected_citations: list[dict[str, object]]
    as_of_date: datetime.date | None
    rubric_notes: str | None
    tags: list[str]
    document_set_id: int
    document_set_name: str
    is_active: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime

    @classmethod
    def from_model(cls, question: BenchmarkQuestion) -> "BenchmarkQuestionSnapshot":
        return cls(
            id=question.id,
            title=question.title,
            prompt=question.prompt,
            reference_answer=question.reference_answer,
            expected_facts=question.expected_facts,
            expected_citations=question.expected_citations,
            as_of_date=question.as_of_date,
            rubric_notes=question.rubric_notes,
            tags=question.tags,
            document_set_id=question.document_set_id,
            document_set_name=question.document_set.name,
            is_active=question.is_active,
            created_at=question.created_at,
            updated_at=question.updated_at,
        )


class BenchmarkModelSelection(BaseModel):
    provider: str = Field(min_length=1, max_length=200)
    provider_id: int | None = Field(default=None, gt=0)
    model_id: str = Field(min_length=1, max_length=500)


class BenchmarkAvailableModel(BaseModel):
    provider: str
    provider_id: int
    model_id: str
    display_name: str
    max_input_tokens: int | None
    is_visible: bool


class BenchmarkCitationOption(BaseModel):
    chunk_id: str
    user_file_id: str
    file_name: str
    heading_path: list[str]
    text_excerpt: str
    status: Literal["active", "superseded"]
    validity_start_date: datetime.date | None
    validity_end_date: datetime.date | None


class BenchmarkRunCreate(BaseModel):
    label: str | None = Field(default=None, max_length=300)
    question_ids: list[int] | None = Field(
        default=None, min_length=1, max_length=REGULATORY_BENCHMARK_MAX_QUESTIONS
    )
    candidates: list[BenchmarkModelSelection] = Field(
        min_length=1, max_length=REGULATORY_BENCHMARK_MAX_CANDIDATES
    )
    judge: BenchmarkModelSelection
    deep_research: bool = False


class BenchmarkJudgmentSnapshot(BaseModel):
    correctness_score: int
    groundedness_score: int
    completeness_score: int
    clarity_score: int
    overall_score: int
    rationale: str
    report: dict[str, object]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_cents: float | None
    cost_source: str

    @classmethod
    def from_model(cls, judgment: BenchmarkRunJudgment) -> "BenchmarkJudgmentSnapshot":
        return cls.model_validate(judgment, from_attributes=True)


class BenchmarkRunItemSnapshot(BaseModel):
    id: int
    provider: str
    provider_id: int | None
    model_id: str
    question_id: int
    question_prompt: str
    question_title: str
    question_snapshot: dict[str, object]
    status: str
    final_result: str | None
    error_message: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    duration_ms: int | None
    cost_cents: float | None
    cost_source: str
    cited_chunk_ids: list[str]
    cited_sources: list[dict[str, object]]
    execution_steps: list[dict[str, object]]
    llm_calls: list[dict[str, object]]
    answer_reasoning: str | None
    chat_session_id: str | None
    assistant_message_id: int | None
    citation_recall: float | None
    citation_precision: float | None
    judge_error: str | None
    judgment: BenchmarkJudgmentSnapshot | None

    @classmethod
    def from_model(cls, item: BenchmarkRunItem) -> "BenchmarkRunItemSnapshot":
        snapshot = item.question_snapshot or {}
        return cls(
            id=item.id,
            provider=item.provider,
            provider_id=item.provider_id,
            model_id=item.model_id,
            question_id=item.question_id,
            question_prompt=item.question.prompt,
            question_title=str(snapshot.get("title") or item.question.title),
            question_snapshot=snapshot,
            status=item.status,
            final_result=item.final_result,
            error_message=item.error_message,
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            total_tokens=item.total_tokens,
            duration_ms=item.duration_ms,
            cost_cents=item.cost_cents,
            cost_source=item.cost_source,
            cited_chunk_ids=item.cited_chunk_ids,
            cited_sources=item.cited_sources,
            execution_steps=item.execution_steps,
            llm_calls=item.llm_calls,
            answer_reasoning=item.answer_reasoning,
            chat_session_id=(
                str(item.chat_session_id) if item.chat_session_id else None
            ),
            assistant_message_id=item.assistant_message_id,
            citation_recall=item.citation_recall,
            citation_precision=item.citation_precision,
            judge_error=item.judge_error,
            judgment=(
                BenchmarkJudgmentSnapshot.from_model(item.judgment)
                if item.judgment
                else None
            ),
        )


class BenchmarkModelAggregate(BaseModel):
    provider: str
    provider_id: int | None
    model_id: str
    item_count: int
    completed_count: int
    failed_count: int
    average_score: float | None
    average_tokens: float | None
    average_duration_ms: float | None
    total_cost_cents: float | None
    average_citation_recall: float | None
    average_citation_precision: float | None


class BenchmarkRunSnapshot(BaseModel):
    id: int
    label: str | None
    status: str
    judge_provider: str
    judge_provider_id: int | None
    judge_model: str
    deep_research: bool
    total_items: int
    completed_items: int
    failed_items: int
    started_at: datetime.datetime | None
    completed_at: datetime.datetime | None
    created_at: datetime.datetime
    report: dict[str, object] | None
    report_error: str | None
    report_input_tokens: int | None
    report_output_tokens: int | None
    report_cost_cents: float | None
    items: list[BenchmarkRunItemSnapshot]
    aggregates: list[BenchmarkModelAggregate]


def benchmark_run_snapshot(run: BenchmarkRun) -> BenchmarkRunSnapshot:
    grouped: dict[tuple[str, int | None, str], list[BenchmarkRunItem]] = {}
    for item in run.items:
        grouped.setdefault((item.provider, item.provider_id, item.model_id), []).append(
            item
        )
    aggregates: list[BenchmarkModelAggregate] = []
    for (provider, provider_id, model_id), items in grouped.items():
        scores = [item.judgment.overall_score for item in items if item.judgment]
        token_values = [
            item.total_tokens for item in items if item.total_tokens is not None
        ]
        durations = [item.duration_ms for item in items if item.duration_ms is not None]
        costs = [
            (item.cost_cents or 0)
            + (item.judgment.cost_cents or 0 if item.judgment else 0)
            for item in items
            if item.cost_cents is not None
            or (item.judgment is not None and item.judgment.cost_cents is not None)
        ]
        recalls = [
            item.citation_recall for item in items if item.citation_recall is not None
        ]
        precisions = [
            item.citation_precision
            for item in items
            if item.citation_precision is not None
        ]
        aggregates.append(
            BenchmarkModelAggregate(
                provider=provider,
                provider_id=provider_id,
                model_id=model_id,
                item_count=len(items),
                completed_count=sum(item.status == "completed" for item in items),
                failed_count=sum(item.status == "error" for item in items),
                average_score=sum(scores) / len(scores) if scores else None,
                average_tokens=(
                    sum(token_values) / len(token_values) if token_values else None
                ),
                average_duration_ms=(
                    sum(durations) / len(durations) if durations else None
                ),
                total_cost_cents=sum(costs) if costs else None,
                average_citation_recall=(
                    sum(recalls) / len(recalls) if recalls else None
                ),
                average_citation_precision=(
                    sum(precisions) / len(precisions) if precisions else None
                ),
            )
        )
    return BenchmarkRunSnapshot(
        id=run.id,
        label=run.label,
        status=run.status,
        judge_provider=run.judge_provider,
        judge_provider_id=run.judge_provider_id,
        judge_model=run.judge_model,
        deep_research=run.deep_research,
        total_items=run.total_items,
        completed_items=run.completed_items,
        failed_items=run.failed_items,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
        report=run.report,
        report_error=run.report_error,
        report_input_tokens=run.report_input_tokens,
        report_output_tokens=run.report_output_tokens,
        report_cost_cents=run.report_cost_cents,
        items=[BenchmarkRunItemSnapshot.from_model(item) for item in run.items],
        aggregates=aggregates,
    )
