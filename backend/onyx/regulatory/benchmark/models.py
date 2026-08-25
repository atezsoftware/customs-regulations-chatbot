from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BenchmarkExpectedCitationInput(BaseModel):
    chunk_id: str = Field(min_length=1)
    requirement: Literal["required", "supporting"] = "required"
    notes: str | None = Field(default=None, max_length=2000)


class BenchmarkExpectedCitation(BenchmarkExpectedCitationInput):
    file_name: str
    heading_path: list[str]
    text_excerpt: str


class BenchmarkFactAssessment(BaseModel):
    fact: str
    verdict: Literal["met", "partial", "missing", "contradicted", "unverifiable"]
    explanation: str


class BenchmarkCitationAssessment(BaseModel):
    expected_chunk_id: str
    verdict: Literal[
        "cited", "supported_elsewhere", "missing", "incorrect", "unverifiable"
    ]
    explanation: str


class BenchmarkCriterionReport(BaseModel):
    score: int
    rationale: str = Field(min_length=1)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("score must be between 1 and 5")
        return value


class BenchmarkCriteriaReport(BaseModel):
    correctness: BenchmarkCriterionReport
    groundedness: BenchmarkCriterionReport
    completeness: BenchmarkCriterionReport
    clarity: BenchmarkCriterionReport


class BenchmarkJudgeResult(BaseModel):
    correctness_score: int
    groundedness_score: int
    completeness_score: int
    clarity_score: int
    overall_score: int = Field(
        description="Overall benchmark score on a 0-100 scale, never a 1-5 scale"
    )
    rationale: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    criteria: BenchmarkCriteriaReport
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    fact_assessments: list[BenchmarkFactAssessment] = Field(default_factory=list)
    citation_assessments: list[BenchmarkCitationAssessment] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def derive_duplicate_scores_from_criteria(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        criteria = normalized.get("criteria")
        if not isinstance(criteria, dict):
            return normalized

        score_fields = {
            "correctness_score": "correctness",
            "groundedness_score": "groundedness",
            "completeness_score": "completeness",
            "clarity_score": "clarity",
        }
        for score_field, criterion_name in score_fields.items():
            criterion = criteria.get(criterion_name)
            if score_field not in normalized and isinstance(criterion, dict):
                criterion_score = criterion.get("score")
                if criterion_score is not None:
                    normalized[score_field] = criterion_score

        if "overall_score" not in normalized:
            criterion_scores = [normalized.get(field) for field in score_fields]
            if all(isinstance(score, int) for score in criterion_scores):
                # Four 1-5 criteria map linearly to the benchmark's 0-100 scale.
                normalized["overall_score"] = sum(criterion_scores) * 5
        return normalized

    @field_validator(
        "correctness_score",
        "groundedness_score",
        "completeness_score",
        "clarity_score",
    )
    @classmethod
    def validate_criterion_score(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("criterion score must be between 1 and 5")
        return value

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, value: int) -> int:
        if 0 <= value <= 5:
            return value * 20
        if not 0 <= value <= 100:
            raise ValueError("overall score must be between 0 and 100")
        return value


class BenchmarkModelReport(BaseModel):
    provider: str
    provider_id: int | None
    model_id: str
    rank: int
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommended_use: str

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, value: int) -> int:
        if value < 1:
            raise ValueError("rank must be positive")
        return value


class BenchmarkRunReport(BaseModel):
    executive_summary: str
    model_reports: list[BenchmarkModelReport]
    common_failure_patterns: list[str] = Field(default_factory=list)
    recommendation: str
