export interface BenchmarkQuestion {
  id: number;
  title: string;
  prompt: string;
  reference_answer: string | null;
  expected_facts: string[];
  expected_citations: BenchmarkExpectedCitation[];
  as_of_date: string | null;
  rubric_notes: string | null;
  tags: string[];
  document_set_id: number;
  document_set_name: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface BenchmarkModelSelection {
  provider: string;
  provider_id: number | null;
  model_id: string;
}

export interface BenchmarkAvailableModel extends BenchmarkModelSelection {
  provider_id: number;
  display_name: string;
  max_input_tokens: number | null;
  is_visible: boolean;
}

export interface BenchmarkExpectedCitationInput {
  chunk_id: string;
  requirement: "required" | "supporting";
  notes: string | null;
}

export interface BenchmarkExpectedCitation extends BenchmarkExpectedCitationInput {
  file_name: string;
  heading_path: string[];
  text_excerpt: string;
}

export interface BenchmarkCitationOption {
  chunk_id: string;
  user_file_id: string;
  file_name: string;
  heading_path: string[];
  text_excerpt: string;
  status: "active" | "superseded";
  validity_start_date: string | null;
  validity_end_date: string | null;
}

export interface BenchmarkJudgment {
  correctness_score: number;
  groundedness_score: number;
  completeness_score: number;
  clarity_score: number;
  overall_score: number;
  rationale: string;
  report: {
    summary?: string;
    criteria?: Record<string, { score: number; rationale: string }>;
    strengths?: string[];
    weaknesses?: string[];
    fact_assessments?: Array<{
      fact: string;
      verdict: string;
      explanation: string;
    }>;
    citation_assessments?: Array<{
      expected_chunk_id: string;
      verdict: string;
      explanation: string;
    }>;
  };
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  cost_cents: number | null;
  cost_source: "measured" | "unavailable";
}

export interface BenchmarkRunItem {
  id: number;
  provider: string;
  provider_id: number | null;
  model_id: string;
  question_id: number;
  question_prompt: string;
  question_title: string;
  question_snapshot: Record<string, unknown>;
  status: string;
  execution_phase:
    | "starting"
    | "preparing_session"
    | "answering"
    | "researching"
    | "judging"
    | null;
  heartbeat_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  final_result: string | null;
  error_message: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  duration_ms: number | null;
  cost_cents: number | null;
  cost_source: "measured" | "unavailable";
  cited_chunk_ids: string[];
  cited_sources: Array<Record<string, unknown>>;
  execution_steps: Array<Record<string, unknown>>;
  llm_calls: Array<Record<string, unknown>>;
  answer_reasoning: string | null;
  chat_session_id: string | null;
  assistant_message_id: number | null;
  citation_recall: number | null;
  citation_precision: number | null;
  judge_error: string | null;
  judgment: BenchmarkJudgment | null;
}

export interface BenchmarkModelAggregate {
  provider: string;
  provider_id: number | null;
  model_id: string;
  item_count: number;
  completed_count: number;
  failed_count: number;
  average_score: number | null;
  average_tokens: number | null;
  average_duration_ms: number | null;
  total_cost_cents: number | null;
  average_citation_recall: number | null;
  average_citation_precision: number | null;
}

export interface BenchmarkRunReport {
  executive_summary: string;
  model_reports: Array<{
    provider: string;
    provider_id: number | null;
    model_id: string;
    rank: number;
    summary: string;
    strengths: string[];
    weaknesses: string[];
    recommended_use: string;
  }>;
  common_failure_patterns: string[];
  recommendation: string;
}

export interface BenchmarkRun {
  id: number;
  label: string | null;
  status:
    | "pending"
    | "queued"
    | "running"
    | "completed"
    | "error"
    | "cancelled";
  judge_provider: string;
  judge_provider_id: number | null;
  judge_model: string;
  deep_research: boolean;
  total_items: number;
  completed_items: number;
  failed_items: number;
  queued_at: string | null;
  started_at: string | null;
  heartbeat_at: string | null;
  completed_at: string | null;
  created_at: string;
  failure_code:
    | "dependency_unavailable"
    | "worker_unavailable"
    | "dispatch_failed"
    | "execution_timeout"
    | "execution_failed"
    | null;
  failure_message: string | null;
  report: BenchmarkRunReport | null;
  report_error: string | null;
  report_input_tokens: number | null;
  report_output_tokens: number | null;
  report_cost_cents: number | null;
  items: BenchmarkRunItem[];
  aggregates: BenchmarkModelAggregate[];
}

export interface BenchmarkQuestionInput {
  title: string;
  prompt: string;
  reference_answer: string | null;
  expected_facts: string[];
  expected_citations: BenchmarkExpectedCitationInput[];
  as_of_date: string | null;
  rubric_notes: string | null;
  tags: string[];
  document_set_id: number;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

const jsonInit = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const listBenchmarkQuestions = () =>
  requestJson<BenchmarkQuestion[]>("/api/regulatory/benchmark/questions");

export const createBenchmarkQuestion = (input: BenchmarkQuestionInput) =>
  requestJson<BenchmarkQuestion>(
    "/api/regulatory/benchmark/questions",
    jsonInit("POST", input)
  );

export const updateBenchmarkQuestion = (
  id: number,
  input: Partial<BenchmarkQuestionInput> & { is_active?: boolean }
) =>
  requestJson<BenchmarkQuestion>(
    `/api/regulatory/benchmark/questions/${id}`,
    jsonInit("PATCH", input)
  );

export async function deleteBenchmarkQuestion(id: number): Promise<void> {
  const response = await fetch(`/api/regulatory/benchmark/questions/${id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Delete failed (${response.status})`);
  }
}

export const listBenchmarkRuns = () =>
  requestJson<BenchmarkRun[]>("/api/regulatory/benchmark/runs");

export const createBenchmarkRun = (input: {
  label: string | null;
  question_ids: number[] | null;
  candidates: BenchmarkModelSelection[];
  judge: BenchmarkModelSelection;
  deep_research: boolean;
}) =>
  requestJson<BenchmarkRun>(
    "/api/regulatory/benchmark/runs",
    jsonInit("POST", input)
  );

export const startBenchmarkRun = (id: number) =>
  requestJson<BenchmarkRun>(`/api/regulatory/benchmark/runs/${id}/start`, {
    method: "POST",
  });

export const cancelBenchmarkRun = (id: number) =>
  requestJson<BenchmarkRun>(`/api/regulatory/benchmark/runs/${id}/cancel`, {
    method: "POST",
  });

export const getBenchmarkRun = (id: number) =>
  requestJson<BenchmarkRun>(`/api/regulatory/benchmark/runs/${id}`);

export const listBenchmarkModels = () =>
  requestJson<BenchmarkAvailableModel[]>("/api/regulatory/benchmark/models");

export const listBenchmarkCitationOptions = (documentSetId: number) =>
  requestJson<BenchmarkCitationOption[]>(
    `/api/regulatory/benchmark/document-sets/${documentSetId}/citation-options`
  );
