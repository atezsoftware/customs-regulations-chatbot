export interface UsageExportDateRange {
  from: Date;
  to: Date;
}

export interface UsageExportTotals {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cost_cents: number;
  total_tokens: number;
  total_user_queries: number;
  total_user_sessions: number;
  average_tokens_per_query: number;
  average_tokens_per_session: number;
  average_cost_cents_per_query: number;
  average_cost_cents_per_session: number;
  average_queries_per_session: number;
}

export type WorkflowUsageKey = "asv3" | "asv3_exp_cropped";

export interface WorkflowUsage {
  query_count: number;
  session_count: number;
  cost_cents: number | null;
  known_cost_cents: number;
  average_cost_cents_per_query: number | null;
  average_cost_cents_per_session: number | null;
  cost_status: "complete" | "partial" | "unavailable" | "running";
  unpriced_call_count: number;
}

export interface UsageExportUser {
  email: string;
  totals: UsageExportTotals;
  workflow_usage?: Partial<Record<WorkflowUsageKey, WorkflowUsage>>;
}

export interface UsageExportResponse {
  start: string;
  end: string;
  users: UsageExportUser[];
}
