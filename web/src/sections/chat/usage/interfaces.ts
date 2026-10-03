export interface TokenCostLine {
  category: string;
  tokens: number;
  usd_per_million: number | null;
  cost_usd: number | null;
}
export interface GenerationCost {
  model: string;
  provider: string | null;
  source: string;
  priced_at: string;
  input_tokens: number | null;
  output_tokens: number | null;
  reasoning_tokens: number | null;
  lines: TokenCostLine[];
  complete: boolean;
  known_cost_usd: number;
}
export interface ResponseUsage {
  duration_seconds: number | null;
  status: "complete" | "partial" | "unavailable" | "running";
  total_cost_usd: number | null;
  known_cost_usd: number | null;
  currency: "USD";
  calls: number;
  unpriced_calls: number;
  models: GenerationCost[];
}
