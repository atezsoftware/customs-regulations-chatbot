-- Up Migration

-- Per-step chunk/token retrieval breakdown (one entry per chunk-bearing
-- tool call: semantic_search, get_document, parse_file, read,
-- get_chunk_context), distinct from the existing tool_result_chars column
-- (which sums every tool call's output, not just retrieval ones). Mirrors
-- step_path's JSONB shape so a future per-item drill-down UI can show
-- "step N: tool_name — X chunks, ~Y tokens" alongside the existing step
-- trace.
ALTER TABLE benchmark_run_items ADD COLUMN retrieval_steps JSONB;

-- Down Migration

ALTER TABLE benchmark_run_items DROP COLUMN IF EXISTS retrieval_steps;
