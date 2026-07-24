-- Up Migration

-- Correlates each LLM call with the agent's step counter
-- (FsExplorerAgent._step_count at call time), so per-call token growth can
-- be attributed to a specific research step instead of only appearing as
-- an end-of-run aggregate.
ALTER TABLE llm_calls ADD COLUMN step INTEGER;

-- Down Migration

ALTER TABLE llm_calls DROP COLUMN IF EXISTS step;
