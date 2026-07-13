-- Up Migration

-- Tracks how full the model's context window was on the most recent
-- completed turn of this session (fraction 0-1), so the chat UI can show
-- a per-chat "context limit fill" indicator without recomputing it from
-- llm_calls on every page load. Total token usage per session is still
-- computed on demand from llm_calls (see LlmCallRepository) rather than
-- duplicated here, since it's a simple aggregate.
ALTER TABLE chat_sessions
  ADD COLUMN last_context_usage_ratio REAL;

-- Down Migration

ALTER TABLE chat_sessions
  DROP COLUMN last_context_usage_ratio;
