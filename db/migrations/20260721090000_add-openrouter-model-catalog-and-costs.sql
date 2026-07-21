-- Up Migration

CREATE TABLE llm_models (
  provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  canonical_slug TEXT,
  display_name TEXT NOT NULL,
  description TEXT,
  context_length INTEGER NOT NULL,
  max_completion_tokens INTEGER,
  input_modalities JSONB NOT NULL DEFAULT '[]'::jsonb,
  output_modalities JSONB NOT NULL DEFAULT '[]'::jsonb,
  supported_parameters JSONB NOT NULL DEFAULT '[]'::jsonb,
  architecture JSONB NOT NULL DEFAULT '{}'::jsonb,
  reasoning_config JSONB,
  raw_pricing JSONB NOT NULL DEFAULT '{}'::jsonb,
  prompt_usd_per_token NUMERIC(24, 18),
  completion_usd_per_token NUMERIC(24, 18),
  request_usd NUMERIC(24, 18),
  cache_read_usd_per_token NUMERIC(24, 18),
  cache_write_usd_per_token NUMERIC(24, 18),
  pricing_hash TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  expires_at TIMESTAMPTZ,
  last_synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (provider, model_id)
);

CREATE TABLE llm_model_price_snapshots (
  id BIGSERIAL PRIMARY KEY,
  provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  pricing_hash TEXT NOT NULL,
  raw_pricing JSONB NOT NULL,
  prompt_usd_per_token NUMERIC(24, 18),
  completion_usd_per_token NUMERIC(24, 18),
  request_usd NUMERIC(24, 18),
  cache_read_usd_per_token NUMERIC(24, 18),
  cache_write_usd_per_token NUMERIC(24, 18),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, model_id, pricing_hash)
);

CREATE TABLE llm_model_sync_runs (
  id BIGSERIAL PRIMARY KEY,
  provider TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'skipped')),
  received_count INTEGER NOT NULL DEFAULT 0,
  active_count INTEGER NOT NULL DEFAULT 0,
  skipped_count INTEGER NOT NULL DEFAULT 0,
  error_message TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ
);

ALTER TABLE chat_sessions ADD COLUMN llm_provider TEXT;
UPDATE chat_sessions SET llm_provider = 'gemini' WHERE llm_provider IS NULL;
ALTER TABLE chat_sessions ALTER COLUMN llm_provider SET DEFAULT 'openrouter';
ALTER TABLE chat_sessions ALTER COLUMN llm_provider SET NOT NULL;

ALTER TABLE llm_calls
  ADD COLUMN generation_id TEXT,
  ADD COLUMN cached_input_tokens INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN billed_cost_usd NUMERIC(24, 12),
  ADD COLUMN upstream_cost_usd NUMERIC(24, 12),
  ADD COLUMN cost_source TEXT CHECK (cost_source IN ('provider', 'estimated')),
  ADD COLUMN price_snapshot_id BIGINT REFERENCES llm_model_price_snapshots(id) ON DELETE SET NULL;

CREATE INDEX llm_models_active_index ON llm_models (provider, is_active, display_name);
CREATE INDEX llm_model_sync_runs_provider_index ON llm_model_sync_runs (provider, started_at DESC);
CREATE INDEX llm_calls_cost_source_index ON llm_calls (cost_source);

-- Down Migration
DROP INDEX IF EXISTS llm_calls_cost_source_index;
DROP INDEX IF EXISTS llm_model_sync_runs_provider_index;
DROP INDEX IF EXISTS llm_models_active_index;
ALTER TABLE llm_calls
  DROP COLUMN IF EXISTS price_snapshot_id,
  DROP COLUMN IF EXISTS cost_source,
  DROP COLUMN IF EXISTS upstream_cost_usd,
  DROP COLUMN IF EXISTS billed_cost_usd,
  DROP COLUMN IF EXISTS cache_write_tokens,
  DROP COLUMN IF EXISTS cached_input_tokens,
  DROP COLUMN IF EXISTS generation_id;
ALTER TABLE chat_sessions DROP COLUMN IF EXISTS llm_provider;
DROP TABLE IF EXISTS llm_model_sync_runs;
DROP TABLE IF EXISTS llm_model_price_snapshots;
DROP TABLE IF EXISTS llm_models;
