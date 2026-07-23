-- Up Migration

CREATE TABLE benchmark_questions (
  id SERIAL PRIMARY KEY,
  prompt TEXT NOT NULL,
  reference_answer TEXT,
  expected_facts JSONB NOT NULL DEFAULT '[]'::jsonb,
  rubric_notes TEXT,
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  is_active BOOLEAN NOT NULL DEFAULT true,
  created_by INTEGER REFERENCES users ON DELETE SET NULL,
  updated_by INTEGER REFERENCES users ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX benchmark_questions_active_index ON benchmark_questions (is_active);

-- Many-to-many: which indexed directories a benchmark question searches
-- against. Mirrors chat_session_directories exactly, so a question resolves
-- to the same virtual-corpus folders a real chat session would.
CREATE TABLE benchmark_question_directories (
  id SERIAL PRIMARY KEY,
  question_id INTEGER NOT NULL REFERENCES benchmark_questions ON DELETE CASCADE,
  directory_id INTEGER NOT NULL REFERENCES directories ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT benchmark_question_directories_unique UNIQUE (question_id, directory_id)
);
CREATE INDEX benchmark_question_directories_directory_id_index
  ON benchmark_question_directories (directory_id);

CREATE TABLE benchmark_runs (
  id SERIAL PRIMARY KEY,
  label TEXT,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'completed', 'error', 'cancelled')),
  judge_provider TEXT NOT NULL,
  judge_model TEXT NOT NULL,
  created_by INTEGER REFERENCES users ON DELETE SET NULL,
  total_items INTEGER NOT NULL DEFAULT 0,
  completed_items INTEGER NOT NULL DEFAULT 0,
  failed_items INTEGER NOT NULL DEFAULT 0,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX benchmark_runs_status_index ON benchmark_runs (status);

-- One row per (run, candidate model, question, repeat). repeat_index
-- defaults to 1 and is not exposed in the first-release UI, but keeps
-- future pass@k / multi-sample support from needing another migration.
CREATE TABLE benchmark_run_items (
  id BIGSERIAL PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES benchmark_runs ON DELETE CASCADE,
  provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  -- No ON DELETE CASCADE here (unlike the join table above): a question
  -- with historical run items must not be hard-deletable, so results stay
  -- readable. The application only allows deleting questions with zero
  -- run items and otherwise offers deactivation (is_active) instead.
  question_id INTEGER NOT NULL REFERENCES benchmark_questions,
  repeat_index INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'completed', 'error')),
  final_result TEXT,
  error_message TEXT,
  -- True when the run ended via the step-budget safety net rather than a
  -- real stop action (mirrors core-api's "complete" event `incomplete`
  -- flag) — no error, but the answer may be an unsatisfying apology rather
  -- than a real conclusion, so it's excluded from a strict success rate.
  incomplete BOOLEAN NOT NULL DEFAULT false,
  steps INTEGER,
  api_calls INTEGER,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  thinking_tokens INTEGER,
  total_tokens INTEGER,
  tool_result_chars INTEGER,
  context_summaries INTEGER,
  duration_ms INTEGER,
  cost_usd NUMERIC(20, 10),
  cost_source TEXT CHECK (cost_source IN ('provider', 'estimated')),
  cited_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
  step_path JSONB NOT NULL DEFAULT '[]'::jsonb,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  CONSTRAINT benchmark_run_items_unique
    UNIQUE (run_id, provider, model_id, question_id, repeat_index)
);
CREATE INDEX benchmark_run_items_run_id_index ON benchmark_run_items (run_id);
CREATE INDEX benchmark_run_items_status_index ON benchmark_run_items (status);
CREATE INDEX benchmark_run_items_question_id_index ON benchmark_run_items (question_id);

-- Separate from benchmark_run_items because the judge model is selected
-- independently and judgments may later be recomputed with a different
-- judge without re-running the benchmark itself.
CREATE TABLE benchmark_run_judgments (
  id BIGSERIAL PRIMARY KEY,
  run_item_id BIGINT NOT NULL UNIQUE REFERENCES benchmark_run_items ON DELETE CASCADE,
  judge_provider TEXT NOT NULL,
  judge_model TEXT NOT NULL,
  correctness_score SMALLINT NOT NULL CHECK (correctness_score BETWEEN 1 AND 5),
  groundedness_score SMALLINT NOT NULL CHECK (groundedness_score BETWEEN 1 AND 5),
  completeness_score SMALLINT NOT NULL CHECK (completeness_score BETWEEN 1 AND 5),
  clarity_score SMALLINT NOT NULL CHECK (clarity_score BETWEEN 1 AND 5),
  overall_score SMALLINT NOT NULL CHECK (overall_score BETWEEN 0 AND 100),
  rationale TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Down Migration

DROP TABLE IF EXISTS benchmark_run_judgments;
DROP INDEX IF EXISTS benchmark_run_items_question_id_index;
DROP INDEX IF EXISTS benchmark_run_items_status_index;
DROP INDEX IF EXISTS benchmark_run_items_run_id_index;
DROP TABLE IF EXISTS benchmark_run_items;
DROP INDEX IF EXISTS benchmark_runs_status_index;
DROP TABLE IF EXISTS benchmark_runs;
DROP INDEX IF EXISTS benchmark_question_directories_directory_id_index;
DROP TABLE IF EXISTS benchmark_question_directories;
DROP INDEX IF EXISTS benchmark_questions_active_index;
DROP TABLE IF EXISTS benchmark_questions;
