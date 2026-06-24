-- Up Migration

ALTER TABLE chat_sessions
  ADD COLUMN model TEXT,
  ADD COLUMN temperature REAL;

CREATE TABLE chat_messages (
  id SERIAL PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES chat_sessions ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'streaming', 'completed', 'error', 'cancelled')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chat_messages_session_id_index ON chat_messages (session_id);
CREATE INDEX chat_messages_created_at_index ON chat_messages (created_at);

CREATE TABLE chat_research_steps (
  id SERIAL PRIMARY KEY,
  message_id INTEGER NOT NULL REFERENCES chat_messages ON DELETE CASCADE,
  step_key TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'error')),
  title TEXT NOT NULL,
  preview TEXT,
  details TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ
);
CREATE INDEX chat_research_steps_message_id_index ON chat_research_steps (message_id);
CREATE UNIQUE INDEX chat_research_steps_message_step_key_unique
  ON chat_research_steps (message_id, step_key);

CREATE TABLE chat_sources (
  id SERIAL PRIMARY KEY,
  message_id INTEGER NOT NULL REFERENCES chat_messages ON DELETE CASCADE,
  title TEXT NOT NULL,
  snippet TEXT,
  url TEXT,
  file_path TEXT,
  page INTEGER,
  chunk_id TEXT REFERENCES core_chunks(id) ON DELETE SET NULL,
  score REAL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chat_sources_message_id_index ON chat_sources (message_id);
CREATE INDEX chat_sources_chunk_id_index ON chat_sources (chunk_id);

CREATE TABLE llm_calls (
  id SERIAL PRIMARY KEY,
  message_id INTEGER REFERENCES chat_messages ON DELETE SET NULL,
  session_id INTEGER NOT NULL REFERENCES chat_sessions ON DELETE CASCADE,
  provider TEXT NOT NULL,
  model TEXT,
  purpose TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  thinking_tokens INTEGER NOT NULL DEFAULT 0,
  duration_ms INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX llm_calls_session_id_index ON llm_calls (session_id);
CREATE INDEX llm_calls_message_id_index ON llm_calls (message_id);

-- Down Migration

DROP TABLE llm_calls;
DROP TABLE chat_sources;
DROP TABLE chat_research_steps;
DROP TABLE chat_messages;

ALTER TABLE chat_sessions
  DROP COLUMN temperature,
  DROP COLUMN model;
