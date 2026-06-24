-- Up Migration

CREATE TABLE chat_sessions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users ON DELETE CASCADE,
  title TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chat_sessions_user_id_index ON chat_sessions (user_id);

-- Many-to-many: which directories a chat session is allowed to see. A
-- session must only ever surface files from directories listed here.
CREATE TABLE chat_session_directories (
  id SERIAL PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES chat_sessions ON DELETE CASCADE,
  directory_id INTEGER NOT NULL REFERENCES directories ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT chat_session_directories_unique UNIQUE (session_id, directory_id)
);
CREATE INDEX chat_session_directories_directory_id_index ON chat_session_directories (directory_id);

-- Down Migration

DROP TABLE chat_session_directories;
DROP TABLE chat_sessions;
