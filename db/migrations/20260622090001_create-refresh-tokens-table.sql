-- Up Migration

CREATE TABLE refresh_tokens (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users ON DELETE CASCADE,
  token_hash TEXT NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  replaced_by_token_hash TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX refresh_tokens_user_id_index ON refresh_tokens (user_id);
CREATE UNIQUE INDEX refresh_tokens_token_hash_unique_index ON refresh_tokens (token_hash);

-- Down Migration

DROP TABLE refresh_tokens;
