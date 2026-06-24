-- Up Migration

CREATE TABLE directories (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users ON DELETE CASCADE,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX directories_user_id_index ON directories (user_id);

CREATE TABLE directory_files (
  id SERIAL PRIMARY KEY,
  directory_id INTEGER NOT NULL REFERENCES directories ON DELETE CASCADE,
  original_name TEXT NOT NULL,
  stored_path TEXT NOT NULL,
  mime_type TEXT,
  size_bytes BIGINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX directory_files_directory_id_index ON directory_files (directory_id);

-- Down Migration

DROP TABLE directory_files;
DROP TABLE directories;
