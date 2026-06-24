-- Up Migration

CREATE EXTENSION IF NOT EXISTS vector;

-- Safe numeric cast for metadata filtering (Postgres has no built-in TRY_CAST).
CREATE OR REPLACE FUNCTION try_cast_double(value text) RETURNS double precision AS $$
BEGIN
  RETURN value::double precision;
EXCEPTION WHEN others THEN
  RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE TABLE core_corpora (
  id TEXT PRIMARY KEY,
  root_path TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE core_documents (
  id TEXT PRIMARY KEY,
  corpus_id TEXT NOT NULL REFERENCES core_corpora(id),
  relative_path TEXT NOT NULL,
  absolute_path TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  file_mtime DOUBLE PRECISION NOT NULL,
  file_size BIGINT NOT NULL,
  content_sha256 TEXT NOT NULL,
  last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  is_deleted BOOLEAN NOT NULL DEFAULT false,
  UNIQUE (corpus_id, relative_path)
);
CREATE INDEX core_documents_corpus_id_index ON core_documents (corpus_id);

-- RegulatoryChunker's full locator metadata (chunk_type, article_no,
-- paragraph_no, parent_path, heading_path, ...) lives in `metadata`. Which
-- chunk came from which file is structural via document_id -> core_documents
-- (relative_path/absolute_path), not convention.
CREATE TABLE core_chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES core_documents(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  position INTEGER NOT NULL,
  start_char INTEGER NOT NULL,
  end_char INTEGER NOT NULL,
  chunk_type TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX core_chunks_document_id_index ON core_chunks (document_id);

CREATE TABLE core_schemas (
  id TEXT PRIMARY KEY,
  corpus_id TEXT NOT NULL REFERENCES core_corpora(id),
  name TEXT NOT NULL,
  schema_def JSONB NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (corpus_id, name)
);

-- Dimension matches embeddings.py's gemini-embedding-001 default
-- (FS_EXPLORER_EMBEDDING_DIM); if that env var changes, this column's
-- dimension must be migrated too.
CREATE TABLE core_chunk_embeddings (
  chunk_id TEXT PRIMARY KEY REFERENCES core_chunks(id) ON DELETE CASCADE,
  corpus_id TEXT NOT NULL,
  embedding vector(768) NOT NULL
);
CREATE INDEX core_chunk_embeddings_corpus_id_index ON core_chunk_embeddings (corpus_id);
CREATE INDEX core_chunk_embeddings_hnsw_idx ON core_chunk_embeddings
  USING hnsw (embedding vector_cosine_ops);

-- Down Migration

DROP TABLE core_chunk_embeddings;
DROP TABLE core_schemas;
DROP TABLE core_chunks;
DROP TABLE core_documents;
DROP TABLE core_corpora;
DROP FUNCTION try_cast_double(text);
