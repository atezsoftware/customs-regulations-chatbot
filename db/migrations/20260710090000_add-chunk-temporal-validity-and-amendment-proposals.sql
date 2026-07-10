-- Up Migration

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Amendment-created chunks (see core_amendment_proposals below) have no real
-- parse offsets in a source document, so these can no longer be universally
-- required. Indexer-produced chunks still must have them (enforced below via
-- core_chunks_indexed_offsets_check), so make_chunk_id's determinism is
-- untouched for the indexing path.
ALTER TABLE core_chunks
  ALTER COLUMN start_char DROP NOT NULL,
  ALTER COLUMN end_char DROP NOT NULL;

ALTER TABLE core_chunks
  ADD COLUMN source TEXT NOT NULL DEFAULT 'indexed'
    CHECK (source IN ('indexed', 'amendment')),
  ADD COLUMN status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'superseded', 'expired')),
  ADD COLUMN effective_start_date DATE,
  ADD COLUMN effective_end_date DATE,
  ADD COLUMN supersedes_chunk_id TEXT REFERENCES core_chunks(id) ON DELETE SET NULL,
  ADD COLUMN superseded_by_chunk_id TEXT REFERENCES core_chunks(id) ON DELETE SET NULL;

ALTER TABLE core_chunks
  ADD CONSTRAINT core_chunks_indexed_offsets_check
  CHECK (source <> 'indexed' OR (start_char IS NOT NULL AND end_char IS NOT NULL));

CREATE INDEX core_chunks_status_index ON core_chunks (status);
CREATE INDEX core_chunks_document_id_status_index ON core_chunks (document_id, status);

-- Turkish-aware fuzzy matching for the amendment hybrid candidate finder.
-- Trigram similarity is character-n-gram based, so it works on Turkish text
-- without needing a language-specific tokenizer/dictionary.
CREATE INDEX core_chunks_text_trgm_idx ON core_chunks USING gin (text gin_trgm_ops);

CREATE OR REPLACE FUNCTION core_chunk_heading_path_text(metadata jsonb) RETURNS text AS $$
  SELECT COALESCE(
    array_to_string(ARRAY(SELECT jsonb_array_elements_text(metadata->'heading_path')), ' > '),
    ''
  );
$$ LANGUAGE sql IMMUTABLE;

CREATE INDEX core_chunks_heading_path_trgm_idx ON core_chunks
  USING gin (core_chunk_heading_path_text(metadata) gin_trgm_ops);

-- Short structured locators (exact/prefix match candidates), separate from
-- the fuzzy trigram indexes above which exist for noisy/OCR'd text instead.
CREATE INDEX core_chunks_article_no_idx ON core_chunks ((metadata->>'article_no'));
CREATE INDEX core_chunks_document_number_idx ON core_chunks ((metadata->>'document_number'));

-- One admin paste of Official Gazette amendment text can contain several
-- atomic per-article changes; core_amendment_batches is the audit record for
-- the paste itself, core_amendment_proposals is the durable, individually
-- approvable/rejectable unit derived from it. Nothing here writes to
-- core_chunks until a proposal is approved.
CREATE TABLE core_amendment_batches (
  id TEXT PRIMARY KEY,
  corpus_id TEXT NOT NULL REFERENCES core_corpora(id),
  raw_text TEXT NOT NULL,
  reference_date DATE,
  status TEXT NOT NULL DEFAULT 'analyzing'
    CHECK (status IN ('analyzing', 'analyzed', 'failed')),
  error_message TEXT,
  created_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX core_amendment_batches_corpus_id_index ON core_amendment_batches (corpus_id);

-- old_chunk_snapshot/new_chunk_draft are frozen JSONB copies, not live joins,
-- so what an admin reviewed and approved can't drift if core_chunks changes
-- between analysis and approval, and stays inspectable after old_chunk_id
-- itself is later marked superseded.
CREATE TABLE core_amendment_proposals (
  id TEXT PRIMARY KEY,
  batch_id TEXT NOT NULL REFERENCES core_amendment_batches(id) ON DELETE CASCADE,
  instruction_index INTEGER NOT NULL,
  instruction_text TEXT NOT NULL,
  old_chunk_id TEXT REFERENCES core_chunks(id) ON DELETE SET NULL,
  old_chunk_snapshot JSONB NOT NULL,
  new_chunk_draft JSONB NOT NULL,
  match_confidence DOUBLE PRECISION,
  match_rationale TEXT,
  date_rationale TEXT,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'approved', 'rejected')),
  applied_new_chunk_id TEXT REFERENCES core_chunks(id) ON DELETE SET NULL,
  decided_by TEXT,
  decided_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX core_amendment_proposals_batch_id_index ON core_amendment_proposals (batch_id);
CREATE INDEX core_amendment_proposals_status_index ON core_amendment_proposals (status);
CREATE INDEX core_amendment_proposals_old_chunk_id_index ON core_amendment_proposals (old_chunk_id);

-- Down Migration

DROP TABLE core_amendment_proposals;
DROP TABLE core_amendment_batches;

DROP INDEX core_chunks_document_number_idx;
DROP INDEX core_chunks_article_no_idx;
DROP INDEX core_chunks_heading_path_trgm_idx;
DROP FUNCTION core_chunk_heading_path_text(jsonb);
DROP INDEX core_chunks_text_trgm_idx;
DROP INDEX core_chunks_document_id_status_index;
DROP INDEX core_chunks_status_index;

ALTER TABLE core_chunks
  DROP CONSTRAINT core_chunks_indexed_offsets_check,
  DROP COLUMN superseded_by_chunk_id,
  DROP COLUMN supersedes_chunk_id,
  DROP COLUMN effective_end_date,
  DROP COLUMN effective_start_date,
  DROP COLUMN status,
  DROP COLUMN source;

ALTER TABLE core_chunks
  ALTER COLUMN start_char SET NOT NULL,
  ALTER COLUMN end_char SET NOT NULL;
