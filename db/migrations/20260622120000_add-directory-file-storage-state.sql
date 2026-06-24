-- Up Migration

ALTER TABLE directory_files
  ADD COLUMN storage_status TEXT NOT NULL DEFAULT 'stored'
    CHECK (storage_status IN ('stored', 'indexed', 'error')),
  ADD COLUMN indexed_at TIMESTAMPTZ,
  ADD COLUMN raw_deleted_at TIMESTAMPTZ,
  ADD COLUMN storage_error TEXT;

CREATE INDEX directory_files_storage_status_index
  ON directory_files (directory_id, storage_status);

-- Down Migration

DROP INDEX directory_files_storage_status_index;

ALTER TABLE directory_files
  DROP COLUMN storage_error,
  DROP COLUMN raw_deleted_at,
  DROP COLUMN indexed_at,
  DROP COLUMN storage_status;
