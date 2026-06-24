-- Up Migration

ALTER TABLE directory_files
  DROP CONSTRAINT directory_files_storage_status_check,
  ADD CONSTRAINT directory_files_storage_status_check
    CHECK (storage_status IN ('stored', 'chunked', 'indexed', 'error')),
  ADD COLUMN chunked_at TIMESTAMPTZ;

-- Down Migration

ALTER TABLE directory_files
  DROP COLUMN chunked_at,
  DROP CONSTRAINT directory_files_storage_status_check,
  ADD CONSTRAINT directory_files_storage_status_check
    CHECK (storage_status IN ('stored', 'indexed', 'error'));
