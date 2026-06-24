-- Up Migration

WITH legacy_corpora AS (
  SELECT
    id,
    '/__customs_regulations__/directories/' ||
      substring(root_path FROM '/_indexes/([0-9]+)$') AS virtual_root
  FROM core_corpora
  WHERE root_path ~ '/_indexes/[0-9]+$'
),
updatable AS (
  SELECT legacy_corpora.id, legacy_corpora.virtual_root
  FROM legacy_corpora
  WHERE legacy_corpora.virtual_root IS NOT NULL
    AND NOT EXISTS (
      SELECT 1
      FROM core_corpora existing
      WHERE existing.root_path = legacy_corpora.virtual_root
    )
)
UPDATE core_corpora corpus
SET root_path = updatable.virtual_root
FROM updatable
WHERE corpus.id = updatable.id;

-- Down Migration

-- Irreversible data migration: old machine-local storage prefixes are not
-- reconstructable once replaced by portable virtual corpus keys.
SELECT 1;
