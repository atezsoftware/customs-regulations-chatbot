-- Up Migration

-- Judge scoring is best-effort: a candidate run that itself succeeded still
-- completes even if the judge call fails (rate limit, judge model outage,
-- schema validation). Previously that failure was only logged to the
-- backend console, with nothing visible in the UI/DB — this column makes
-- it diagnosable per item without needing server log access.
ALTER TABLE benchmark_run_items ADD COLUMN judge_error TEXT;

-- Down Migration

ALTER TABLE benchmark_run_items DROP COLUMN IF EXISTS judge_error;
