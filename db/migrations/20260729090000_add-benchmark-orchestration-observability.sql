-- Up Migration

-- Existing benchmark runs flatten one candidate provider/model across every
-- orchestration role. Production-profile runs keep their heterogeneous
-- planner/task/worker/final models and are identified explicitly here.
ALTER TABLE benchmark_runs
  ADD COLUMN IF NOT EXISTS profile_mode TEXT NOT NULL DEFAULT 'candidate_all_roles';

DO $migration$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'benchmark_runs_profile_mode_check'
      AND conrelid = 'benchmark_runs'::regclass
  ) THEN
    ALTER TABLE benchmark_runs
      ADD CONSTRAINT benchmark_runs_profile_mode_check
      CHECK (profile_mode IN ('candidate_all_roles', 'production_roles'));
  END IF;
END
$migration$;

-- JSONB keeps the trace contract evolvable while each payload carries its own
-- schema version. Both columns are nullable so historical rows and temporarily
-- older core-api deployments remain readable during a rolling deployment.
ALTER TABLE benchmark_run_items
  ADD COLUMN IF NOT EXISTS plan_trace JSONB;

ALTER TABLE benchmark_run_items
  ADD COLUMN IF NOT EXISTS role_usage JSONB;

-- Down Migration

ALTER TABLE benchmark_run_items
  DROP COLUMN IF EXISTS role_usage;

ALTER TABLE benchmark_run_items
  DROP COLUMN IF EXISTS plan_trace;

ALTER TABLE benchmark_runs
  DROP CONSTRAINT IF EXISTS benchmark_runs_profile_mode_check;

ALTER TABLE benchmark_runs
  DROP COLUMN IF EXISTS profile_mode;
