import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import test from 'node:test';
import {
  buildCompletedItemPatch,
  buildErroredItemPatch,
  buildRunQuestionPayload,
  resolveBenchmarkCandidates,
  RunQuestionResult,
} from './benchmark-runner.service';

function resultWithStats(
  overrides: Partial<RunQuestionResult['stats']> = {},
): RunQuestionResult {
  return {
    final_result: 'Supported benchmark answer.',
    error: null,
    incomplete: false,
    cited_sources: ['Regulation A'],
    step_path: ['1. plan_created (global-planner)'],
    stats: {
      steps: 1,
      api_calls: 2,
      prompt_tokens: 100,
      completion_tokens: 20,
      thinking_tokens: 10,
      total_tokens: 130,
      tool_result_chars: 500,
      context_summaries: 0,
      retrieval_steps: [],
      duration_ms: 250,
      cost_usd: '0.001',
      cost_source: 'provider',
      ...overrides,
    },
  };
}

test('candidate benchmark payload preserves the legacy provider/model API shape', () => {
  const legacySelection = resolveBenchmarkCandidates(undefined, [
    {provider: 'openrouter', modelId: 'candidate/model'},
  ]);
  assert.equal(legacySelection.profileMode, 'candidate_all_roles');

  assert.deepEqual(
    buildRunQuestionPayload({
      task: 'Question',
      indexFolders: ['virtual://corpus-1'],
      databaseUrl: 'postgresql://test/test',
      profileMode: 'candidate_all_roles',
      provider: 'openrouter',
      model: 'candidate/model',
    }),
    {
      task: 'Question',
      index_folders: ['virtual://corpus-1'],
      database_url: 'postgresql://test/test',
      provider: 'openrouter',
      model: 'candidate/model',
    },
  );
});

test('production profile payload omits candidate fields for the existing optional core API', () => {
  const payload = buildRunQuestionPayload({
    task: 'Question',
    indexFolders: ['virtual://corpus-1'],
    databaseUrl: null,
    profileMode: 'production_roles',
  });

  assert.deepEqual(payload, {
    task: 'Question',
    index_folders: ['virtual://corpus-1'],
    database_url: null,
  });
  assert.equal('provider' in payload, false);
  assert.equal('model' in payload, false);
});

test('old core response remains persistable when observability fields are absent', () => {
  const patch = buildCompletedItemPatch(resultWithStats(), {
    completedAt: '2026-07-29T00:00:00.000Z',
  });

  assert.equal(patch.status, 'completed');
  assert.equal(patch.planTrace, undefined);
  assert.equal(patch.roleUsage, undefined);
});

test('plan trace and heterogeneous role usage are persisted without reshaping', () => {
  const planTrace = {
    schema_version: 1,
    profile_mode: 'production_roles',
    execution: {contract_version: '3', plan: {mode: 'decomposed'}},
  };
  const roleUsage = [
    {
      role: 'planner',
      purpose: 'global_plan',
      provider: 'openrouter',
      model: 'openai/gpt-5.6-sol',
      calls: 1,
      prompt_tokens: 100,
      completion_tokens: 20,
      thinking_tokens: 10,
      total_tokens: 130,
      cached_input_tokens: 0,
      cache_write_tokens: 0,
      duration_ms: 100,
      cost_usd: '0.001',
      cost_source: 'provider' as const,
    },
  ];

  const patch = buildCompletedItemPatch(
    resultWithStats({plan_trace: planTrace, role_usage: roleUsage}),
    {completedAt: '2026-07-29T00:00:00.000Z'},
  );

  assert.deepEqual(patch.planTrace, planTrace);
  assert.deepEqual(patch.roleUsage, roleUsage);

  const errorPatch = buildErroredItemPatch('planner failed', {
    completedAt: '2026-07-29T00:00:00.000Z',
    runResult: {
      ...resultWithStats({plan_trace: planTrace, role_usage: roleUsage}),
      error: 'planner failed',
    },
  });
  assert.equal(errorPatch.status, 'error');
  assert.deepEqual(errorPatch.planTrace, planTrace);
  assert.deepEqual(errorPatch.roleUsage, roleUsage);
});

test('observability migration is rerunnable and has an idempotent rollback', () => {
  const migrationPath = resolve(
    __dirname,
    '../../../../../db/migrations/20260729090000_add-benchmark-orchestration-observability.sql',
  );
  const migration = readFileSync(migrationPath, 'utf8');
  const [up, down] = migration.split('-- Down Migration');

  assert.match(up, /ADD COLUMN IF NOT EXISTS profile_mode/);
  assert.match(up, /ADD COLUMN IF NOT EXISTS plan_trace/);
  assert.match(up, /ADD COLUMN IF NOT EXISTS role_usage/);
  assert.match(up, /IF NOT EXISTS[\s\S]+benchmark_runs_profile_mode_check/);
  assert.match(down, /DROP COLUMN IF EXISTS role_usage/);
  assert.match(down, /DROP COLUMN IF EXISTS plan_trace/);
  assert.match(down, /DROP CONSTRAINT IF EXISTS benchmark_runs_profile_mode_check/);
  assert.match(down, /DROP COLUMN IF EXISTS profile_mode/);
});
