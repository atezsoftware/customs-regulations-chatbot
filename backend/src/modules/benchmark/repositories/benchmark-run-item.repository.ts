import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {BenchmarkRunItem, BenchmarkRunItemRelations} from '../models';

export interface BenchmarkModelMetricsRow {
  provider: string;
  model_id: string;
  total_count: number | string;
  completed_count: number | string;
  error_count: number | string;
  avg_steps: number | string | null;
  avg_tokens_per_step: number | string | null;
  avg_total_tokens: number | string | null;
  avg_duration_ms: number | string | null;
  avg_duration_per_step_ms: number | string | null;
  avg_cost_usd: number | string | null;
  total_cost_usd: number | string | null;
  success_rate: number | string | null;
  error_rate: number | string | null;
  citation_rate: number | string | null;
  avg_api_calls: number | string | null;
  avg_context_summaries: number | string | null;
  judge_overall_score: number | string | null;
  judge_correctness: number | string | null;
  judge_groundedness: number | string | null;
  judge_completeness: number | string | null;
  judge_clarity: number | string | null;
}

export interface BenchmarkModelPercentilesRow {
  provider: string;
  model_id: string;
  p50_duration_ms: number | string | null;
  p95_duration_ms: number | string | null;
}

export class BenchmarkRunItemRepository extends DefaultCrudRepository<
  BenchmarkRunItem,
  typeof BenchmarkRunItem.prototype.id,
  BenchmarkRunItemRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(BenchmarkRunItem, dataSource);
  }

  /**
   * Atomically claims up to `limit` pending items belonging to `running`
   * runs, marking them `running` in the same statement. Safe to call from
   * only one orchestrator tick at a time (see `BenchmarkRunnerService`'s
   * advisory lock) — no `FOR UPDATE SKIP LOCKED` needed on top of that.
   */
  async claimPendingItems(limit: number): Promise<BenchmarkRunItem[]> {
    // The postgresql connector wraps UPDATE/DELETE results as
    // {affectedRows, count, rows} even with RETURNING — unlike a plain
    // SELECT or INSERT, which return `rows` directly. See
    // loopback-connector-postgresql's executeSQL: only the DELETE/UPDATE
    // branch does this wrapping.
    const result = (await this.dataSource.execute(
      `
        UPDATE benchmark_run_items
        SET status = 'running', started_at = now()
        WHERE id IN (
          SELECT bri.id
          FROM benchmark_run_items bri
          JOIN benchmark_runs br ON br.id = bri.run_id
          WHERE bri.status = 'pending' AND br.status = 'running'
          ORDER BY bri.id
          LIMIT $1
        )
        RETURNING *
      `,
      [limit],
    )) as {rows: Record<string, unknown>[]};
    return result.rows.map(row => new BenchmarkRunItem(this.toEntityData(row)));
  }

  /**
   * Reclaims items stuck `running` past `staleMinutes` (a stranded backend
   * restart, a hung fetch) back to `pending` so they get retried on a later
   * tick instead of leaving the parent run stuck forever.
   */
  async reclaimStaleItems(staleMinutes: number): Promise<number> {
    const result = (await this.dataSource.execute(
      `
        UPDATE benchmark_run_items
        SET status = 'pending', started_at = NULL
        WHERE status = 'running' AND started_at < now() - make_interval(mins => $1)
        RETURNING id
      `,
      [staleMinutes],
    )) as {count: number};
    return result.count;
  }

  /**
   * Per-(provider, model) aggregate metrics for one run, computed at read
   * time from item rows — no stored aggregate columns to keep in sync.
   * Pooled sums-over-sums for the per-step ratios (avg_tokens_per_step,
   * avg_duration_per_step_ms), not an average of each item's own ratio, so a
   * handful of tiny/huge runs don't distort the figure.
   */
  async aggregateByModel(runId: number): Promise<BenchmarkModelMetricsRow[]> {
    return (await this.dataSource.execute(
      `
        SELECT
          bri.provider,
          bri.model_id,
          COUNT(*) AS total_count,
          COUNT(*) FILTER (WHERE bri.status = 'completed') AS completed_count,
          COUNT(*) FILTER (WHERE bri.status = 'error') AS error_count,
          AVG(bri.steps) FILTER (WHERE bri.status = 'completed') AS avg_steps,
          (SUM(bri.total_tokens) FILTER (WHERE bri.status = 'completed'))::float
            / NULLIF(SUM(bri.steps) FILTER (WHERE bri.status = 'completed'), 0) AS avg_tokens_per_step,
          AVG(bri.total_tokens) FILTER (WHERE bri.status = 'completed') AS avg_total_tokens,
          AVG(bri.duration_ms) FILTER (WHERE bri.status = 'completed') AS avg_duration_ms,
          (SUM(bri.duration_ms) FILTER (WHERE bri.status = 'completed'))::float
            / NULLIF(SUM(bri.steps) FILTER (WHERE bri.status = 'completed'), 0) AS avg_duration_per_step_ms,
          AVG(bri.cost_usd) FILTER (WHERE bri.status = 'completed') AS avg_cost_usd,
          SUM(bri.cost_usd) FILTER (WHERE bri.status = 'completed') AS total_cost_usd,
          (COUNT(*) FILTER (WHERE bri.status = 'completed' AND NOT bri.incomplete))::float
            / NULLIF(COUNT(*), 0) AS success_rate,
          (COUNT(*) FILTER (WHERE bri.status = 'error'))::float
            / NULLIF(COUNT(*), 0) AS error_rate,
          (COUNT(*) FILTER (
            WHERE bri.status = 'completed' AND jsonb_array_length(bri.cited_sources) > 0
          ))::float / NULLIF(COUNT(*) FILTER (WHERE bri.status = 'completed'), 0) AS citation_rate,
          AVG(bri.api_calls) FILTER (WHERE bri.status = 'completed') AS avg_api_calls,
          AVG(bri.context_summaries) FILTER (WHERE bri.status = 'completed') AS avg_context_summaries,
          AVG(brj.overall_score) AS judge_overall_score,
          AVG(brj.correctness_score) AS judge_correctness,
          AVG(brj.groundedness_score) AS judge_groundedness,
          AVG(brj.completeness_score) AS judge_completeness,
          AVG(brj.clarity_score) AS judge_clarity
        FROM benchmark_run_items bri
        LEFT JOIN benchmark_run_judgments brj ON brj.run_item_id = bri.id
        WHERE bri.run_id = $1
        GROUP BY bri.provider, bri.model_id
        ORDER BY bri.provider, bri.model_id
      `,
      [runId],
    )) as BenchmarkModelMetricsRow[];
  }

  /**
   * Latency percentiles, kept as a separate query (filtered with WHERE
   * instead of FILTER) so the ordered-set aggregate only ever sees
   * completed rows — simpler than reasoning about FILTER's interaction
   * with WITHIN GROUP.
   */
  async percentilesByModel(runId: number): Promise<BenchmarkModelPercentilesRow[]> {
    return (await this.dataSource.execute(
      `
        SELECT
          provider,
          model_id,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_duration_ms,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms
        FROM benchmark_run_items
        WHERE run_id = $1 AND status = 'completed'
        GROUP BY provider, model_id
      `,
      [runId],
    )) as BenchmarkModelPercentilesRow[];
  }

  private toEntityData(row: Record<string, unknown>): Partial<BenchmarkRunItem> {
    return {
      id: row.id as number,
      runId: row.run_id as number,
      provider: row.provider as string,
      modelId: row.model_id as string,
      questionId: row.question_id as number,
      repeatIndex: row.repeat_index as number,
      status: row.status as BenchmarkRunItem['status'],
      finalResult: (row.final_result as string | null) ?? undefined,
      errorMessage: (row.error_message as string | null) ?? undefined,
      incomplete: row.incomplete as boolean,
      steps: (row.steps as number | null) ?? undefined,
      apiCalls: (row.api_calls as number | null) ?? undefined,
      promptTokens: (row.prompt_tokens as number | null) ?? undefined,
      completionTokens: (row.completion_tokens as number | null) ?? undefined,
      thinkingTokens: (row.thinking_tokens as number | null) ?? undefined,
      totalTokens: (row.total_tokens as number | null) ?? undefined,
      toolResultChars: (row.tool_result_chars as number | null) ?? undefined,
      contextSummaries: (row.context_summaries as number | null) ?? undefined,
      durationMs: (row.duration_ms as number | null) ?? undefined,
      costUsd: (row.cost_usd as string | null) ?? undefined,
      costSource: (row.cost_source as BenchmarkRunItem['costSource']) ?? undefined,
      citedSources: (row.cited_sources as string[] | null) ?? undefined,
      stepPath: (row.step_path as string[] | null) ?? undefined,
      startedAt: (row.started_at as string | null) ?? undefined,
      completedAt: (row.completed_at as string | null) ?? undefined,
    };
  }
}
