import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {LlmCall, LlmCallRelations} from '../models';

export interface UsageTotalsRow {
  call_count: number | string;
  session_count: number | string;
  input_tokens: number | string;
  output_tokens: number | string;
  thinking_tokens: number | string;
  total_tokens: number | string;
  avg_duration_ms: number | string | null;
}

export interface DailyUsageRow {
  day: string;
  input_tokens: number | string;
  output_tokens: number | string;
  thinking_tokens: number | string;
  total_tokens: number | string;
  call_count: number | string;
}

export interface SessionUsageRow {
  session_id: number | string;
  title: string | null;
  updated_at: string | Date | null;
  call_count: number | string;
  total_tokens: number | string;
  input_tokens: number | string;
  output_tokens: number | string;
  thinking_tokens: number | string;
}

export interface ModelUsageRow {
  provider: string;
  model: string | null;
  call_count: number | string;
  total_tokens: number | string;
}

export class LlmCallRepository extends DefaultCrudRepository<
  LlmCall,
  typeof LlmCall.prototype.id,
  LlmCallRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(LlmCall, dataSource);
  }

  /**
   * Aggregate token usage for a user's chat sessions, optionally since a
   * given timestamp. Cross-table SUM/COUNT/GROUP BY and date-bucketing the
   * juggler query builder cannot express, so this stays as raw SQL.
   */
  async usageTotals(userId: number, since: string | null): Promise<UsageTotalsRow[]> {
    return (await this.dataSource.execute(
      `
        WITH usage_events AS (
          ${usageEventsSql()}
        )
        SELECT
          COUNT(*) AS call_count,
          COUNT(DISTINCT session_id) AS session_count,
          COALESCE(SUM(input_tokens), 0) AS input_tokens,
          COALESCE(SUM(output_tokens), 0) AS output_tokens,
          COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
          COALESCE(SUM(input_tokens + output_tokens + thinking_tokens), 0) AS total_tokens,
          ROUND(AVG(duration_ms)) AS avg_duration_ms
        FROM usage_events
      `,
      [userId, since],
    )) as UsageTotalsRow[];
  }

  async usageDaily(userId: number, since: string | null): Promise<DailyUsageRow[]> {
    return (await this.dataSource.execute(
      `
        WITH usage_events AS (
          ${usageEventsSql()}
        )
        SELECT
          to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS day,
          COALESCE(SUM(input_tokens), 0) AS input_tokens,
          COALESCE(SUM(output_tokens), 0) AS output_tokens,
          COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
          COALESCE(SUM(input_tokens + output_tokens + thinking_tokens), 0) AS total_tokens,
          COUNT(*) AS call_count
        FROM usage_events
        GROUP BY date_trunc('day', created_at)
        ORDER BY date_trunc('day', created_at) ASC
      `,
      [userId, since],
    )) as DailyUsageRow[];
  }

  async usageTopSessions(userId: number, since: string | null): Promise<SessionUsageRow[]> {
    return (await this.dataSource.execute(
      `
        WITH usage_events AS (
          ${usageEventsSql()}
        )
        SELECT
          s.id AS session_id,
          s.title,
          s.updated_at,
          COUNT(e.id) AS call_count,
          COALESCE(SUM(e.input_tokens), 0) AS input_tokens,
          COALESCE(SUM(e.output_tokens), 0) AS output_tokens,
          COALESCE(SUM(e.thinking_tokens), 0) AS thinking_tokens,
          COALESCE(SUM(e.input_tokens + e.output_tokens + e.thinking_tokens), 0) AS total_tokens
        FROM usage_events e
        JOIN chat_sessions s ON s.id = e.session_id
        GROUP BY s.id
        ORDER BY total_tokens DESC, s.updated_at DESC
        LIMIT 10
      `,
      [userId, since],
    )) as SessionUsageRow[];
  }

  async usageByModel(userId: number, since: string | null): Promise<ModelUsageRow[]> {
    return (await this.dataSource.execute(
      `
        WITH usage_events AS (
          ${usageEventsSql()}
        )
        SELECT
          provider,
          model,
          COUNT(*) AS call_count,
          COALESCE(SUM(input_tokens + output_tokens + thinking_tokens), 0) AS total_tokens
        FROM usage_events
        GROUP BY provider, model
        ORDER BY total_tokens DESC
      `,
      [userId, since],
    )) as ModelUsageRow[];
  }
}

function usageEventsSql(): string {
  return `
    SELECT
      c.id,
      c.message_id,
      c.session_id,
      c.provider,
      c.model,
      c.input_tokens,
      c.output_tokens,
      c.thinking_tokens,
      c.duration_ms,
      c.created_at
    FROM llm_calls c
    JOIN chat_sessions s ON s.id = c.session_id
    WHERE s.user_id = $1
      AND ($2::timestamptz IS NULL OR c.created_at >= $2::timestamptz)

    UNION ALL

    SELECT
      -m.id AS id,
      m.id AS message_id,
      m.session_id,
      'chat' AS provider,
      COALESCE(s.model, 'estimated') AS model,
      0 AS input_tokens,
      GREATEST(1, CEIL(char_length(m.content)::numeric / 4))::integer AS output_tokens,
      0 AS thinking_tokens,
      NULL::integer AS duration_ms,
      COALESCE(m.updated_at, m.created_at) AS created_at
    FROM chat_messages m
    JOIN chat_sessions s ON s.id = m.session_id
    LEFT JOIN llm_calls c ON c.message_id = m.id
    WHERE s.user_id = $1
      AND m.role = 'assistant'
      AND m.status = 'completed'
      AND c.id IS NULL
      AND ($2::timestamptz IS NULL OR COALESCE(m.updated_at, m.created_at) >= $2::timestamptz)
  `;
}
