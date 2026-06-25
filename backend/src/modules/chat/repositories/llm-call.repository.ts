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
        SELECT
          COUNT(c.id) AS call_count,
          COUNT(DISTINCT s.id) AS session_count,
          COALESCE(SUM(c.input_tokens), 0) AS input_tokens,
          COALESCE(SUM(c.output_tokens), 0) AS output_tokens,
          COALESCE(SUM(c.thinking_tokens), 0) AS thinking_tokens,
          COALESCE(SUM(c.input_tokens + c.output_tokens + c.thinking_tokens), 0) AS total_tokens,
          ROUND(AVG(c.duration_ms)) AS avg_duration_ms
        FROM chat_sessions s
        LEFT JOIN llm_calls c ON c.session_id = s.id
        WHERE s.user_id = $1
          AND ($2::timestamptz IS NULL OR c.created_at >= $2::timestamptz)
      `,
      [userId, since],
    )) as UsageTotalsRow[];
  }

  async usageDaily(userId: number, since: string | null): Promise<DailyUsageRow[]> {
    return (await this.dataSource.execute(
      `
        SELECT
          to_char(date_trunc('day', c.created_at), 'YYYY-MM-DD') AS day,
          COALESCE(SUM(c.input_tokens), 0) AS input_tokens,
          COALESCE(SUM(c.output_tokens), 0) AS output_tokens,
          COALESCE(SUM(c.thinking_tokens), 0) AS thinking_tokens,
          COALESCE(SUM(c.input_tokens + c.output_tokens + c.thinking_tokens), 0) AS total_tokens,
          COUNT(c.id) AS call_count
        FROM llm_calls c
        JOIN chat_sessions s ON s.id = c.session_id
        WHERE s.user_id = $1
          AND ($2::timestamptz IS NULL OR c.created_at >= $2::timestamptz)
        GROUP BY date_trunc('day', c.created_at)
        ORDER BY date_trunc('day', c.created_at) ASC
      `,
      [userId, since],
    )) as DailyUsageRow[];
  }

  async usageTopSessions(userId: number, since: string | null): Promise<SessionUsageRow[]> {
    return (await this.dataSource.execute(
      `
        SELECT
          s.id AS session_id,
          s.title,
          s.updated_at,
          COUNT(c.id) AS call_count,
          COALESCE(SUM(c.input_tokens), 0) AS input_tokens,
          COALESCE(SUM(c.output_tokens), 0) AS output_tokens,
          COALESCE(SUM(c.thinking_tokens), 0) AS thinking_tokens,
          COALESCE(SUM(c.input_tokens + c.output_tokens + c.thinking_tokens), 0) AS total_tokens
        FROM chat_sessions s
        JOIN llm_calls c ON c.session_id = s.id
        WHERE s.user_id = $1
          AND ($2::timestamptz IS NULL OR c.created_at >= $2::timestamptz)
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
        SELECT
          c.provider,
          c.model,
          COUNT(c.id) AS call_count,
          COALESCE(SUM(c.input_tokens + c.output_tokens + c.thinking_tokens), 0) AS total_tokens
        FROM llm_calls c
        JOIN chat_sessions s ON s.id = c.session_id
        WHERE s.user_id = $1
          AND ($2::timestamptz IS NULL OR c.created_at >= $2::timestamptz)
        GROUP BY c.provider, c.model
        ORDER BY total_tokens DESC
      `,
      [userId, since],
    )) as ModelUsageRow[];
  }
}
