import {inject} from '@loopback/core';
import {get, param, Request, response, RestBindings} from '@loopback/rest';
import {PostgresDataSource} from '../../../datasources';
import {getCurrentUser} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {repository} from '@loopback/repository';

type UsageRange = '7d' | '30d' | '90d' | 'all';

interface UsageTotalsRow {
  call_count: number | string;
  session_count: number | string;
  input_tokens: number | string;
  output_tokens: number | string;
  thinking_tokens: number | string;
  total_tokens: number | string;
  avg_duration_ms: number | string | null;
}

interface DailyUsageRow {
  day: string;
  input_tokens: number | string;
  output_tokens: number | string;
  thinking_tokens: number | string;
  total_tokens: number | string;
  call_count: number | string;
}

interface SessionUsageRow {
  session_id: number | string;
  title: string | null;
  updated_at: string | Date | null;
  call_count: number | string;
  total_tokens: number | string;
  input_tokens: number | string;
  output_tokens: number | string;
  thinking_tokens: number | string;
}

interface ModelUsageRow {
  provider: string;
  model: string | null;
  call_count: number | string;
  total_tokens: number | string;
}

export class UsageController {
  constructor(
    @repository(UserRepository) private userRepository: UserRepository,
    @inject('datasources.postgres') private dataSource: PostgresDataSource,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  @get('/analytics/usage')
  @response(200, {description: 'Token usage analytics for the current user'})
  async usage(@param.query.string('range') range?: UsageRange) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const normalizedRange = normalizeRange(range);
    const since = rangeStart(normalizedRange);
    const params = [user.id, since];

    const [totalsRows, dailyRows, sessionRows, modelRows] = await Promise.all([
      this.dataSource.execute(
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
        params,
      ) as Promise<UsageTotalsRow[]>,
      this.dataSource.execute(
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
        params,
      ) as Promise<DailyUsageRow[]>,
      this.dataSource.execute(
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
        params,
      ) as Promise<SessionUsageRow[]>,
      this.dataSource.execute(
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
        params,
      ) as Promise<ModelUsageRow[]>,
    ]);

    const totals = totalsRows[0] ?? emptyTotals();
    return {
      range: normalizedRange,
      since,
      totals: {
        calls: numberValue(totals.call_count),
        sessions: numberValue(totals.session_count),
        inputTokens: numberValue(totals.input_tokens),
        outputTokens: numberValue(totals.output_tokens),
        thinkingTokens: numberValue(totals.thinking_tokens),
        totalTokens: numberValue(totals.total_tokens),
        avgDurationMs: nullableNumber(totals.avg_duration_ms),
      },
      daily: dailyRows.map(row => ({
        day: row.day,
        inputTokens: numberValue(row.input_tokens),
        outputTokens: numberValue(row.output_tokens),
        thinkingTokens: numberValue(row.thinking_tokens),
        totalTokens: numberValue(row.total_tokens),
        calls: numberValue(row.call_count),
      })),
      topSessions: sessionRows.map(row => ({
        sessionId: numberValue(row.session_id),
        title: row.title ?? 'Untitled chat',
        updatedAt: dateString(row.updated_at),
        calls: numberValue(row.call_count),
        inputTokens: numberValue(row.input_tokens),
        outputTokens: numberValue(row.output_tokens),
        thinkingTokens: numberValue(row.thinking_tokens),
        totalTokens: numberValue(row.total_tokens),
      })),
      models: modelRows.map(row => ({
        provider: row.provider,
        model: row.model ?? 'default',
        calls: numberValue(row.call_count),
        totalTokens: numberValue(row.total_tokens),
      })),
    };
  }
}

function normalizeRange(range?: string): UsageRange {
  return range === '7d' || range === '90d' || range === 'all' ? range : '30d';
}

function rangeStart(range: UsageRange): string | null {
  if (range === 'all') return null;
  const days = range === '7d' ? 7 : range === '90d' ? 90 : 30;
  return new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
}

function emptyTotals(): UsageTotalsRow {
  return {
    call_count: 0,
    session_count: 0,
    input_tokens: 0,
    output_tokens: 0,
    thinking_tokens: 0,
    total_tokens: 0,
    avg_duration_ms: null,
  };
}

function numberValue(value: number | string | null | undefined): number {
  return Number(value ?? 0);
}

function nullableNumber(value: number | string | null | undefined): number | null {
  return value === null || value === undefined ? null : Number(value);
}

function dateString(value: string | Date | null | undefined): string | undefined {
  if (!value) return undefined;
  return value instanceof Date ? value.toISOString() : value;
}
