import {repository} from '@loopback/repository';
import {get, param, Request, response, RestBindings} from '@loopback/rest';
import {inject} from '@loopback/core';
import {getCurrentUser} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {LlmCallRepository, UsageTotalsRow} from '../../chat/repositories';

type UsageRange = '7d' | '30d' | '90d' | 'all';

export class UsageController {
  constructor(
    @repository(UserRepository) private userRepository: UserRepository,
    @repository(LlmCallRepository) private llmCallRepository: LlmCallRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  @get('/analytics/usage')
  @response(200, {description: 'Token usage analytics for the current user'})
  async usage(@param.query.string('range') range?: UsageRange) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const normalizedRange = normalizeRange(range);
    const since = rangeStart(normalizedRange);
    const userId = user.id!;

    const [totalsRows, dailyRows, sessionRows, modelRows] = await Promise.all([
      this.llmCallRepository.usageTotals(userId, since),
      this.llmCallRepository.usageDaily(userId, since),
      this.llmCallRepository.usageTopSessions(userId, since),
      this.llmCallRepository.usageByModel(userId, since),
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
