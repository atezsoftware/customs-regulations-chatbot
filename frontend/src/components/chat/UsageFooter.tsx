import type {LlmUsage} from '../../types';
import {formatUsd} from '../../lib/format';

export function UsageFooter({usage}: {usage: LlmUsage[]}) {
  if (!usage.length) return null;

  const totalTokens = usage.reduce(
    (sum, call) => sum + call.inputTokens + call.outputTokens + call.thinkingTokens,
    0,
  );
  const totalDurationMs = usage.reduce((sum, call) => sum + (call.durationMs ?? 0), 0);
  const models = [...new Set(usage.map(call => call.model).filter(Boolean))] as string[];
  const costs = usage.map(call => Number(call.billedCostUsd)).filter(Number.isFinite);
  const cost = costs.length ? costs.reduce((sum, value) => sum + value, 0) : undefined;
  const estimated = usage.some(call => call.costSource === 'estimated');

  if (!totalTokens && !totalDurationMs && !models.length) return null;

  const parts = [
    models.length ? models.join(', ') : undefined,
    totalTokens ? `${totalTokens.toLocaleString()} tokens` : undefined,
    totalDurationMs ? `${(totalDurationMs / 1000).toFixed(1)}s` : undefined,
    usage.length > 1 ? `${usage.length} calls` : undefined,
    formatUsd(cost),
    estimated ? 'Estimated cost' : undefined,
  ].filter(Boolean);

  return (
    <p className="mt-3 text-[11px] font-medium uppercase tracking-wide text-slate-400">
      {parts.join(' · ')}
    </p>
  );
}
