import type {ChatSession} from '../../types';

export function SessionUsageBadge({session}: {session: ChatSession}) {
  const totalTokens = session.totalTokens ?? 0;
  const ratio = session.lastContextUsageRatio;
  const hasRatio = typeof ratio === 'number';
  if (!totalTokens && !hasRatio) return null;

  const parts = [
    totalTokens ? `${new Intl.NumberFormat().format(totalTokens)} tokens total` : undefined,
    hasRatio ? `Context ${(ratio * 100).toFixed(ratio < 0.01 ? 2 : 1)}% full` : undefined,
  ].filter(Boolean);

  const isNearLimit = hasRatio && ratio >= 0.85;

  return (
    <p className={`mt-1 text-xs font-medium ${isNearLimit ? 'text-amber-600' : 'text-slate-400'}`}>
      {parts.join(' · ')}
    </p>
  );
}
