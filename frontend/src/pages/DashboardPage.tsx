import {useEffect, useMemo, useState} from 'react';
import {Link} from 'react-router-dom';
import {useAuth} from '../context/useAuth';
import {usageApi} from '../lib/endpoints';
import type {UsageAnalytics, UsageRange} from '../types';

const ranges: Array<{label: string; value: UsageRange}> = [
  {label: '7D', value: '7d'},
  {label: '30D', value: '30d'},
  {label: '90D', value: '90d'},
  {label: 'All', value: 'all'},
];

export function DashboardPage() {
  const {user} = useAuth();
  const [range, setRange] = useState<UsageRange>('30d');
  const [usage, setUsage] = useState<UsageAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    usageApi
      .get(range)
      .then(data => {
        if (!cancelled) setUsage(data);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [range]);

  const tokenBreakdown = useMemo(() => {
    const totals = usage?.totals;
    if (!totals || totals.totalTokens <= 0) return [];
    return [
      {label: 'Input', value: totals.inputTokens, className: 'bg-sky-500'},
      {label: 'Output', value: totals.outputTokens, className: 'bg-emerald-500'},
      {label: 'Thinking', value: totals.thinkingTokens, className: 'bg-amber-500'},
    ].filter(item => item.value > 0);
  }, [usage]);

  const maxDaily = Math.max(...(usage?.daily.map(item => item.totalTokens) ?? [0]), 1);

  return (
    <main className="min-w-0 flex-1 overflow-y-auto bg-slate-50">
      <div className="mx-auto max-w-6xl px-6 py-6">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Dashboard
            </p>
            <h1 className="mt-1 truncate text-2xl font-semibold text-slate-900">
              {user?.fullName ?? user?.email ?? 'Profile'}
            </h1>
            <p className="mt-1 text-sm text-slate-500">{user?.email}</p>
          </div>
          <div className="flex rounded-xl border border-slate-200 bg-white p-1">
            {ranges.map(item => (
              <button
                key={item.value}
                type="button"
                onClick={() => setRange(item.value)}
                className={`h-9 min-w-12 rounded-lg px-3 text-sm font-medium transition-colors ${
                  range === item.value
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        {loading || !usage ? (
          <div className="rounded-xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-400">
            Loading usage...
          </div>
        ) : (
          <div className="space-y-5">
            <section className="grid gap-3 md:grid-cols-4">
              <Metric label="Total tokens" value={formatNumber(usage.totals.totalTokens)} />
              <Metric label="LLM calls" value={formatNumber(usage.totals.calls)} />
              <Metric label="Active chats" value={formatNumber(usage.totals.sessions)} />
              <Metric
                label="Avg response"
                value={
                  usage.totals.avgDurationMs
                    ? formatDuration(usage.totals.avgDurationMs)
                    : 'No data'
                }
              />
            </section>

            <section className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <h2 className="text-base font-semibold text-slate-900">Daily usage</h2>
                    <p className="mt-1 text-sm text-slate-400">
                      Tokens generated from completed assistant calls.
                    </p>
                  </div>
                </div>
                {usage.daily.length === 0 ? (
                  <EmptyPanel text="No token usage recorded for this period." />
                ) : (
                  <div className="flex h-56 items-end gap-2 border-b border-slate-100 pt-4">
                    {usage.daily.map(day => (
                      <div key={day.day} className="flex min-w-0 flex-1 flex-col items-center gap-2">
                        <div className="flex h-44 w-full items-end">
                          <div
                            title={`${day.day}: ${formatNumber(day.totalTokens)} tokens`}
                            className="w-full rounded-t-md bg-indigo-500/80 transition-colors hover:bg-indigo-600"
                            style={{
                              height: `${Math.max(6, (day.totalTokens / maxDaily) * 100)}%`,
                            }}
                          />
                        </div>
                        <span className="w-full truncate text-center text-[11px] text-slate-400">
                          {shortDate(day.day)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="text-base font-semibold text-slate-900">Token mix</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Input, output and reasoning-style token distribution.
                </p>
                {tokenBreakdown.length === 0 ? (
                  <EmptyPanel text="No token breakdown yet." />
                ) : (
                  <div className="mt-5 space-y-4">
                    {tokenBreakdown.map(item => (
                      <div key={item.label}>
                        <div className="mb-1.5 flex items-center justify-between text-sm">
                          <span className="font-medium text-slate-700">{item.label}</span>
                          <span className="text-slate-400">{formatNumber(item.value)}</span>
                        </div>
                        <div className="h-2 rounded-full bg-slate-100">
                          <div
                            className={`h-2 rounded-full ${item.className}`}
                            style={{
                              width: `${Math.max(
                                2,
                                (item.value / usage.totals.totalTokens) * 100,
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>

            <section className="grid gap-5 xl:grid-cols-2">
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-4 text-base font-semibold text-slate-900">
                  Highest usage chats
                </h2>
                {usage.topSessions.length === 0 ? (
                  <EmptyPanel text="No chat usage yet." />
                ) : (
                  <div className="space-y-2">
                    {usage.topSessions.map(session => (
                      <Link
                        key={session.sessionId}
                        to={`/chat/${session.sessionId}`}
                        className="flex items-center justify-between gap-4 rounded-lg border border-slate-100 px-3 py-2.5 transition-colors hover:border-indigo-100 hover:bg-indigo-50/40"
                      >
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-medium text-slate-800">
                            {session.title}
                          </span>
                          <span className="mt-0.5 block text-xs text-slate-400">
                            {session.calls} calls
                          </span>
                        </span>
                        <span className="shrink-0 text-sm font-semibold text-slate-700">
                          {formatNumber(session.totalTokens)}
                        </span>
                      </Link>
                    ))}
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="mb-4 text-base font-semibold text-slate-900">Model usage</h2>
                {usage.models.length === 0 ? (
                  <EmptyPanel text="No model usage yet." />
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead className="text-xs uppercase tracking-wide text-slate-400">
                        <tr>
                          <th className="pb-2 font-semibold">Model</th>
                          <th className="pb-2 font-semibold">Calls</th>
                          <th className="pb-2 text-right font-semibold">Tokens</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {usage.models.map(model => (
                          <tr key={`${model.provider}-${model.model}`}>
                            <td className="py-2.5">
                              <span className="block font-medium text-slate-700">
                                {model.model}
                              </span>
                              <span className="text-xs text-slate-400">{model.provider}</span>
                            </td>
                            <td className="py-2.5 text-slate-500">{model.calls}</td>
                            <td className="py-2.5 text-right font-semibold text-slate-700">
                              {formatNumber(model.totalTokens)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}
      </div>
    </main>
  );
}

function Metric({label, value}: {label: string; value: string}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <p className="text-xs font-medium text-slate-400">{label}</p>
      <p className="mt-1 text-xl font-semibold text-slate-900">{value}</p>
    </div>
  );
}

function EmptyPanel({text}: {text: string}) {
  return (
    <div className="mt-4 rounded-lg border border-dashed border-slate-200 px-4 py-8 text-center text-sm text-slate-400">
      {text}
    </div>
  );
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(Math.round(value));
}

function formatDuration(value: number): string {
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(1)} s`;
}

function shortDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
}
