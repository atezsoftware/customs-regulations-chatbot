import type {BenchmarkModelMetrics} from '../../types';

// Fixed categorical order (blue, green, magenta, yellow, aqua, orange, violet,
// red) — color follows the model's position, never reassigned when the set
// changes. Every bar also carries its own model-name + value text label, so
// identity never depends on the color alone.
const SERIES_COLORS = [
  '#2a78d6',
  '#008300',
  '#e87ba4',
  '#eda100',
  '#1baf7a',
  '#eb6834',
  '#4a3aa7',
  '#e34948',
];

interface MetricSpec {
  key: string;
  label: string;
  value: (metrics: BenchmarkModelMetrics) => number | null;
  format: (value: number) => string;
}

const METRICS: MetricSpec[] = [
  {
    key: 'duration',
    label: 'Avg duration',
    value: m => m.avgDurationMs,
    format: value => `${(value / 1000).toFixed(1)}s`,
  },
  {
    key: 'tokens',
    label: 'Avg total tokens',
    value: m => m.avgTotalTokens,
    format: value => new Intl.NumberFormat().format(Math.round(value)),
  },
  {
    key: 'cost',
    label: 'Avg cost',
    value: m => m.avgCostUsd,
    format: value =>
      new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 6,
      }).format(value),
  },
  {
    key: 'judge',
    label: 'Judge score',
    value: m => m.judgeOverallScore,
    format: value => `${Math.round(value)}/100`,
  },
];

function shortModelName(modelId: string): string {
  const afterSlash = modelId.includes('/') ? modelId.split('/').slice(1).join('/') : modelId;
  return afterSlash.length > 22 ? `${afterSlash.slice(0, 21)}…` : afterSlash;
}

export function BenchmarkMetricBars({metrics}: {metrics: BenchmarkModelMetrics[]}) {
  const comparable = metrics.filter(metric => metric.completedCount > 0);
  if (comparable.length === 0) {
    return (
      <p className="rounded-xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-400">
        No completed items yet — the chart fills in once at least one item per model finishes.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {METRICS.map(metric => {
        const rows = comparable
          .map((model, index) => ({model, index, value: metric.value(model)}))
          .filter((row): row is {model: BenchmarkModelMetrics; index: number; value: number} => row.value !== null);
        const max = Math.max(...rows.map(row => row.value), 0);

        return (
          <div key={metric.key} className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
              {metric.label}
            </p>
            {rows.length === 0 ? (
              <p className="text-sm text-slate-400">No data yet.</p>
            ) : (
              <div className="space-y-2">
                {rows.map(({model, index, value}) => (
                  <div key={`${model.provider}/${model.modelId}`} className="flex items-center gap-2">
                    <span
                      className="w-28 shrink-0 truncate text-xs text-slate-500"
                      title={`${model.provider}/${model.modelId}`}
                    >
                      {shortModelName(model.modelId)}
                    </span>
                    <div className="h-4 flex-1 rounded-sm bg-slate-100">
                      <div
                        className="h-4 rounded-r"
                        style={{
                          width: max > 0 ? `${Math.max((value / max) * 100, 3)}%` : '0%',
                          backgroundColor: SERIES_COLORS[index % SERIES_COLORS.length],
                        }}
                      />
                    </div>
                    <span className="w-16 shrink-0 text-right text-xs tabular-nums text-slate-600">
                      {metric.format(value)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
