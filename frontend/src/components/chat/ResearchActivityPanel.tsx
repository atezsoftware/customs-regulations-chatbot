import {useState} from 'react';
import type {LlmUsage, ResearchStep} from '../../types';
import {ResearchStepItem} from './ResearchStepItem';

/**
 * Collapsed by default, always — the caller expands it manually if they
 * want the full step-by-step trace. While a step is running, the header
 * shows the most recent active step. Multi-agent runs can have several live
 * rows at once, so the header also reports their concurrent agent count
 * while the full per-agent list remains available when expanded.
 */
export function ResearchActivityPanel({steps, usage}: {steps: ResearchStep[]; usage: LlmUsage[]}) {
  const [expanded, setExpanded] = useState(false);
  const [expandAllDetails, setExpandAllDetails] = useState(false);

  if (!steps.length) return null;

  const runningSteps = steps.filter(step => step.status === 'running');
  const running = runningSteps.length > 0;
  const hasError = steps.some(step => step.status === 'error');
  const liveStep = runningSteps[runningSteps.length - 1] ?? steps[steps.length - 1];
  const runningAgentCount = new Set(
    runningSteps
      .filter(step => step.stepId.startsWith('agent-'))
      .map(step => {
        const agentId = step.metadata?.agentId;
        return typeof agentId === 'string' && agentId ? agentId : step.stepId;
      }),
  ).size;

  const dotClass = running ? 'bg-indigo-500 animate-pulse' : hasError ? 'bg-rose-500' : 'bg-emerald-500';

  // Only shown while the run is still in progress — once it completes,
  // UsageFooter below already shows the final per-call breakdown, so this
  // would just be a redundant, now-static duplicate of the same numbers.
  const currentStep = usage.length
    ? Math.max(...usage.map(call => call.step ?? 0))
    : undefined;
  const tokensSoFar = usage.reduce(
    (sum, call) => sum + call.inputTokens + call.outputTokens + call.thinkingTokens,
    0,
  );

  return (
    <div className="mb-4 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm transition duration-200">
      <button
        type="button"
        onClick={() => setExpanded(prev => !prev)}
        className="flex w-full items-center justify-between gap-3 bg-gradient-to-r from-slate-50 to-white px-3 py-2 text-left"
      >
        <span className="flex min-w-0 items-center gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full transition-colors ${dotClass}`} />
          {running ? (
            <span key={liveStep.stepId} className="min-w-0 animate-[fadeIn_180ms_ease-out] truncate text-sm">
              <span
                className="bg-[length:200%_100%] bg-clip-text font-medium text-transparent [background-image:linear-gradient(110deg,#475569_40%,#a5b4fc_50%,#475569_60%)] motion-safe:animate-[shimmer_1.8s_linear_infinite]"
              >
                {runningAgentCount > 1
                  ? `${runningAgentCount} agents working · ${liveStep.title}`
                  : liveStep.title}
              </span>
            </span>
          ) : (
            <span className="animate-[fadeIn_180ms_ease-out] truncate text-sm font-medium text-slate-700">
              Research activity ({steps.length})
            </span>
          )}
        </span>
        <span className="flex shrink-0 items-center gap-1.5 text-xs font-medium text-slate-400">
          {running && tokensSoFar > 0 && (
            <span className="animate-[fadeIn_180ms_ease-out] text-slate-400">
              {currentStep ? `step ${currentStep} · ` : ''}
              {tokensSoFar.toLocaleString()} tokens
            </span>
          )}
          {expanded ? 'Hide steps' : 'Show steps'}
          <svg
            viewBox="0 0 20 20"
            fill="currentColor"
            className={`h-3.5 w-3.5 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          >
            <path
              fillRule="evenodd"
              clipRule="evenodd"
              d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 10.94l3.71-3.71a.75.75 0 1 1 1.06 1.06l-4.24 4.25a.75.75 0 0 1-1.06 0L5.21 8.29a.75.75 0 0 1 .02-1.08Z"
            />
          </svg>
        </span>
      </button>
      {expanded && (
        <div className="animate-[fadeIn_180ms_ease-out] border-t border-slate-100">
          <div className="flex justify-end px-2 pt-1">
            <button
              type="button"
              onClick={() => setExpandAllDetails(prev => !prev)}
              className="rounded-full px-2 py-1 text-xs font-medium text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
            >
              {expandAllDetails ? 'Collapse details' : 'Expand details'}
            </button>
          </div>
          <ul className="p-1">
            {steps.map(step => (
              <ResearchStepItem key={`${step.id}-${step.stepId}`} step={step} expandAll={expandAllDetails} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
