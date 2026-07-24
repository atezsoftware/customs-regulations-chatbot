import {useState} from 'react';
import type {ResearchStep} from '../../types';

export function ResearchStepItem({
  step,
  expandAll,
}: {
  step: ResearchStep;
  expandAll: boolean;
}) {
  const isRunning = step.status === 'running';
  const [manualOpen, setManualOpen] = useState(false);
  const open = expandAll || isRunning || manualOpen;
  const dotClass = isRunning
    ? 'bg-indigo-500 shadow-[0_0_0_4px_rgba(99,102,241,0.12)]'
    : step.status === 'error'
      ? 'bg-rose-500'
      : 'bg-emerald-500';

  // Only present for chunk-bearing tool calls (semantic_search,
  // get_document, parse_file, read, get_chunk_context) — see agent.py's
  // CHUNK_BEARING_TOOLS. A genuine 0-chunk result still shows (e.g. "0
  // chunks"), so this stays visible rather than silently vanishing.
  const chunkCount = step.metadata?.chunkCount;
  const retrievalTokens = step.metadata?.retrievalTokensEstimated;
  const chunkSummary =
    typeof chunkCount === 'number'
      ? `${chunkCount} chunk${chunkCount === 1 ? '' : 's'}${
          typeof retrievalTokens === 'number' ? ` · ~${retrievalTokens.toLocaleString()} tokens` : ''
        }`
      : undefined;

  return (
    <li>
      <button
        type="button"
        onClick={() => setManualOpen(prev => !open || !prev)}
        className="group flex w-full items-start gap-3 rounded-lg px-2 py-2 text-left transition duration-200 hover:bg-slate-50"
      >
        <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full transition ${dotClass}`} />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-slate-700 transition group-hover:text-slate-900">
            {step.title}
          </span>
          {step.preview && (
            <span className="mt-0.5 block truncate text-xs text-slate-400">{step.preview}</span>
          )}
          {chunkSummary && (
            <span className="mt-0.5 block truncate text-xs text-slate-400">{chunkSummary}</span>
          )}
        </span>
        <span className="text-xs text-slate-300 transition group-hover:text-slate-500">
          {open ? 'Hide' : 'Show'}
        </span>
      </button>
      {open && step.details && (
        <div className="ml-7 animate-[fadeIn_180ms_ease-out] rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2 text-xs leading-5 text-slate-500">
          {step.details && <p className="whitespace-pre-wrap">{step.details}</p>}
        </div>
      )}
    </li>
  );
}
