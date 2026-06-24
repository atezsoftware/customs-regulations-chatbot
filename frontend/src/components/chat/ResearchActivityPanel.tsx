import {useState} from 'react';
import type {ResearchStep} from '../../types';
import {ResearchStepItem} from './ResearchStepItem';

export function ResearchActivityPanel({steps}: {steps: ResearchStep[]}) {
  const running = steps.some(step => step.status === 'running');
  const [manualOpen, setManualOpen] = useState(false);
  const [expandAll, setExpandAll] = useState(false);
  const open = running || manualOpen;

  if (!steps.length) return null;

  return (
    <div className="mb-4 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm transition duration-200">
      <div className="flex items-center justify-between gap-3 bg-gradient-to-r from-slate-50 to-white px-3 py-2">
        <button
          type="button"
          onClick={() => setManualOpen(prev => !open || !prev)}
          className="flex items-center gap-2 text-sm font-medium text-slate-700"
        >
          <span className={`h-2 w-2 rounded-full ${running ? 'bg-indigo-500 animate-pulse' : 'bg-emerald-500'}`} />
          Research activity ({steps.length})
        </button>
        <div className="flex items-center gap-2">
          {open && (
            <button
              type="button"
              onClick={() => setExpandAll(prev => !prev)}
              className="rounded-full px-2 py-1 text-xs font-medium text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
            >
              {expandAll ? 'Collapse details' : 'Expand details'}
            </button>
          )}
          <span className="text-xs text-slate-300">{open ? 'Open' : 'Closed'}</span>
        </div>
      </div>
      {open && (
        <ul className="animate-[fadeIn_180ms_ease-out] border-t border-slate-100 p-1">
          {steps.map(step => (
            <ResearchStepItem key={`${step.id}-${step.stepId}`} step={step} expandAll={expandAll} />
          ))}
        </ul>
      )}
    </div>
  );
}
