import {useMemo, useState} from 'react';
import {formatUsd} from '../../lib/format';
import type {LlmModelOption} from '../../types';
import {applyModelSelection} from './model-selector-state';

export function ModelSelector({models, modelId, defaultModelId, disabled, onChange}: {
  models: LlmModelOption[];
  modelId?: string;
  defaultModelId?: string;
  disabled?: boolean;
  onChange: (model: LlmModelOption) => void;
}) {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const selected = models.find(model => model.modelId === modelId) ?? models.find(model => model.modelId === defaultModelId);
  const shown = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return models.filter(model => !normalized || `${model.displayName} ${model.modelId}`.toLowerCase().includes(normalized));
  }, [models, query]);
  if (!selected) return <span className="text-xs text-slate-400">Models unavailable</span>;
  return (
    <details
      open={isOpen}
      className="relative min-w-0"
      onToggle={event => {
        const open = (event.currentTarget as HTMLDetailsElement).open;
        setIsOpen(open);
        if (!open) setQuery('');
      }}
    >
      <summary className="cursor-pointer list-none rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-600 disabled:cursor-not-allowed">
        {selected.displayName} · {formatUsd(selected.promptUsdPerMillion) ?? '$—'} in / {formatUsd(selected.completionUsdPerMillion) ?? '$—'} out
      </summary>
      <div className="absolute bottom-9 left-0 z-20 w-[min(32rem,calc(100vw-3rem))] rounded-xl border border-slate-200 bg-white p-2 shadow-xl">
        <input autoFocus value={query} onChange={event => setQuery(event.target.value)} placeholder="Search models" className="mb-2 w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm" />
        <ul className="max-h-64 overflow-y-auto">
          {shown.map(model => <li key={model.modelId}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => {
                const nextState = applyModelSelection(model, onChange);
                setIsOpen(nextState.isOpen);
                setQuery(nextState.query);
              }}
              className="w-full rounded-lg px-2 py-2 text-left hover:bg-slate-50 disabled:cursor-not-allowed"
            >
              <span className="block text-sm font-medium text-slate-800">{model.displayName}{model.modelId === defaultModelId ? ' · Default' : ''}</span>
              <span className="block text-xs text-slate-500">{model.modelId} · {formatUsd(model.promptUsdPerMillion) ?? '$—'} / 1M in · {formatUsd(model.completionUsdPerMillion) ?? '$—'} / 1M out</span>
            </button>
          </li>)}
        </ul>
      </div>
    </details>
  );
}
