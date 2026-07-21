import {useState} from 'react';
import {Button} from '../ui/Button';
import {ModelSelector} from './ModelSelector';
import type {LlmModelOption} from '../../types';

export function ChatInput({
  disabled,
  disabledReason,
  sending,
  onSend,
  onStop,
  models = [],
  modelId,
  defaultModelId,
  onModelChange,
}: {
  disabled?: boolean;
  disabledReason?: string;
  sending: boolean;
  onSend: (content: string) => void;
  onStop: () => void;
  models?: LlmModelOption[];
  modelId?: string;
  defaultModelId?: string;
  onModelChange?: (model: LlmModelOption) => void;
}) {
  const [draft, setDraft] = useState('');

  function submit() {
    const content = draft.trim();
    if (!content || disabled || sending) return;
    setDraft('');
    onSend(content);
  }

  return (
    <div className="border-t border-slate-200 bg-slate-50/80 px-1 pt-4">
      <div className="rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
        <textarea
          value={draft}
          onChange={event => setDraft(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          disabled={disabled || sending}
          rows={3}
          placeholder={disabledReason ?? 'Ask about the linked customs documents...'}
          className="block w-full resize-none rounded-lg border-0 px-3 py-2 text-sm text-slate-800 outline-none placeholder:text-slate-400 disabled:bg-white disabled:text-slate-300"
        />
        <div className="flex items-center justify-between gap-3 px-1 pb-1">
          <div className="flex min-w-0 items-center gap-2">
            {onModelChange && models.length > 0 && <ModelSelector models={models} modelId={modelId} defaultModelId={defaultModelId} disabled={sending} onChange={onModelChange} />}
            <p className="text-xs text-slate-400">
            {sending ? 'Research is running.' : 'Enter sends, Shift+Enter adds a line.'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {sending && (
              <Button type="button" variant="secondary" onClick={onStop}>
                Stop
              </Button>
            )}
            <Button
              type="button"
              onClick={submit}
              disabled={disabled || sending || !draft.trim()}
            >
              Send
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
