import {useState} from 'react';
import type {ChatMessageRecord} from '../../types';
import {Button} from '../ui/Button';
import {ResearchActivityPanel} from './ResearchActivityPanel';
import {SourceList} from './SourceList';
import {StreamingAnswer} from './StreamingAnswer';
import {UsageFooter} from './UsageFooter';

export function AssistantMessage({
  message,
  onStop,
  onRegenerate,
  onContinue,
}: {
  message: ChatMessageRecord;
  onStop: () => void;
  onRegenerate: () => void;
  onContinue: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const streaming = message.status === 'streaming' || message.status === 'pending';
  const content = message.content || (streaming ? 'Preparing research...' : '');
  // core-api keeps an interrupted run's agent state (chat history, tool
  // results already gathered) around for a while — Continue picks that up
  // instead of throwing it away and starting over like Regenerate does.
  const canContinue =
    !streaming &&
    (message.status === 'error' || message.status === 'cancelled') &&
    Boolean(message.runId);

  async function copy() {
    if (!message.content) return;
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <article className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <ResearchActivityPanel steps={message.steps} />
      <StreamingAnswer content={content} streaming={streaming} sources={message.sources} />
      {message.status === 'error' && (
        <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">
          The answer could not be completed.
          {canContinue ? ' Click Continue to pick up where it left off.' : ''}
        </p>
      )}
      {message.status === 'cancelled' && (
        <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-500">
          This response was stopped.
          {canContinue ? ' Click Continue to pick up where it left off.' : ''}
        </p>
      )}
      <SourceList sources={message.sources} />
      <UsageFooter usage={message.usage} />
      <div className="mt-4 flex flex-wrap items-center gap-2">
        {streaming ? (
          <Button variant="secondary" className="px-3 py-1.5 text-xs" onClick={onStop}>
            Stop
          </Button>
        ) : canContinue ? (
          <>
            <Button variant="primary" className="px-3 py-1.5 text-xs" onClick={onContinue}>
              Continue
            </Button>
            <Button variant="ghost" className="px-3 py-1.5 text-xs" onClick={onRegenerate}>
              Regenerate
            </Button>
          </>
        ) : (
          <Button variant="secondary" className="px-3 py-1.5 text-xs" onClick={onRegenerate}>
            Regenerate
          </Button>
        )}
        <Button
          variant="ghost"
          className="px-3 py-1.5 text-xs"
          onClick={() => void copy()}
          disabled={!message.content}
        >
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </div>
    </article>
  );
}
