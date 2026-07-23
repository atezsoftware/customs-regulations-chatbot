import {useEffect, useState} from 'react';
import {adminSupportApi} from '../lib/endpoints';
import type {
  AdminSupportSession,
  AdminSupportSessionDetail,
  ChatMessageRecord,
} from '../types';

export function AdminSupportPage() {
  const [sessions, setSessions] = useState<AdminSupportSession[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<AdminSupportSessionDetail | null>(null);
  const [search, setSearch] = useState('');
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadSessions(query = search) {
    setLoadingList(true);
    setError(null);
    try {
      const data = await adminSupportApi.sessions({search: query, limit: 75});
      setSessions(data.sessions);
      setSelectedId(current => current ?? data.sessions[0]?.id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingList(false);
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadSessions('');
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (selectedId === null) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoadingDetail(true);
    adminSupportApi
      .messages(selectedId)
      .then(data => {
        if (!cancelled) setDetail(data);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  return (
    <div className="flex min-h-0 flex-1">
      <aside className="flex w-96 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-100 px-4 py-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Admin support
          </p>
          <h1 className="mt-1 text-lg font-semibold text-slate-900">User conversations</h1>
          <form
            className="mt-4 flex gap-2"
            onSubmit={event => {
              event.preventDefault();
              void loadSessions(search);
            }}
          >
            <input
              value={search}
              onChange={event => setSearch(event.target.value)}
              placeholder="Search user or chat"
              className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none transition focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100"
            />
            <button
              type="submit"
              className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-700"
            >
              Search
            </button>
          </form>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {loadingList ? (
            <p className="px-2 py-6 text-sm text-slate-400">Loading conversations...</p>
          ) : sessions.length === 0 ? (
            <p className="px-2 py-6 text-sm text-slate-400">No conversations found.</p>
          ) : (
            <ul className="space-y-1">
              {sessions.map(session => (
                <li key={session.id}>
                  <button
                    type="button"
                    onClick={() => setSelectedId(session.id)}
                    className={`w-full rounded-xl border px-3 py-3 text-left transition-colors ${
                      session.id === selectedId
                        ? 'border-indigo-200 bg-indigo-50'
                        : 'border-transparent hover:border-slate-200 hover:bg-slate-50'
                    }`}
                  >
                    <span className="flex items-center justify-between gap-3">
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-semibold text-slate-800">
                          {session.title}
                        </span>
                        <span className="mt-0.5 block truncate text-xs text-slate-500">
                          {session.user.fullName ?? session.user.email}
                        </span>
                      </span>
                      <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                        {formatNumber(session.totalTokens)}
                      </span>
                    </span>
                    {session.lastMessage?.preview && (
                      <span className="mt-2 block truncate text-xs text-slate-400">
                        {session.lastMessage.preview}
                      </span>
                    )}
                    {session.model && (
                      <span className="mt-2 block truncate text-xs font-medium text-indigo-600">
                        {formatModel(session.model.provider, session.model.modelId)}
                      </span>
                    )}
                    <span className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                      <span>{session.messageCount} messages</span>
                      <span>{formatDate(session.lastMessageAt ?? session.updatedAt)}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-y-auto bg-slate-50">
        <div className="mx-auto max-w-5xl px-6 py-6">
          {error && (
            <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {error}
            </div>
          )}

          {selectedId === null ? (
            <EmptyState title="Pick a conversation" text="Select a chat to inspect its history." />
          ) : loadingDetail || !detail ? (
            <div className="rounded-xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-400">
              Loading conversation...
            </div>
          ) : (
            <>
              <header className="mb-5 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      {detail.session.user.email}
                    </p>
                    <h2 className="mt-1 truncate text-xl font-semibold text-slate-900">
                      {detail.session.title}
                    </h2>
                    <p className="mt-1 text-sm text-slate-500">
                      {detail.session.user.fullName ?? 'Unnamed user'} ·{' '}
                      {detail.messages.length} messages
                    </p>
                    {detail.session.model && (
                      <p className="mt-1 text-sm font-medium text-indigo-600">
                        Model: {formatModel(detail.session.model.provider, detail.session.model.modelId)}
                      </p>
                    )}
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <MiniMetric label="Input" value={formatNumber(tokenSum(detail.messages, 'inputTokens'))} />
                    <MiniMetric label="Output" value={formatNumber(tokenSum(detail.messages, 'outputTokens'))} />
                    <MiniMetric label="Thinking" value={formatNumber(tokenSum(detail.messages, 'thinkingTokens'))} />
                  </div>
                </div>
              </header>

              <div className="space-y-3">
                {detail.messages.length === 0 ? (
                  <EmptyState title="No messages" text="This chat has no stored messages yet." />
                ) : (
                  detail.messages.map(message => (
                    <SupportMessage key={message.id} message={message} />
                  ))
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function SupportMessage({message}: {message: ChatMessageRecord}) {
  const totalTokens = message.usage.reduce(
    (sum, usage) => sum + usage.inputTokens + usage.outputTokens + usage.thinkingTokens,
    0,
  );
  const isUser = message.role === 'user';

  return (
    <article
      className={`rounded-xl border p-4 shadow-sm ${
        isUser
          ? 'ml-auto max-w-[82%] border-indigo-500 bg-indigo-600 text-white'
          : 'border-slate-200 bg-white text-slate-700'
      }`}
    >
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className={isUser ? 'font-semibold text-indigo-100' : 'font-semibold text-slate-400'}>
          {isUser ? 'User' : 'Assistant'} · {message.status}
        </span>
        <span className={isUser ? 'text-indigo-100' : 'text-slate-400'}>
          {formatDate(message.createdAt)}
        </span>
      </div>
      <p className="whitespace-pre-wrap text-sm leading-6">
        {message.content || (message.errorMessage ? `Error: ${message.errorMessage}` : 'No content')}
      </p>
      {message.errorMessage && message.content && (
        <p className="mt-2 whitespace-pre-wrap rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">
          Error: {message.errorMessage}
        </p>
      )}
      {!isUser && (
        <div className="mt-4 flex flex-wrap gap-2 text-xs">
          <Badge label={`${message.steps.length} steps`} />
          <Badge label={`${message.sources.length} sources`} />
          <Badge label={`${formatNumber(totalTokens)} tokens`} />
        </div>
      )}
      {!isUser && message.sources.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {message.sources.slice(0, 4).map(source => (
            <div key={source.id} className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
              <span className="font-medium text-slate-700">{source.title}</span>
              {source.snippet && <span className="mt-0.5 block line-clamp-2">{source.snippet}</span>}
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function MiniMetric({label, value}: {label: string; value: string}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="mt-0.5 text-sm font-semibold text-slate-800">{value}</p>
    </div>
  );
}

function Badge({label}: {label: string}) {
  return <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-500">{label}</span>;
}

function EmptyState({title, text}: {title: string; text: string}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
      <p className="text-sm font-semibold text-slate-700">{title}</p>
      <p className="mt-1 text-sm text-slate-400">{text}</p>
    </div>
  );
}

function tokenSum(
  messages: ChatMessageRecord[],
  key: 'inputTokens' | 'outputTokens' | 'thinkingTokens',
): number {
  return messages.reduce(
    (sum, message) => sum + message.usage.reduce((usageSum, usage) => usageSum + usage[key], 0),
    0,
  );
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(Math.round(value));
}

function formatDate(value?: string): string {
  if (!value) return 'No date';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatModel(provider: string, modelId: string): string {
  return `${provider} · ${modelId}`;
}
