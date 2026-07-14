import {useCallback, useEffect, useRef, useState} from 'react';
import {useNavigate, useParams} from 'react-router-dom';
import {ChatInput} from '../components/chat/ChatInput';
import {ChatMessage} from '../components/chat/ChatMessage';
import {EmptyState} from '../components/chat/EmptyState';
import {DirectoryIndexStatusBadge} from '../components/DirectoryIndexStatusBadge';
import {SessionUsageBadge} from '../components/chat/SessionUsageBadge';
import {ChatSidebar} from '../components/ChatSidebar';
import {LinkedDirectoriesPanel} from '../components/LinkedDirectoriesPanel';
import {Spinner} from '../components/ui/Spinner';
import {chatSessionsApi, directoriesApi} from '../lib/endpoints';
import {formatBytes} from '../lib/format';
import {streamMessageEvents} from '../lib/sse';
import type {
  AgentEvent,
  ChatMessageRecord,
  ChatSession,
  Directory,
  DirectoryIndexStatus,
  LlmUsage,
  SessionFile,
} from '../types';

function normalizeMessage(message: Partial<ChatMessageRecord> & {id: number; sessionId: number; role: 'user' | 'assistant'; content: string; status: ChatMessageRecord['status']}): ChatMessageRecord {
  return {
    ...message,
    steps: message.steps ?? [],
    sources: message.sources ?? [],
    usage: message.usage ?? [],
  };
}

// The live 'done' event's aggregate `stats` (one number per token type,
// summed over every LLM call the turn made) arrives before the DB-backed
// per-call `llm_calls` rows can be refetched. Shown as a single synthetic
// summary row so token/model/duration are visible immediately; a page
// reload replaces it with the real per-call breakdown from the database.
function usageFromStats(stats: Record<string, unknown> | undefined): LlmUsage[] | undefined {
  if (!stats) return undefined;
  const inputTokens = Number(stats.prompt_tokens);
  const outputTokens = Number(stats.completion_tokens);
  if (!Number.isFinite(inputTokens) && !Number.isFinite(outputTokens)) return undefined;
  return [
    {
      id: -1,
      provider: 'gemini',
      model: typeof stats.model === 'string' ? stats.model : undefined,
      purpose: 'turn_summary',
      inputTokens: Number.isFinite(inputTokens) ? inputTokens : 0,
      outputTokens: Number.isFinite(outputTokens) ? outputTokens : 0,
      thinkingTokens: Number.isFinite(Number(stats.thinking_tokens))
        ? Number(stats.thinking_tokens)
        : 0,
      durationMs: Number.isFinite(Number(stats.duration_ms))
        ? Number(stats.duration_ms)
        : undefined,
    },
  ];
}

function previousUserContent(messages: ChatMessageRecord[], index: number): string | undefined {
  for (let i = index - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') return messages[i].content;
  }
  return undefined;
}

export function ChatPage() {
  const {sessionId} = useParams();
  const navigate = useNavigate();
  const selectedId = sessionId ? Number(sessionId) : null;
  const activeStreamRef = useRef<{sessionId: number; messageId: number; controller: AbortController} | null>(null);

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [directories, setDirectories] = useState<Directory[]>([]);
  const [linkedIds, setLinkedIds] = useState<number[]>([]);
  const [visibleFiles, setVisibleFiles] = useState<SessionFile[]>([]);
  const [indexStatuses, setIndexStatuses] = useState<Record<number, DirectoryIndexStatus>>({});
  const [messages, setMessages] = useState<ChatMessageRecord[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [creating, setCreating] = useState(false);
  const [savingLinks, setSavingLinks] = useState(false);
  const [sending, setSending] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([chatSessionsApi.list(), directoriesApi.list()])
      .then(([sessionList, directoryList]) => {
        setSessions(sessionList);
        setDirectories(directoryList);
      })
      .finally(() => setLoadingSessions(false));
  }, []);

  const loadIndexStatuses = useCallback(async (ids: number[]) => {
    if (!ids.length) {
      setIndexStatuses({});
      return;
    }
    const entries = await Promise.all(
      ids.map(async id => {
        try {
          return [id, await directoriesApi.indexStatus(id)] as const;
        } catch {
          return null;
        }
      }),
    );
    setIndexStatuses(
      Object.fromEntries(entries.filter((entry): entry is readonly [number, DirectoryIndexStatus] => Boolean(entry))),
    );
  }, []);

  const loadDetail = useCallback((id: number) => {
    setLoadingDetail(true);
    Promise.all([
      chatSessionsApi.linkedDirectories(id),
      chatSessionsApi.visibleFiles(id),
      chatSessionsApi.messages(id),
    ])
      .then(([linked, files, history]) => {
        const ids = linked.map(d => d.id);
        setLinkedIds(ids);
        setVisibleFiles(files);
        setMessages(history.map(normalizeMessage));
        void loadIndexStatuses(ids);
      })
      .finally(() => setLoadingDetail(false));
  }, [loadIndexStatuses]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (selectedId !== null) loadDetail(selectedId);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selectedId, loadDetail]);

  useEffect(() => {
    return () => {
      activeStreamRef.current?.controller.abort();
      activeStreamRef.current = null;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!linkedIds.length) return;
    const hasIndexing = linkedIds.some(id => indexStatuses[id]?.status === 'indexing');
    if (!hasIndexing) return;
    const timer = window.setInterval(() => {
      void loadIndexStatuses(linkedIds);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [linkedIds, indexStatuses, loadIndexStatuses]);

  async function handleCreate() {
    setCreating(true);
    try {
      const session = await chatSessionsApi.create();
      setSessions(prev => [session, ...prev]);
      navigate(`/chat/${session.id}`);
    } finally {
      setCreating(false);
    }
  }

  async function handleSaveLinks(ids: number[]) {
    if (selectedId === null) return;
    setSavingLinks(true);
    try {
      await chatSessionsApi.setLinkedDirectories(selectedId, ids);
      setLinkedIds(ids);
      const [files] = await Promise.all([
        chatSessionsApi.visibleFiles(selectedId),
        loadIndexStatuses(ids),
      ]);
      setVisibleFiles(files);
    } finally {
      setSavingLinks(false);
    }
  }

  function applyAgentEvent(event: AgentEvent, assistantMessageId: number) {
    if (event.type === 'run_started') {
      setMessages(prev =>
        prev.map(message =>
          message.id === assistantMessageId ? {...message, runId: event.runId} : message,
        ),
      );
      return;
    }

    if (event.type === 'research_step') {
      setMessages(prev =>
        prev.map(message => {
          if (message.id !== assistantMessageId) return message;
          const exists = message.steps.some(
            step => step.id === event.step.id || step.stepId === event.step.stepId,
          );
          return {
            ...message,
            status: message.status === 'pending' ? 'streaming' : message.status,
            steps: exists
              ? message.steps.map(step =>
                  step.id === event.step.id || step.stepId === event.step.stepId
                    ? event.step
                    : step,
                )
              : [...message.steps, event.step],
          };
        }),
      );
      return;
    }

    if (event.type === 'answer_delta') {
      setMessages(prev =>
        prev.map(message =>
          message.id === assistantMessageId
            ? {...message, status: 'streaming', content: `${message.content}${event.text}`}
            : message,
        ),
      );
      return;
    }

    if (event.type === 'source') {
      setMessages(prev =>
        prev.map(message => {
          if (message.id !== assistantMessageId) return message;
          const exists = message.sources.some(source => source.id === event.source.id);
          return {
            ...message,
            sources: exists
              ? message.sources.map(source =>
                  source.id === event.source.id ? event.source : source,
                )
              : [...message.sources, event.source],
          };
        }),
      );
      return;
    }

    if (event.type === 'done') {
      const usage = usageFromStats(event.stats);
      setMessages(prev =>
        prev.map(message =>
          message.id === assistantMessageId
            ? {
                ...message,
                status: 'completed',
                content: event.content,
                usage: usage ?? message.usage,
              }
            : message,
        ),
      );
      // Session-level totals live only in the sessions list, not per
      // message — update them from this turn's stats the same way the
      // message's own usage footer just was, instead of waiting for a
      // full session-list refetch to reflect the chat's new state.
      const stats = event.stats;
      const turnTotalTokens = Number(stats?.total_tokens);
      const contextUsageRatio = stats?.context_usage_ratio;
      if (Number.isFinite(turnTotalTokens) || typeof contextUsageRatio === 'number') {
        setSessions(prev =>
          prev.map(session =>
            session.id === selectedId
              ? {
                  ...session,
                  totalTokens: Number.isFinite(turnTotalTokens)
                    ? (session.totalTokens ?? 0) + turnTotalTokens
                    : session.totalTokens,
                  lastContextUsageRatio:
                    typeof contextUsageRatio === 'number'
                      ? contextUsageRatio
                      : session.lastContextUsageRatio,
                }
              : session,
          ),
        );
      }
      return;
    }

    if (event.type === 'cancelled') {
      setMessages(prev =>
        prev.map(message =>
          message.id === assistantMessageId ? {...message, status: 'cancelled'} : message,
        ),
      );
      return;
    }

    if (event.type === 'error') {
      setStreamError(event.message);
      setMessages(prev =>
        prev.map(message =>
          message.id === assistantMessageId
            ? {...message, status: 'error', runId: event.runId ?? message.runId}
            : message,
        ),
      );
    }
  }

  async function sendMessage(content: string) {
    if (selectedId === null || sending) return;
    setSending(true);
    setStreamError(null);
    const controller = new AbortController();
    let assistantMessageId: number | undefined;
    try {
      const created = await chatSessionsApi.sendMessage(selectedId, {content});
      assistantMessageId = created.messageId;
      const userMessage = normalizeMessage(created.userMessage);
      const assistantMessage = normalizeMessage(created.assistantMessage);
      setMessages(prev => [...prev, userMessage, assistantMessage]);
      setSessions(prev =>
        prev.map(session =>
          session.id === selectedId && (!session.title || session.title === 'New chat')
            ? {...session, title: content.length > 60 ? `${content.slice(0, 57)}...` : content}
            : session,
        ),
      );
      activeStreamRef.current = {
        sessionId: selectedId,
        messageId: created.messageId,
        controller,
      };
      await streamMessageEvents({
        sessionId: selectedId,
        messageId: created.messageId,
        signal: controller.signal,
        onEvent: event => applyAgentEvent(event, created.messageId),
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        setStreamError(error instanceof Error ? error.message : String(error));
        if (assistantMessageId !== undefined) {
          setMessages(prev =>
            prev.map(message =>
              message.id === assistantMessageId ? {...message, status: 'error'} : message,
            ),
          );
        }
      }
    } finally {
      if (activeStreamRef.current?.controller === controller) {
        activeStreamRef.current = null;
      }
      setSending(false);
    }
  }

  // Continues a run that errored out or was manually stopped, instead of
  // starting a brand-new one (Regenerate) — reuses core-api's still-held
  // agent state (chat history, tool results already gathered) via the
  // run_id captured off that message's earlier 'run_started' event, so
  // work already done/paid for isn't thrown away.
  async function continueMessage(assistantMessageId: number) {
    if (selectedId === null || sending) return;
    const target = messages.find(message => message.id === assistantMessageId);
    if (!target?.runId) return;
    setSending(true);
    setStreamError(null);
    const controller = new AbortController();
    setMessages(prev =>
      prev.map(message =>
        message.id === assistantMessageId ? {...message, status: 'streaming'} : message,
      ),
    );
    activeStreamRef.current = {sessionId: selectedId, messageId: assistantMessageId, controller};
    try {
      await streamMessageEvents({
        sessionId: selectedId,
        messageId: assistantMessageId,
        signal: controller.signal,
        resumeRunId: target.runId,
        onEvent: event => applyAgentEvent(event, assistantMessageId),
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        setStreamError(error instanceof Error ? error.message : String(error));
        setMessages(prev =>
          prev.map(message =>
            message.id === assistantMessageId ? {...message, status: 'error'} : message,
          ),
        );
      }
    } finally {
      if (activeStreamRef.current?.controller === controller) {
        activeStreamRef.current = null;
      }
      setSending(false);
    }
  }

  async function stopStream() {
    const active = activeStreamRef.current;
    if (!active) return;
    active.controller.abort();
    await chatSessionsApi.cancelMessage(active.sessionId, active.messageId).catch(() => {
      // The stream may already have closed; local abort is still authoritative.
    });
    setMessages(prev =>
      prev.map(message =>
        message.id === active.messageId ? {...message, status: 'cancelled'} : message,
      ),
    );
    setSending(false);
  }

  const selectedSession = sessions.find(s => s.id === selectedId) ?? null;

  if (loadingSessions) {
    return (
      <div className="flex flex-1 items-center justify-center text-slate-400">
        <Spinner />
      </div>
    );
  }

  return (
    <>
      <ChatSidebar
        sessions={sessions}
        selectedId={selectedId}
        onSelect={id => navigate(`/chat/${id}`)}
        onCreate={handleCreate}
        creating={creating}
      />
      <main className="flex flex-1 flex-col overflow-y-auto">
        {!selectedSession ? (
          <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
            <p className="text-lg font-medium text-slate-700">Pick a chat, or start a new one</p>
            <p className="mt-1 max-w-sm text-sm text-slate-400">
              Each chat only sees the directories you explicitly link to it.
            </p>
          </div>
        ) : (
          <div className="mx-auto flex h-full w-full max-w-6xl flex-1 flex-col gap-6 px-6 py-6 lg:flex-row">
            {loadingDetail ? (
              <div className="flex flex-1 justify-center py-10 text-slate-400">
                <Spinner />
              </div>
            ) : (
              <>
                <section className="flex min-w-0 flex-1 flex-col">
                  <div className="mb-4">
                    <h1 className="text-xl font-semibold text-slate-900">
                      {selectedSession.title || 'Untitled chat'}
                    </h1>
                    <SessionUsageBadge session={selectedSession} />
                    {streamError && (
                      <p className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">
                        {streamError}
                      </p>
                    )}
                  </div>

                  <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
                    {messages.length === 0 ? (
                      <EmptyState hasDirectories={linkedIds.length > 0} />
                    ) : (
                      messages.map((message, index) => (
                        <ChatMessage
                          key={message.id}
                          message={message}
                          previousUserContent={previousUserContent(messages, index)}
                          onStop={() => void stopStream()}
                          onRegenerate={content => void sendMessage(content)}
                          onContinue={() => void continueMessage(message.id)}
                        />
                      ))
                    )}
                  </div>

                  <ChatInput
                    disabled={linkedIds.length === 0}
                    disabledReason={
                      linkedIds.length === 0 ? 'Link at least one directory first.' : undefined
                    }
                    sending={sending}
                    onSend={content => void sendMessage(content)}
                    onStop={() => void stopStream()}
                  />
                </section>

                <aside className="w-full shrink-0 space-y-5 lg:w-80">
                  <LinkedDirectoriesPanel
                    allDirectories={directories}
                    linkedIds={linkedIds}
                    onSave={handleSaveLinks}
                    saving={savingLinks}
                    indexStatuses={indexStatuses}
                  />

                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <h3 className="mb-3 text-sm font-semibold text-slate-700">
                      Visible to this chat ({visibleFiles.length})
                    </h3>
                    {visibleFiles.length === 0 ? (
                      <p className="text-sm text-slate-400">Nothing yet - link a directory above.</p>
                    ) : (
                      <ul className="max-h-64 space-y-1.5 overflow-y-auto pr-1">
                        {visibleFiles.map(file => (
                          <li
                            key={file.id}
                            className="flex items-center justify-between rounded-lg px-2 py-1.5 text-sm"
                          >
                            <span className="truncate text-slate-700">{file.name}</span>
                            <span className="ml-3 shrink-0 text-xs text-slate-400">
                              {formatBytes(file.sizeBytes)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  {linkedIds.length > 0 && (
                    <div className="rounded-2xl border border-slate-200 bg-white p-4">
                      <h3 className="mb-3 text-sm font-semibold text-slate-700">
                        Index status
                      </h3>
                      <ul className="space-y-2">
                        {linkedIds.map(id => {
                          const directory = directories.find(dir => dir.id === id);
                          return (
                            <li
                              key={id}
                              className="flex items-center justify-between gap-3 rounded-lg bg-slate-50 px-2 py-1.5 text-sm"
                            >
                              <span className="min-w-0 truncate text-slate-600">
                                {directory?.name ?? `Directory ${id}`}
                              </span>
                              <DirectoryIndexStatusBadge status={indexStatuses[id]} />
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </aside>
              </>
            )}
          </div>
        )}
      </main>
    </>
  );
}
