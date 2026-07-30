import {WebSocket} from 'undici';
import {ChatEffortLevel, ChatMessage, ChatResearchStep, ChatSource} from '../models';
import {
  ChatMessageRepository,
  ChatResearchStepRepository,
  ChatSessionDirectoryRepository,
  ChatSessionRepository,
  ChatSourceRepository,
  LlmCallRepository,
} from '../repositories';
import {virtualCorpusKey} from '../../directories/services';
import {DirectoryFileRepository, DirectoryRepository} from '../../directories/repositories';
import {stripNulBytes} from '../../../common/text';
import {formatChatError} from '../../../common/chat-error';

const DEFAULT_CORE_URL = 'ws://127.0.0.1:8000/ws/explore';
const DEFAULT_PROVIDER = 'openrouter';
const DEFAULT_MODEL = 'google/gemini-3.6-flash';

type JsonObject = Record<string, unknown>;

interface CoreEvent {
  type: string;
  data?: JsonObject;
}

interface SessionView {
  viewFolder: string;
  directoryIds: number[];
  indexFolders: string[];
}

interface IndexedHit {
  directoryId: number;
  docId: string;
  chunkId?: string;
  relativePath: string;
  absolutePath: string;
  position?: number;
  chunkType?: string;
  metadata: JsonObject;
  text: string;
  score: number;
  citationLabel: string;
  chunkPath: string;
}

type IndexedHitBase = Omit<IndexedHit, 'citationLabel' | 'chunkPath'>;

export interface ChatHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

export interface BridgeInput {
  sessionId: number;
  assistantMessageId: number;
  task: string;
  conversationContext: ChatHistoryItem[];
  signal?: AbortSignal;
  // When set, continues a run core-api is still holding onto as resumable
  // (see core-api's runs.py) instead of starting a brand-new one — sent as
  // `{"type": "resume", "run_id": ...}` instead of the usual task payload.
  resumeRunId?: string;
  // Retrieval breadth the user picked for this message (see agent.py's
  // EffortLevel) — only meaningful on a fresh run; a resumed run keeps
  // whatever effort core-api already captured for it.
  effort?: ChatEffortLevel;
}

export interface AgentResearchStep {
  id: number;
  stepId: string;
  status: string;
  title: string;
  preview?: string;
  details?: string;
  metadata?: object;
  createdAt?: string;
  completedAt?: string;
}

export interface AgentSource {
  id: number;
  title: string;
  snippet?: string;
  url?: string;
  filePath?: string;
  page?: number;
  chunkId?: string;
  score?: number;
  createdAt?: string;
}

export interface AgentLlmUsage {
  id: number;
  provider: string;
  model?: string;
  purpose: string;
  step?: number;
  agentRole?: string;
  taskId?: string;
  agentId?: string;
  sequence?: number;
  inputTokens: number;
  outputTokens: number;
  thinkingTokens: number;
  durationMs?: number;
  billedCostUsd?: string;
  costSource?: 'provider' | 'estimated';
  createdAt?: string;
}

export type AgentEvent =
  | {type: 'message_created'; messageId: number}
  | {type: 'run_started'; runId: string; resumed: boolean}
  | {type: 'research_step'; step: AgentResearchStep}
  | {type: 'answer_delta'; text: string}
  | {type: 'source'; source: AgentSource}
  | {type: 'llm_call'; usage: AgentLlmUsage}
  | {
      type: 'done';
      messageId: number;
      content: string;
      stats?: JsonObject;
      incomplete?: boolean;
      resumable?: boolean;
      unresolvedInformation?: string[];
    }
  | {type: 'cancelled'; messageId: number}
  | {type: 'error'; message: string; runId?: string};

interface QueueItem {
  event?: CoreEvent;
  error?: Error;
  closed?: boolean;
}

class AsyncQueue {
  private items: QueueItem[] = [];
  private waiters: Array<(item: QueueItem) => void> = [];

  push(item: QueueItem) {
    const waiter = this.waiters.shift();
    if (waiter) waiter(item);
    else this.items.push(item);
  }

  next(): Promise<QueueItem> {
    const item = this.items.shift();
    if (item) return Promise.resolve(item);
    return new Promise(resolve => this.waiters.push(resolve));
  }
}

export class CoreBridgeService {
  constructor(
    private chatSessionRepository: ChatSessionRepository,
    private chatSessionDirectoryRepository: ChatSessionDirectoryRepository,
    private chatMessageRepository: ChatMessageRepository,
    private chatResearchStepRepository: ChatResearchStepRepository,
    private chatSourceRepository: ChatSourceRepository,
    private llmCallRepository: LlmCallRepository,
    private directoryRepository: DirectoryRepository,
    private directoryFileRepository: DirectoryFileRepository,
  ) {}

  async *streamAssistantResponse(input: BridgeInput): AsyncGenerator<AgentEvent> {
    const startedAt = Date.now();
    await this.chatMessageRepository.updateById(input.assistantMessageId, {
      status: 'streaming',
      errorMessage: null,
      updatedAt: new Date().toISOString(),
    });

    yield {type: 'message_created', messageId: input.assistantMessageId};

    const session = await this.chatSessionRepository.findById(input.sessionId);
    const sessionView = await this.buildSessionView(input.sessionId);
    // Not injected into the agent's task — the agent decides for itself whether
    // to call semantic_search. Kept only to resolve citation labels in the final
    // answer to chunk-backed sources (snippet/score) once the agent has responded.
    const indexedHits = await this.searchLinkedDirectories(
      sessionView.directoryIds,
      input.task,
    );
    const ws = new WebSocket(resolveCoreWebSocketUrl());
    const queue = new AsyncQueue();
    let finalContent = '';
    let lastRunningStepKey: string | undefined;
    // Multi-agent progress events overlap by design. Track those rows
    // independently from the legacy, strictly sequential status/tool
    // stream so one worker finishing cannot close a sibling's live step.
    const activeAgentStepKeys = new Set<string>();
    const agentStepKeysByCorrelation = new Map<string, string>();
    const pendingAgentRetrievalMetadata = new Map<string, Record<string, unknown>>();
    let statusCounter = 0;
    let closedByAbort = false;
    let terminalEventSeen = false;
    let llmCallsRecorded = 0;
    // Captured from core-api's `start` event so a later error/disconnect
    // can be reported back with the run_id the caller needs to resume it.
    let capturedRunId: string | undefined;

    const closeSocket = () => {
      try {
        ws.close();
      } catch {
        // The socket may already be closed; nothing else to do.
      }
    };
    const abortHandler = () => {
      closedByAbort = true;
      closeSocket();
      queue.push({closed: true});
    };
    input.signal?.addEventListener('abort', abortHandler, {once: true});

    try {
      await waitForOpen(ws, input.signal);
      ws.addEventListener('message', message => {
        try {
          const raw = typeof message.data === 'string' ? message.data : String(message.data);
          queue.push({event: JSON.parse(raw) as CoreEvent});
        } catch (error) {
          queue.push({error: error instanceof Error ? error : new Error(String(error))});
        }
      });
      ws.addEventListener('error', () => {
        queue.push({error: new Error('Core WebSocket connection failed.')});
      });
      ws.addEventListener('close', () => {
        queue.push({closed: true});
      });

      ws.send(
        JSON.stringify(
          input.resumeRunId
            ? {
                type: 'resume',
                run_id: input.resumeRunId,
                internal_token: process.env.CORE_INTERNAL_TOKEN,
              }
            : {
                task: input.task,
                folder: sessionView.viewFolder,
                use_index: sessionView.indexFolders.length > 0,
                index_folders: sessionView.indexFolders,
                database_url: resolveDatabaseUrl(),
                enable_semantic: sessionView.indexFolders.length > 0,
                enable_metadata: sessionView.indexFolders.length > 0,
                conversation_context: input.conversationContext,
                internal_token: process.env.CORE_INTERNAL_TOKEN,
                provider: session.llmProvider,
                model: session.model,
                temperature: session.temperature,
                effort: input.effort ?? 'low',
              },
        ),
      );

      while (true) {
        const item = await queue.next();
        if (item.closed) break;
        if (item.error) throw item.error;
        if (!item.event) continue;

        const event = item.event;
        const data = event.data ?? {};

        if (event.type === 'start') {
          capturedRunId = text(data.run_id) || undefined;
          if (capturedRunId) {
            yield {type: 'run_started', runId: capturedRunId, resumed: Boolean(data.resumed)};
          }
          continue;
        }

        if (event.type === 'research_progress') {
          const progress = parseResearchProgress(data);
          if (!progress) continue;
          let step = await this.saveStep({
            messageId: input.assistantMessageId,
            ...progress,
          });
          const correlationKey = researchProgressCorrelationKey(progress);
          if (correlationKey) {
            agentStepKeysByCorrelation.set(correlationKey, progress.stepKey);
            const pendingMetadata = pendingAgentRetrievalMetadata.get(correlationKey);
            if (pendingMetadata) {
              step =
                (await this.mergeStepMetadata(
                  input.assistantMessageId,
                  progress.stepKey,
                  pendingMetadata,
                )) ?? step;
              pendingAgentRetrievalMetadata.delete(correlationKey);
            }
          }
          if (progress.status === 'running') {
            activeAgentStepKeys.add(progress.stepKey);
          } else {
            activeAgentStepKeys.delete(progress.stepKey);
          }
          yield {type: 'research_step', step};
          continue;
        }

        if (event.type === 'status') {
          const stepKey = `status-${++statusCounter}`;
          if (lastRunningStepKey) {
            yield {
              type: 'research_step',
              step: await this.completeStep(input.assistantMessageId, lastRunningStepKey),
            };
          }
          const step = await this.saveStep({
            messageId: input.assistantMessageId,
            stepKey,
            status: 'running',
            title: text(data.label, 'Thinking'),
            preview: text(data.detail),
            metadata: {},
          });
          lastRunningStepKey = stepKey;
          yield {type: 'research_step', step};
          continue;
        }

        if (event.type === 'tool_call') {
          const stepKey = `tool-${text(data.step, String(++statusCounter))}`;
          if (lastRunningStepKey) {
            yield {
              type: 'research_step',
              step: await this.completeStep(input.assistantMessageId, lastRunningStepKey),
            };
          }
          const title = text(data.status_label, text(data.tool_name, 'Using tool'));
          const step = await this.saveStep({
            messageId: input.assistantMessageId,
            stepKey,
            status: 'running',
            title,
            preview: text(data.status_detail),
            details: text(data.reason),
            metadata: {},
          });
          lastRunningStepKey = stepKey;
          yield {type: 'research_step', step};
          continue;
        }

        if (event.type === 'retrieval_stats' || event.type === 'retrieval_done') {
          // Legacy calls correlate through `data.step` to a tool row.
          // Multi-agent searches have no tool row, so their task/agent/
          // sequence tuple targets the parallel search progress row instead.
          const step = await this.mergeRetrievalMetadata({
            messageId: input.assistantMessageId,
            data,
            agentStepKeysByCorrelation,
            pendingAgentRetrievalMetadata,
          });
          if (step) yield {type: 'research_step', step};
          continue;
        }

        if (event.type === 'go_deeper') {
          const stepKey = `go-deeper-${text(data.step, String(++statusCounter))}`;
          if (lastRunningStepKey) {
            yield {
              type: 'research_step',
              step: await this.completeStep(input.assistantMessageId, lastRunningStepKey),
            };
          }
          const step = await this.saveStep({
            messageId: input.assistantMessageId,
            stepKey,
            status: 'running',
            title: 'Inspecting folder',
            preview: text(data.directory),
            details: text(data.reason),
            metadata: {directory: data.directory},
          });
          lastRunningStepKey = stepKey;
          yield {type: 'research_step', step};
          continue;
        }

        if (event.type === 'ask_human') {
          if (lastRunningStepKey) {
            yield {
              type: 'research_step',
              step: await this.completeStep(input.assistantMessageId, lastRunningStepKey),
            };
            lastRunningStepKey = undefined;
          }
          const step = await this.saveStep({
            messageId: input.assistantMessageId,
            stepKey: `ask-${text(data.step, String(++statusCounter))}`,
            status: 'error',
            title: 'Needs clarification',
            preview: text(data.question),
            details: text(data.reason),
            metadata: {question: data.question},
          });
          yield {type: 'research_step', step};
          throw new Error('Core requested human clarification, which is not supported in SSE chat yet.');
        }

        if (event.type === 'llm_call') {
          // One row per actual LLM call (tool-planning steps and the
          // final-answer generation are separate calls), not just one
          // end-of-run aggregate — lets the dashboard show per-message
          // token/model/duration instead of only a session-wide total.
          llmCallsRecorded += 1;
          const provider =
            text(data.provider) ||
            session.llmProvider ||
            process.env.FS_EXPLORER_LLM_PROVIDER ||
            DEFAULT_PROVIDER;
          const model =
            text(data.model) ||
            session.model ||
            process.env.FS_EXPLORER_LLM_MODEL ||
            DEFAULT_MODEL;
          const purpose = text(data.purpose, 'action');
          const step = numberValue(data.step);
          const agentRole = text(data.agent_role) || undefined;
          const taskId = text(data.task_id) || undefined;
          const agentId = text(data.agent_id) || undefined;
          const sequence = optionalInteger(data.sequence);
          const inputTokens = numberValue(data.prompt_tokens);
          const outputTokens = numberValue(data.completion_tokens);
          const thinkingTokens = numberValue(data.thinking_tokens);
          const durationMs = numberValue(data.duration_ms);
          const billedCostUsd = text(data.billed_cost_usd) || undefined;
          const costSource =
            text(data.cost_source) === 'provider' || text(data.cost_source) === 'estimated'
              ? (text(data.cost_source) as 'provider' | 'estimated')
              : undefined;
          const created = await this.llmCallRepository.create({
            messageId: input.assistantMessageId,
            sessionId: input.sessionId,
            provider,
            model,
            purpose,
            step,
            inputTokens,
            outputTokens,
            thinkingTokens,
            durationMs,
            generationId: text(data.generation_id) || undefined,
            cachedInputTokens: numberValue(data.cached_input_tokens),
            cacheWriteTokens: numberValue(data.cache_write_tokens),
            billedCostUsd,
            upstreamCostUsd: text(data.upstream_cost_usd) || undefined,
            costSource,
          });
          // Previously persisted-only — the live run never showed per-call
          // token growth, only a single post-hoc total once `done` arrived.
          yield {
            type: 'llm_call',
            usage: {
              id: created.id ?? -1,
              provider,
              model,
              purpose,
              step,
              agentRole,
              taskId,
              agentId,
              sequence,
              inputTokens,
              outputTokens,
              thinkingTokens,
              durationMs,
              billedCostUsd,
              costSource,
              createdAt: created.createdAt,
            },
          };
          continue;
        }

        if (event.type === 'answer_start') {
          if (lastRunningStepKey) {
            yield {
              type: 'research_step',
              step: await this.completeStep(input.assistantMessageId, lastRunningStepKey),
            };
            lastRunningStepKey = undefined;
          }
          for (const step of await this.completeRunningAgentSteps(
            input.assistantMessageId,
            activeAgentStepKeys,
          )) {
            yield {type: 'research_step', step};
          }
          activeAgentStepKeys.clear();
          agentStepKeysByCorrelation.clear();
          pendingAgentRetrievalMetadata.clear();
          continue;
        }

        if (event.type === 'answer_delta') {
          const delta = text(data.text);
          if (!delta) continue;
          finalContent += delta;
          await this.chatMessageRepository.updateById(input.assistantMessageId, {
            content: finalContent,
            updatedAt: new Date().toISOString(),
          });
          yield {type: 'answer_delta', text: delta};
          continue;
        }

        if (event.type === 'answer_done') {
          finalContent = text(data.final_result, finalContent);
          await this.chatMessageRepository.updateById(input.assistantMessageId, {
            content: finalContent,
            updatedAt: new Date().toISOString(),
          });
          for (const source of await this.persistSources(
            input.assistantMessageId,
            data,
            indexedHits,
            finalContent,
          )) {
            yield {type: 'source', source};
          }
          continue;
        }

        if (event.type === 'complete') {
          terminalEventSeen = true;
          if (lastRunningStepKey) {
            yield {
              type: 'research_step',
              step: await this.completeStep(input.assistantMessageId, lastRunningStepKey),
            };
            lastRunningStepKey = undefined;
          }
          for (const step of await this.completeRunningAgentSteps(
            input.assistantMessageId,
            activeAgentStepKeys,
          )) {
            yield {type: 'research_step', step};
          }
          activeAgentStepKeys.clear();
          agentStepKeysByCorrelation.clear();
          pendingAgentRetrievalMetadata.clear();

          const error = text(data.error);
          const stats = objectOrUndefined(data.stats);
          // Normally superseded by the per-call 'llm_call' rows recorded
          // above (real per-call durations from core-api). Only fall back
          // to one aggregate row here if none arrived — e.g. an older
          // core-api that doesn't emit 'llm_call' yet, or a run that
          // produced usage without any incremental events.
          if (stats && llmCallsRecorded === 0) {
            await this.llmCallRepository.create({
              messageId: input.assistantMessageId,
              sessionId: input.sessionId,
              provider:
                session.llmProvider ||
                process.env.FS_EXPLORER_LLM_PROVIDER ||
                DEFAULT_PROVIDER,
              model: session.model || process.env.FS_EXPLORER_LLM_MODEL || DEFAULT_MODEL,
              purpose: 'chat_completion',
              inputTokens: numberValue(stats.prompt_tokens),
              outputTokens: numberValue(stats.completion_tokens),
              thinkingTokens: numberValue(stats.thinking_tokens),
              durationMs: Date.now() - startedAt,
            });
          }

          // How full the context window was on this turn's last LLM call
          // (0-1). Stored on the session (not derived from llm_calls) so
          // the chat UI can show it without an extra aggregate query.
          // Guarded on typeof rather than just `stats &&` so an older
          // core-api that doesn't send this field yet leaves the
          // session's existing value alone instead of zeroing it out.
          if (stats && typeof stats.context_usage_ratio === 'number') {
            await this.chatSessionRepository.updateById(input.sessionId, {
              lastContextUsageRatio: decimalValue(stats.context_usage_ratio),
            });
          }

          if (error) throw new Error(error);

          finalContent = text(data.final_result, finalContent);
          await this.chatMessageRepository.updateById(input.assistantMessageId, {
            content: finalContent,
            status: 'completed',
            errorMessage: null,
            updatedAt: new Date().toISOString(),
          });
          yield {
            type: 'done',
            messageId: input.assistantMessageId,
            content: finalContent,
            stats,
            // The run hit its step-budget safety net instead of a real
            // conclusion — technically not an `error`, but the answer may
            // be an unsatisfying apology. Lets the UI offer "Continue" even
            // on this otherwise-successful completion.
            incomplete: data.incomplete === true,
            resumable: data.resumable === true,
            unresolvedInformation: Array.isArray(data.unresolved_information)
              ? data.unresolved_information.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
          };
          break;
        }

        if (event.type === 'error') {
          terminalEventSeen = true;
          throw new Error(text(data.message, 'Core returned an error.'));
        }
      }

      if (closedByAbort || input.signal?.aborted) {
        await this.chatMessageRepository.updateById(input.assistantMessageId, {
          status: 'cancelled',
          updatedAt: new Date().toISOString(),
        });
        yield {type: 'cancelled', messageId: input.assistantMessageId};
      } else if (!terminalEventSeen) {
        throw new Error('Core stream closed before completion.');
      }
    } catch (error) {
      const message = formatChatError(error);
      const status = input.signal?.aborted || closedByAbort ? 'cancelled' : 'error';
      await this.chatMessageRepository.updateById(input.assistantMessageId, {
        status,
        ...(status === 'error' ? {errorMessage: message} : {}),
        updatedAt: new Date().toISOString(),
      });
      if (status === 'cancelled') {
        yield {type: 'cancelled', messageId: input.assistantMessageId};
      } else {
        yield {type: 'error', message, runId: capturedRunId};
      }
    } finally {
      input.signal?.removeEventListener('abort', abortHandler);
      closeSocket();
    }
  }

  private async buildSessionView(sessionId: number): Promise<SessionView> {
    const links = await this.chatSessionDirectoryRepository.find({where: {sessionId}});
    const directoryIds = links.map(link => link.directoryId);
    if (!directoryIds.length) {
      throw new Error('Link at least one directory before sending a message.');
    }

    const indexFolders = directoryIds.map(directoryId => virtualCorpusKey(directoryId));
    return {
      viewFolder: indexFolders[0] ?? `/__customs_regulations__/sessions/${sessionId}`,
      directoryIds,
      indexFolders,
    };
  }

  private async searchLinkedDirectories(
    directoryIds: number[],
    query: string,
  ): Promise<IndexedHit[]> {
    const endpoint = `${resolveCoreRestUrl()}/api/search`;
    const headers: Record<string, string> = {'Content-Type': 'application/json'};
    if (process.env.CORE_INTERNAL_TOKEN) {
      headers['X-Internal-Token'] = process.env.CORE_INTERNAL_TOKEN;
    }

    const settled = await Promise.allSettled(
      directoryIds.map(async directoryId => {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            corpus_folder: virtualCorpusKey(directoryId),
            query,
            limit: 4,
            database_url: resolveDatabaseUrl(),
          }),
        });
        if (!res.ok) return [];
        const data = (await res.json()) as {hits?: Array<Record<string, unknown>>};
        return (data.hits ?? []).map(hit => {
          const indexedHit: IndexedHitBase = {
            directoryId,
            docId: text(hit.doc_id),
            chunkId: text(hit.chunk_id) || undefined,
            relativePath: text(hit.relative_path),
            absolutePath: text(hit.absolute_path),
            position: optionalInteger(hit.position),
            chunkType: text(hit.chunk_type) || undefined,
            metadata: objectOrUndefined(hit.metadata) ?? {},
            text: cleanSnippet(text(hit.text), 1200),
            score: decimalValue(hit.score),
          };
          const chunkPath = chunkPathForHit(indexedHit);
          return {
            ...indexedHit,
            chunkPath,
            citationLabel: citationLabelForHit(indexedHit, chunkPath),
          };
        });
      }),
    );

    return settled
      .flatMap(result => (result.status === 'fulfilled' ? result.value : []))
      .filter(hit => hit.text)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
  }

  private async saveStep(input: {
    messageId: number;
    stepKey: string;
    status: 'pending' | 'running' | 'completed' | 'error';
    title: string;
    preview?: string;
    details?: string;
    metadata?: object;
  }): Promise<AgentResearchStep> {
    const existing = await this.chatResearchStepRepository.findOne({
      where: {messageId: input.messageId, stepKey: input.stepKey},
    });
    const data = {
      status: input.status,
      title: input.title,
      preview: input.preview,
      details: input.details,
      metadata: input.metadata ?? {},
      completedAt:
        input.status === 'completed' || input.status === 'error'
          ? new Date().toISOString()
          : undefined,
    };
    if (existing?.id) {
      await this.chatResearchStepRepository.updateById(existing.id, {
        ...data,
        // Lifecycle updates share one row. Preserve retrieval telemetry
        // merged after the started event when the completed event arrives.
        metadata: {
          ...(objectOrUndefined(existing.metadata) ?? {}),
          ...(input.metadata ?? {}),
        },
      });
      return this.toAgentStep(await this.chatResearchStepRepository.findById(existing.id));
    }
    return this.toAgentStep(
      await this.chatResearchStepRepository.create({
        messageId: input.messageId,
        stepKey: input.stepKey,
        ...data,
      }),
    );
  }

  private async completeStep(messageId: number, stepKey: string): Promise<AgentResearchStep> {
    const existing = await this.chatResearchStepRepository.findOne({where: {messageId, stepKey}});
    if (!existing?.id) {
      return this.saveStep({
        messageId,
        stepKey,
        status: 'completed',
        title: 'Completed',
        metadata: {},
      });
    }
    await this.chatResearchStepRepository.updateById(existing.id, {
      status: 'completed',
      completedAt: new Date().toISOString(),
    });
    return this.toAgentStep(await this.chatResearchStepRepository.findById(existing.id));
  }

  /**
   * Close both agent steps observed on this socket and running agent rows
   * left by an interrupted socket. The database lookup is the recovery
   * path: a resumed stream starts with an empty in-memory active set.
   */
  private async completeRunningAgentSteps(
    messageId: number,
    activeStepKeys: ReadonlySet<string>,
  ): Promise<AgentResearchStep[]> {
    const runningSteps = await this.chatResearchStepRepository.find({
      where: {messageId, status: 'running'},
    });
    const stepKeys = new Set(activeStepKeys);
    for (const step of runningSteps) {
      if (step.stepKey.startsWith('agent-')) stepKeys.add(step.stepKey);
    }

    const completed: AgentResearchStep[] = [];
    for (const stepKey of stepKeys) {
      completed.push(await this.completeStep(messageId, stepKey));
    }
    return completed;
  }

  /**
   * Attach retrieval telemetry to a multi-agent search lifecycle row when
   * correlation fields are present; otherwise retain the legacy tool-step
   * behavior. A pending map tolerates Core sending the side-channel stats
   * just before its corresponding progress event reaches this consumer.
   */
  private async mergeRetrievalMetadata(input: {
    messageId: number;
    data: JsonObject;
    agentStepKeysByCorrelation: Map<string, string>;
    pendingAgentRetrievalMetadata: Map<string, Record<string, unknown>>;
  }): Promise<AgentResearchStep | undefined> {
    const metadata = retrievalMetadata(input.data);
    const correlationKey = agentRetrievalCorrelationKey(input.data);
    if (!correlationKey) {
      return this.mergeStepMetadata(
        input.messageId,
        `tool-${text(input.data.step)}`,
        metadata,
      );
    }

    let stepKey = input.agentStepKeysByCorrelation.get(correlationKey);
    if (!stepKey) {
      const runningSteps = await this.chatResearchStepRepository.find({
        where: {messageId: input.messageId, status: 'running'},
      });
      stepKey = runningSteps.find(step => {
        if (!step.stepKey.startsWith('agent-')) return false;
        const stepMetadata = objectOrUndefined(step.metadata);
        return (
          stepMetadata?.kind === 'search_started' &&
          researchMetadataCorrelationKey(stepMetadata) === correlationKey
        );
      })?.stepKey;
      if (stepKey) input.agentStepKeysByCorrelation.set(correlationKey, stepKey);
    }

    if (!stepKey) {
      input.pendingAgentRetrievalMetadata.set(correlationKey, metadata);
      return undefined;
    }
    return this.mergeStepMetadata(input.messageId, stepKey, metadata);
  }

  // Merges into (rather than replacing) a step's metadata — used for
  // 'retrieval_stats', which arrives as a separate event after the
  // 'tool_call' event that already created this step's row with `{}`
  // metadata. No-ops if the step isn't found: retrieval_stats always
  // follows a real tool_call in practice, but a mismatch here (e.g. a
  // stale client) should be silently dropped, not create a phantom step.
  private async mergeStepMetadata(
    messageId: number,
    stepKey: string,
    metadata: Record<string, unknown>,
  ): Promise<AgentResearchStep | undefined> {
    const existing = await this.chatResearchStepRepository.findOne({where: {messageId, stepKey}});
    if (!existing?.id) return undefined;
    const merged = {...(existing.metadata as object | undefined), ...metadata};
    await this.chatResearchStepRepository.updateById(existing.id, {metadata: merged});
    return this.toAgentStep(await this.chatResearchStepRepository.findById(existing.id));
  }

  private async persistSources(
    messageId: number,
    data: JsonObject,
    indexedHits: IndexedHit[],
    finalContent: string,
  ): Promise<AgentSource[]> {
    const evidenceSources = Array.isArray(data.evidence_sources) ? data.evidence_sources : [];
    const citedSources = Array.isArray(data.cited_sources) ? data.cited_sources : [];
    const links = objectOrUndefined(data.cited_source_links) ?? {};
    const persisted: AgentSource[] = [];
    const seen = new Set<string>();
    // Kept separately from chunk-level deduplication so several cited
    // locators from one exact document can all be persisted while the
    // later bare-title legacy fallback is still suppressed.
    const exactTitleKeys = new Set<string>();

    // New multi-agent runs return the exact evidence selected during
    // synthesis. Persist that authoritative provenance first; the legacy
    // original-query pre-search below remains only as a compatibility
    // fallback for older core-api versions and non-indexed citations.
    for (const rawSource of evidenceSources) {
      const source = objectOrUndefined(rawSource);
      if (!source) continue;
      const title = text(source.title);
      const locator = text(source.locator);
      const chunkId = text(source.chunk_id) || undefined;
      const snippet = text(source.snippet) || undefined;
      if (!title && !chunkId) continue;
      if (title) exactTitleKeys.add(normalizeCitation(title));

      const displayTitle = evidenceSourceTitle(title || 'Indexed document', locator);
      const sourceKey = normalizeCitation(chunkId || displayTitle);
      if (!sourceKey || seen.has(sourceKey)) continue;
      seen.add(sourceKey);
      seen.add(normalizeCitation(displayTitle));
      if (chunkId) seen.add(normalizeCitation(chunkId));

      const record = await this.chatSourceRepository.create({
        messageId,
        title: displayTitle,
        snippet,
        chunkId,
        score: optionalDecimal(source.score),
      });
      persisted.push(this.toAgentSource(record));
    }

    for (const citation of extractCitationLabels(finalContent)) {
      if (seen.has(normalizeCitation(citation))) continue;
      const hit = findHitForCitation(citation, indexedHits);
      if (!hit?.chunkId) continue;
      const key = normalizeCitation(hit.chunkId);
      if (seen.has(key)) continue;
      seen.add(key);
      seen.add(normalizeCitation(citation));
      const record = await this.chatSourceRepository.create({
        messageId,
        title: citation,
        snippet: hit.text,
        chunkId: hit.chunkId,
        score: hit.score,
      });
      persisted.push(this.toAgentSource(record));
    }

    for (const source of citedSources) {
      const title = text(source);
      if (!title) continue;
      const key = normalizeCitation(title);
      if (seen.has(key) || exactTitleKeys.has(key)) continue;
      const hit = findHitForCitation(title, indexedHits);
      if (hit?.chunkId) {
        const chunkKey = normalizeCitation(hit.chunkId);
        if (seen.has(chunkKey)) continue;
        seen.add(key);
        seen.add(chunkKey);
        const record = await this.chatSourceRepository.create({
          messageId,
          title: hit.citationLabel,
          snippet: hit.text,
          chunkId: hit.chunkId,
          score: hit.score,
        });
        persisted.push(this.toAgentSource(record));
        continue;
      }
      seen.add(key);
      const rawUrl = text(links[title]);
      const filePath = filePathFromCoreDocumentUrl(rawUrl);
      const record = await this.chatSourceRepository.create({
        messageId,
        title,
        filePath,
        url: filePath ? undefined : rawUrl || undefined,
      });
      persisted.push(this.toAgentSource(record));
    }

    return persisted;
  }

  private toAgentStep(step: ChatResearchStep): AgentResearchStep {
    return {
      id: step.id!,
      stepId: step.stepKey,
      status: step.status,
      title: step.title,
      preview: step.preview,
      details: step.details,
      metadata: step.metadata,
      createdAt: step.createdAt,
      completedAt: step.completedAt,
    };
  }

  private toAgentSource(source: ChatSource): AgentSource {
    return {
      id: source.id!,
      title: source.title,
      snippet: source.snippet,
      url: source.url,
      filePath: source.filePath,
      page: source.page,
      chunkId: source.chunkId,
      score: source.score,
      createdAt: source.createdAt,
    };
  }
}

function resolveCoreWebSocketUrl(): string {
  const configured = process.env.CORE_INTERNAL_URL ?? DEFAULT_CORE_URL;
  if (configured.endsWith('/ws/explore')) {
    return configured.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
  }
  const base = configured.replace(/\/$/, '').replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
  return `${base}/ws/explore`;
}

function resolveCoreRestUrl(): string {
  const configured = process.env.CORE_INTERNAL_URL ?? DEFAULT_CORE_URL;
  return configured
    .replace(/\/ws\/explore$/, '')
    .replace(/^ws:/, 'http:')
    .replace(/^wss:/, 'https:')
    .replace(/\/$/, '');
}

function resolveDatabaseUrl(): string | undefined {
  if (process.env.DATABASE_URL) return process.env.DATABASE_URL;
  const host = process.env.DB_HOST;
  const user = process.env.DB_USER;
  const password = process.env.DB_PASSWORD;
  const database = process.env.DB_NAME;
  if (!host || !user || !database) return undefined;
  const port = process.env.DB_PORT ?? '5432';
  return `postgresql://${encodeURIComponent(user)}:${encodeURIComponent(
    password ?? '',
  )}@${host}:${port}/${database}`;
}

function chunkPathForHit(hit: IndexedHitBase | IndexedHit): string {
  const title = displaySourceTitle(hit.absolutePath || hit.relativePath) || 'Indexed document';
  const locator = locatorForHit(hit);
  return locator ? `${title} - ${locator}` : `${title} - Chunk ${(hit.position ?? 0) + 1}`;
}

function citationLabelForHit(hit: IndexedHitBase | IndexedHit, chunkPath?: string): string {
  const title = displaySourceTitle(hit.absolutePath || hit.relativePath) || 'Indexed document';
  const locator = locatorForHit(hit);
  if (locator) return `${title}, ${locator}`;
  return chunkPath ?? `${title}, Chunk ${(hit.position ?? 0) + 1}`;
}

function locatorForHit(hit: IndexedHitBase | IndexedHit): string {
  const articleNo = metadataText(hit.metadata, 'article_no');
  const paragraphNo = metadataText(hit.metadata, 'paragraph_no');
  const clauseLabel =
    metadataText(hit.metadata, 'clause_label') || metadataText(hit.metadata, 'clause_no');
  const subclauseLabel = metadataText(hit.metadata, 'subclause_label');
  if (articleNo) {
    return [
      `Madde ${articleNo}${formatLocatorSuffix(paragraphNo)}`,
      clauseLabel,
      subclauseLabel,
    ]
      .filter(Boolean)
      .join(' ');
  }

  const appendixLabel = metadataText(hit.metadata, 'appendix_label');
  if (appendixLabel) return `Ek ${appendixLabel}`;

  const headingPath = hit.metadata.heading_path;
  if (Array.isArray(headingPath) && headingPath.length) {
    return headingPath
      .slice(-2)
      .map(item => text(item))
      .filter(Boolean)
      .join(' > ');
  }

  const tableIndex = metadataText(hit.metadata, 'table_index');
  if (tableIndex) return `Tablo ${tableIndex}`;

  return hit.position === undefined ? '' : `Chunk ${hit.position + 1}`;
}

function metadataText(metadata: JsonObject, key: string): string {
  return text(metadata[key]).trim();
}

function formatLocatorSuffix(value: string): string {
  if (!value) return '';
  if (value.startsWith('(')) return value;
  if (/^[A-Za-z0-9ÇĞİÖŞÜçğıöşü]+$/.test(value)) return `(${value})`;
  return ` ${value}`;
}

function extractCitationLabels(content: string): string[] {
  const labels: string[] = [];
  const seen = new Set<string>();
  const matches = content.matchAll(/\[([^\]\n]+)\]/g);
  for (const match of matches) {
    for (const rawPart of match[1].split(/\s*;\s*/)) {
      const label = cleanCitationLabel(rawPart);
      const key = normalizeCitation(label);
      if (!label || seen.has(key)) continue;
      seen.add(key);
      labels.push(label);
    }
  }
  return labels;
}

function findHitForCitation(citation: string, hits: IndexedHit[]): IndexedHit | undefined {
  const normalized = normalizeCitation(citation);
  if (!normalized) return undefined;

  // Full-equality match first. A hit whose label only *contains* the
  // citation (checked next) must never outrank one that matches it exactly.
  const equalMatch = hits.find(hit =>
    [hit.citationLabel, hit.chunkPath].some(label => normalized === normalizeCitation(label)),
  );
  if (equalMatch) return equalMatch;

  // Among hits whose label is a substring of the citation, prefer the
  // longest (most specific) label rather than the first in score order. A
  // broader locator like "Madde 54" is a substring of a more specific
  // citation like "Madde 54(1)", so without this a same-article hit for a
  // *different* paragraph could shadow the paragraph the citation actually
  // refers to just because it scored higher.
  const bestContainment = hits
    .map(hit => {
      const bestLabelLength = [hit.citationLabel, hit.chunkPath]
        .map(normalizeCitation)
        .filter(label => label && normalized.includes(label))
        .reduce((max, label) => Math.max(max, label.length), 0);
      return bestLabelLength > 0 ? {hit, bestLabelLength} : undefined;
    })
    .filter((candidate): candidate is {hit: IndexedHit; bestLabelLength: number} =>
      Boolean(candidate),
    )
    .sort((a, b) => b.bestLabelLength - a.bestLabelLength)[0]?.hit;
  if (bestContainment) return bestContainment;

  return hits
    .map(hit => ({hit, matchScore: citationMatchScore(normalized, hit)}))
    .filter(candidate => candidate.matchScore > 0)
    .sort((a, b) => b.matchScore - a.matchScore || b.hit.score - a.hit.score)[0]?.hit;
}

function citationMatchScore(normalizedCitation: string, hit: IndexedHit): number {
  let score = 0;
  const title = normalizeCitation(displaySourceTitle(hit.absolutePath || hit.relativePath));
  const locator = normalizeCitation(locatorForHit(hit));
  const label = normalizeCitation(hit.citationLabel);
  const chunkPath = normalizeCitation(hit.chunkPath);

  if (label && normalizedCitation.includes(label)) score += 10;
  if (chunkPath && normalizedCitation.includes(chunkPath)) score += 10;
  if (title && (normalizedCitation.includes(title) || title.includes(normalizedCitation))) {
    score += 4;
  }
  const titleOverlap = tokenOverlap(normalizedCitation, title);
  if (titleOverlap >= 3) score += titleOverlap;
  if (locator && normalizedCitation.includes(locator)) score += 6;

  const articleNo = normalizeCitation(metadataText(hit.metadata, 'article_no'));
  if (articleNo && normalizedCitation.includes(articleNo)) score += 2;

  return score;
}

function tokenOverlap(a: string, b: string): number {
  const ignored = new Set(['source', 'sources', 'madde', 'article', 'section', 'chunk']);
  const aTokens = new Set(a.split(' ').filter(token => token && !ignored.has(token)));
  const bTokens = new Set(b.split(' ').filter(token => token && !ignored.has(token)));
  let overlap = 0;
  for (const token of aTokens) {
    if (bTokens.has(token)) overlap += 1;
  }
  return overlap;
}

function cleanCitationLabel(value: string): string {
  return value
    .trim()
    .replace(/^\[/, '')
    .replace(/\]$/, '')
    .replace(/^source:\s*/i, '')
    .trim();
}

function normalizeCitation(value: string): string {
  return cleanCitationLabel(value)
    .replace(/ı/g, 'i')
    .replace(/İ/g, 'I')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function waitForOpen(ws: WebSocket, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error('Request was cancelled.'));
      return;
    }
    const cleanup = () => {
      ws.removeEventListener('open', onOpen);
      ws.removeEventListener('error', onError);
      ws.removeEventListener('close', onClose);
      signal?.removeEventListener('abort', onAbort);
    };
    const onOpen = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error('Core WebSocket connection failed.'));
    };
    const onAbort = () => {
      cleanup();
      reject(new Error('Request was cancelled.'));
    };
    const onClose = () => {
      cleanup();
      reject(new Error('Core WebSocket closed before it was ready.'));
    };
    ws.addEventListener('open', onOpen);
    ws.addEventListener('error', onError);
    ws.addEventListener('close', onClose);
    signal?.addEventListener('abort', onAbort, {once: true});
  });
}

function text(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return stripNulBytes(value);
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return fallback;
}

function numberValue(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.trunc(value);
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
  }
  return 0;
}

function decimalValue(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}

function optionalDecimal(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function optionalInteger(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.trunc(value);
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
  }
  return undefined;
}

function definedObject(values: Record<string, unknown>): JsonObject {
  return Object.fromEntries(
    Object.entries(values).filter(([, value]) => value !== undefined && value !== ''),
  );
}

/**
 * Normalizes core-api's multi-agent lifecycle event into the existing
 * research-step persistence contract. The event id is deliberately the
 * sole key input: started/completed/failed updates must address the same
 * row even when several agents interleave their events.
 */
export function parseResearchProgress(data: JsonObject):
  | {
      stepKey: string;
      status: 'running' | 'completed' | 'error';
      title: string;
      preview?: string;
      metadata: JsonObject;
    }
  | undefined {
  const eventId = text(data.event_id);
  const status = researchProgressStatus(data.status);
  // Without both fields we cannot safely upsert a parallel step, so a
  // malformed event is intentionally ignored rather than assigned a
  // sequence-derived key that could collide on resume.
  if (!eventId || !status) return undefined;

  const agentRole = text(data.agent_role) || undefined;
  return {
    stepKey: `agent-${eventId}`,
    status,
    title: text(data.label) || researchProgressFallbackLabel(agentRole, data.kind),
    preview: text(data.detail) || undefined,
    metadata: definedObject({
      eventId,
      kind: text(data.kind) || undefined,
      sequence: optionalInteger(data.sequence),
      taskId: text(data.task_id) || undefined,
      agentId: text(data.agent_id) || undefined,
      agentRole,
    }),
  };
}

function researchProgressCorrelationKey(progress: {
  status: string;
  metadata: JsonObject;
}): string | undefined {
  if (progress.status !== 'running' || progress.metadata.kind !== 'search_started') {
    return undefined;
  }
  return researchMetadataCorrelationKey(progress.metadata);
}

function researchMetadataCorrelationKey(metadata: JsonObject): string | undefined {
  return agentCorrelationKey(
    text(metadata.taskId) || undefined,
    text(metadata.agentId) || undefined,
    optionalInteger(metadata.sequence),
  );
}

function agentRetrievalCorrelationKey(data: JsonObject): string | undefined {
  return agentCorrelationKey(
    text(data.task_id) || undefined,
    text(data.agent_id) || undefined,
    optionalInteger(data.sequence),
  );
}

function agentCorrelationKey(
  taskId: string | undefined,
  agentId: string | undefined,
  sequence: number | undefined,
): string | undefined {
  if (!taskId || !agentId || sequence === undefined) return undefined;
  return JSON.stringify([taskId, agentId, sequence]);
}

function retrievalMetadata(data: JsonObject): Record<string, unknown> {
  return definedObject({
    chunkCount: numberValue(data.chunk_count),
    retrievalChars: numberValue(data.chars),
    retrievalTokensEstimated: numberValue(data.estimated_tokens),
    retrievalToolName: text(data.tool_name) || undefined,
    retrievalSequence: optionalInteger(data.sequence),
  });
}

function researchProgressStatus(
  value: unknown,
): 'running' | 'completed' | 'error' | undefined {
  switch (text(value)) {
    case 'started':
      return 'running';
    case 'completed':
      return 'completed';
    case 'failed':
      return 'error';
    default:
      return undefined;
  }
}

function researchProgressFallbackLabel(agentRole: string | undefined, kind: unknown): string {
  const label = agentRole || text(kind);
  if (!label) return 'Researching';
  return label
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase());
}

function evidenceSourceTitle(title: string, locator: string): string {
  if (!locator) return title;
  const normalizedTitle = normalizeCitation(title);
  const normalizedLocator = normalizeCitation(locator);
  return normalizedLocator && normalizedTitle.includes(normalizedLocator)
    ? title
    : `${title}, ${locator}`;
}

function cleanSnippet(value: string, maxChars: number): string {
  const cleaned = value.replace(/\s+/g, ' ').trim();
  if (cleaned.length <= maxChars) return cleaned;
  return `${cleaned.slice(0, maxChars)}...`;
}

function displaySourceTitle(value: string): string {
  const fileName = value.replace(/\\/g, '/').split('/').pop() || value;
  return fileName
    .replace(/^\d+-/, '')
    .replace(/\.[a-z0-9]+$/i, '')
    .replace(/_x1/gi, '(')
    .replace(/x2_/gi, ')_')
    .replace(/x2/gi, ')')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function objectOrUndefined(value: unknown): JsonObject | undefined {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as JsonObject;
  }
  return undefined;
}

function filePathFromCoreDocumentUrl(value: string): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value, 'http://core.local');
    return url.searchParams.get('path') ?? undefined;
  } catch {
    return undefined;
  }
}
