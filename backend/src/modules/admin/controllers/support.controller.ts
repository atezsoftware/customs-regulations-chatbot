import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {
  get,
  HttpErrors,
  param,
  Request,
  response,
  RestBindings,
} from '@loopback/rest';
import {getCurrentUser, requireAdmin} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {ChatMessage} from '../../chat/models';
import {
  ChatMessageRepository,
  ChatResearchStepRepository,
  ChatSessionRepository,
  ChatSourceRepository,
  LlmCallRepository,
} from '../../chat/repositories';

function toSafeMessage(message: ChatMessage) {
  return {
    id: message.id,
    sessionId: message.sessionId,
    role: message.role,
    content: message.content ?? '',
    errorMessage: message.errorMessage ?? null,
    status: message.status,
    createdAt: message.createdAt,
    updatedAt: message.updatedAt,
    steps: [],
    sources: [],
    usage: [],
  };
}

export class AdminSupportController {
  constructor(
    @repository(UserRepository) private userRepository: UserRepository,
    @repository(ChatSessionRepository) private chatSessionRepository: ChatSessionRepository,
    @repository(ChatMessageRepository) private chatMessageRepository: ChatMessageRepository,
    @repository(ChatResearchStepRepository)
    private chatResearchStepRepository: ChatResearchStepRepository,
    @repository(ChatSourceRepository) private chatSourceRepository: ChatSourceRepository,
    @repository(LlmCallRepository) private llmCallRepository: LlmCallRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  @get('/admin/support/sessions')
  @response(200, {description: 'Admin view of user chat sessions'})
  async sessions(
    @param.query.string('search') search?: string,
    @param.query.number('limit') limit?: number,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const trimmedSearch = search?.trim() ?? '';
    const cappedLimit = Math.min(Math.max(Math.floor(limit ?? 50), 1), 100);
    const rows = await this.chatSessionRepository.findAdminSessions(
      trimmedSearch,
      cappedLimit,
    );

    return {
      sessions: rows.map(row => ({
        id: numberValue(row.id),
        title: row.title ?? 'Untitled chat',
        createdAt: dateString(row.created_at),
        updatedAt: dateString(row.updated_at),
        user: {
          id: numberValue(row.user_id),
          email: row.email,
          fullName: row.full_name ?? undefined,
          role: row.role,
        },
        messageCount: numberValue(row.message_count),
        totalTokens: numberValue(row.total_tokens),
        lastMessageAt: dateString(row.last_message_at),
        lastMessage: row.last_message_preview
          ? {
              role: row.last_message_role,
              status: row.last_message_status,
              preview: row.last_message_preview,
            }
          : undefined,
      })),
    };
  }

  @get('/admin/support/sessions/{id}/messages')
  @response(200, {description: 'Admin view of a user chat session and messages'})
  async messages(@param.path.number('id') sessionId: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const session = await this.chatSessionRepository.findById(sessionId).catch(() => null);
    if (!session) {
      throw new HttpErrors.NotFound('Chat session not found.');
    }
    const sessionUser = await this.userRepository.findById(session.userId);

    const messages = await this.chatMessageRepository.find({
      where: {sessionId},
      order: ['createdAt ASC', 'id ASC'],
    });
    const messageIds = messages.map(message => message.id!).filter(Boolean);
    const [steps, sources, usage] = await Promise.all([
      messageIds.length
        ? this.chatResearchStepRepository.find({
            where: {messageId: {inq: messageIds}},
            order: ['createdAt ASC', 'id ASC'],
          })
        : [],
      messageIds.length
        ? this.chatSourceRepository.find({
            where: {messageId: {inq: messageIds}},
            order: ['createdAt ASC', 'id ASC'],
          })
        : [],
      messageIds.length
        ? this.llmCallRepository.find({
            where: {messageId: {inq: messageIds}},
            order: ['createdAt ASC', 'id ASC'],
          })
        : [],
    ]);

    return {
      session: {
        id: numberValue(session.id),
        title: session.title ?? 'Untitled chat',
        createdAt: dateString(session.createdAt),
        updatedAt: dateString(session.updatedAt),
        user: {
          id: numberValue(sessionUser.id),
          email: sessionUser.email,
          fullName: sessionUser.fullName ?? undefined,
          role: sessionUser.role,
        },
      },
      messages: messages.map(message => ({
        ...toSafeMessage(message),
        steps: steps
          .filter(step => step.messageId === message.id)
          .map(step => ({
            id: step.id,
            stepId: step.stepKey,
            status: step.status,
            title: step.title,
            preview: step.preview,
            details: step.details,
            metadata: step.metadata,
            createdAt: step.createdAt,
            completedAt: step.completedAt,
          })),
        sources: sources
          .filter(source => source.messageId === message.id)
          .map(source => ({
            id: source.id,
            title: source.title,
            snippet: source.snippet,
            url: source.url,
            filePath: source.filePath,
            page: source.page,
            chunkId: source.chunkId,
            score: source.score,
            createdAt: source.createdAt,
          })),
        usage: usage
          .filter(call => call.messageId === message.id)
          .map(call => ({
            id: call.id,
            provider: call.provider,
            model: call.model,
            purpose: call.purpose,
            inputTokens: call.inputTokens,
            outputTokens: call.outputTokens,
            thinkingTokens: call.thinkingTokens,
            durationMs: call.durationMs,
            createdAt: call.createdAt,
          })),
      })),
    };
  }
}

function numberValue(value: number | string | null | undefined): number {
  return Number(value ?? 0);
}

function dateString(value: string | Date | null | undefined): string | undefined {
  if (!value) return undefined;
  return value instanceof Date ? value.toISOString() : value;
}
