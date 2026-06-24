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
import {PostgresDataSource} from '../../../datasources';
import {UserRepository} from '../../auth/repositories';
import {ChatMessage} from '../../chat/models';
import {
  ChatMessageRepository,
  ChatResearchStepRepository,
  ChatSourceRepository,
  LlmCallRepository,
} from '../../chat/repositories';

interface AdminSessionRow {
  id: number | string;
  title: string | null;
  created_at: string | Date | null;
  updated_at: string | Date | null;
  user_id: number | string;
  email: string;
  full_name: string | null;
  role: string;
  message_count: number | string;
  total_tokens: number | string;
  last_message_at: string | Date | null;
  last_message_role: string | null;
  last_message_status: string | null;
  last_message_preview: string | null;
}

interface AdminSessionDetailRow {
  id: number | string;
  title: string | null;
  created_at: string | Date | null;
  updated_at: string | Date | null;
  user_id: number | string;
  email: string;
  full_name: string | null;
  role: string;
}

function toSafeMessage(message: ChatMessage) {
  return {
    id: message.id,
    sessionId: message.sessionId,
    role: message.role,
    content: message.content ?? '',
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
    @repository(ChatMessageRepository) private chatMessageRepository: ChatMessageRepository,
    @repository(ChatResearchStepRepository)
    private chatResearchStepRepository: ChatResearchStepRepository,
    @repository(ChatSourceRepository) private chatSourceRepository: ChatSourceRepository,
    @repository(LlmCallRepository) private llmCallRepository: LlmCallRepository,
    @inject('datasources.postgres') private dataSource: PostgresDataSource,
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
    const rows = (await this.dataSource.execute(
      `
        SELECT
          s.id,
          s.title,
          s.created_at,
          s.updated_at,
          u.id AS user_id,
          u.email,
          u.full_name,
          u.role,
          COALESCE(message_stats.message_count, 0) AS message_count,
          COALESCE(token_stats.total_tokens, 0) AS total_tokens,
          message_stats.last_message_at,
          last_message.role AS last_message_role,
          last_message.status AS last_message_status,
          last_message.content AS last_message_preview
        FROM chat_sessions s
        JOIN users u ON u.id = s.user_id
        LEFT JOIN (
          SELECT
            session_id,
            COUNT(id) AS message_count,
            MAX(created_at) AS last_message_at
          FROM chat_messages
          GROUP BY session_id
        ) message_stats ON message_stats.session_id = s.id
        LEFT JOIN (
          SELECT
            session_id,
            SUM(input_tokens + output_tokens + thinking_tokens) AS total_tokens
          FROM llm_calls
          GROUP BY session_id
        ) token_stats ON token_stats.session_id = s.id
        LEFT JOIN LATERAL (
          SELECT role, status, left(content, 220) AS content
          FROM chat_messages lm
          WHERE lm.session_id = s.id
          ORDER BY lm.created_at DESC, lm.id DESC
          LIMIT 1
        ) last_message ON true
        WHERE ($1 = '' OR u.email ILIKE $2 OR u.full_name ILIKE $2 OR s.title ILIKE $2)
        ORDER BY COALESCE(message_stats.last_message_at, s.updated_at, s.created_at) DESC
        LIMIT $3
      `,
      [trimmedSearch, `%${trimmedSearch}%`, cappedLimit],
    )) as AdminSessionRow[];

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

    const sessionRows = (await this.dataSource.execute(
      `
        SELECT
          s.id,
          s.title,
          s.created_at,
          s.updated_at,
          u.id AS user_id,
          u.email,
          u.full_name,
          u.role
        FROM chat_sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.id = $1
      `,
      [sessionId],
    )) as AdminSessionDetailRow[];
    const session = sessionRows[0];
    if (!session) {
      throw new HttpErrors.NotFound('Chat session not found.');
    }

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
        createdAt: dateString(session.created_at),
        updatedAt: dateString(session.updated_at),
        user: {
          id: numberValue(session.user_id),
          email: session.email,
          fullName: session.full_name ?? undefined,
          role: session.role,
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
