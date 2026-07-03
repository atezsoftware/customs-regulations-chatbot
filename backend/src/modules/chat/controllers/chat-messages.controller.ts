import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {get, HttpErrors, param, post, Request, requestBody, response, Response, RestBindings} from '@loopback/rest';
import {getCurrentUser} from '../../../common/auth';
import {stripNulBytes} from '../../../common/text';
import {UserRepository} from '../../auth/repositories';
import {DirectoryFileRepository, DirectoryRepository} from '../../directories/repositories';
import {ChatMessage} from '../models';
import {
  ChatMessageRepository,
  ChatResearchStepRepository,
  ChatSessionDirectoryRepository,
  ChatSessionRepository,
  ChatSourceRepository,
  LlmCallRepository,
} from '../repositories';
import {AgentEvent, ChatHistoryItem, CoreBridgeService} from '../services';

interface SendMessageBody {
  content: string;
  model?: string;
  temperature?: number;
}

const activeStreams = new Map<number, AbortController>();

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

function sseFrame(event: AgentEvent): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

export class ChatMessagesController {
  constructor(
    @repository(ChatSessionRepository) private chatSessionRepository: ChatSessionRepository,
    @repository(ChatSessionDirectoryRepository)
    private chatSessionDirectoryRepository: ChatSessionDirectoryRepository,
    @repository(ChatMessageRepository) private chatMessageRepository: ChatMessageRepository,
    @repository(ChatResearchStepRepository)
    private chatResearchStepRepository: ChatResearchStepRepository,
    @repository(ChatSourceRepository) private chatSourceRepository: ChatSourceRepository,
    @repository(LlmCallRepository) private llmCallRepository: LlmCallRepository,
    @repository(DirectoryRepository) private directoryRepository: DirectoryRepository,
    @repository(DirectoryFileRepository)
    private directoryFileRepository: DirectoryFileRepository,
    @repository(UserRepository) private userRepository: UserRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
    @inject(RestBindings.Http.RESPONSE) private res: Response,
  ) {}

  private bridge() {
    return new CoreBridgeService(
      this.chatSessionRepository,
      this.chatSessionDirectoryRepository,
      this.chatMessageRepository,
      this.chatResearchStepRepository,
      this.chatSourceRepository,
      this.llmCallRepository,
      this.directoryRepository,
      this.directoryFileRepository,
    );
  }

  private async ownedSessionOrThrow(sessionId: number, userId: number) {
    const session = await this.chatSessionRepository.findOne({
      where: {id: sessionId, userId},
    });
    if (!session) {
      throw new HttpErrors.NotFound('Chat session not found.');
    }
    return session;
  }

  private async assistantMessageOrThrow(sessionId: number, messageId: number) {
    const message = await this.chatMessageRepository.findOne({
      where: {id: messageId, sessionId, role: 'assistant'},
    });
    if (!message) throw new HttpErrors.NotFound('Assistant message not found.');
    return message;
  }

  @post('/chat-sessions/{id}/messages')
  @response(202, {description: 'Accepted chat message'})
  async create(
    @param.path.number('id') sessionId: number,
    @requestBody() body: SendMessageBody,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const session = await this.ownedSessionOrThrow(sessionId, user.id);
    const content = stripNulBytes(body.content?.trim() ?? '');
    if (!content) throw new HttpErrors.BadRequest('content is required.');

    const links = await this.chatSessionDirectoryRepository.find({where: {sessionId}});
    if (!links.length) {
      throw new HttpErrors.BadRequest('Link at least one directory before sending a message.');
    }

    const sessionPatch: {updatedAt: string; title?: string; model?: string; temperature?: number} = {
      updatedAt: new Date().toISOString(),
    };
    if (body.model?.trim()) sessionPatch.model = body.model.trim();
    if (typeof body.temperature === 'number' && Number.isFinite(body.temperature)) {
      sessionPatch.temperature = body.temperature;
    }
    if (!session.title || session.title === 'New chat') {
      sessionPatch.title = content.length > 60 ? `${content.slice(0, 57)}...` : content;
    }
    await this.chatSessionRepository.updateById(sessionId, sessionPatch);

    const userMessage = await this.chatMessageRepository.create({
      sessionId,
      role: 'user',
      content,
      status: 'completed',
    });
    const assistantMessage = await this.chatMessageRepository.create({
      sessionId,
      role: 'assistant',
      content: '',
      status: 'pending',
    });

    return {
      messageId: assistantMessage.id,
      assistantMessage: toSafeMessage(assistantMessage),
      userMessage: toSafeMessage(userMessage),
    };
  }

  @get('/chat-sessions/{id}/messages')
  @response(200, {description: 'Chat message history'})
  async list(@param.path.number('id') sessionId: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(sessionId, user.id);

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

    return messages.map(message => ({
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
    }));
  }

  @get('/chat-sessions/{id}/messages/{messageId}/stream')
  @response(200, {description: 'SSE stream for an assistant message'})
  async stream(
    @param.path.number('id') sessionId: number,
    @param.path.number('messageId') messageId: number,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(sessionId, user.id);
    const assistantMessage = await this.assistantMessageOrThrow(sessionId, messageId);
    if (!assistantMessage.id) throw new HttpErrors.NotFound('Assistant message not found.');

    const allMessages = await this.chatMessageRepository.find({
      where: {sessionId},
      order: ['createdAt ASC', 'id ASC'],
    });
    const assistantIndex = allMessages.findIndex(message => message.id === messageId);
    const userMessage = [...allMessages]
      .slice(0, assistantIndex < 0 ? undefined : assistantIndex)
      .reverse()
      .find(message => message.role === 'user' && message.content.trim());
    if (!userMessage) throw new HttpErrors.BadRequest('No user message found for this stream.');

    const conversationContext: ChatHistoryItem[] = allMessages
      .filter(
        message =>
          message.id !== undefined &&
          userMessage.id !== undefined &&
          message.id < userMessage.id &&
          message.status === 'completed' &&
          message.content.trim(),
      )
      .slice(-10)
      .map(message => ({role: message.role, content: message.content}));

    const abortController = new AbortController();
    activeStreams.set(messageId, abortController);

    const onClose = () => {
      abortController.abort();
    };
    this.request.on('close', onClose);

    this.res.status(200);
    this.res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    this.res.setHeader('Cache-Control', 'no-cache, no-transform');
    this.res.setHeader('Connection', 'keep-alive');
    this.res.flushHeaders?.();

    try {
      for await (const event of this.bridge().streamAssistantResponse({
        sessionId,
        assistantMessageId: assistantMessage.id,
        task: userMessage.content,
        conversationContext,
        signal: abortController.signal,
      })) {
        if (this.res.writableEnded) break;
        this.res.write(sseFrame(event));
      }
    } finally {
      activeStreams.delete(messageId);
      this.request.removeListener('close', onClose);
      if (!this.res.writableEnded) this.res.end();
    }

    return this.res;
  }

  @post('/chat-sessions/{id}/messages/{messageId}/cancel')
  @response(204, {description: 'Cancelled message stream'})
  async cancel(
    @param.path.number('id') sessionId: number,
    @param.path.number('messageId') messageId: number,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(sessionId, user.id);
    await this.assistantMessageOrThrow(sessionId, messageId);

    const active = activeStreams.get(messageId);
    if (active) active.abort();
    await this.chatMessageRepository.updateById(messageId, {
      status: 'cancelled',
      updatedAt: new Date().toISOString(),
    });
  }
}
