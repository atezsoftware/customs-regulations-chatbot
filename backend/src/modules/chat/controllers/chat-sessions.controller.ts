import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {del, get, HttpErrors, param, patch, post, put, Request, requestBody, response, RestBindings} from '@loopback/rest';
import {getCurrentUser} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {DirectoryFileRepository, DirectoryRepository} from '../../directories/repositories';
import {toSafeDirectory, toSafeFile} from '../../directories/transformers';
import {ChatSessionDirectoryRepository, ChatSessionRepository, LlmCallRepository} from '../repositories';
import {LlmModelRepository} from '../../llm-catalog/repositories';

interface SessionBody {
  title?: string;
}

interface SetDirectoriesBody {
  directoryIds: number[];
}

interface SetModelBody { provider: string; modelId: string; }

function toSafeSession(
  session: {
    id?: number;
    title?: string;
    createdAt?: string;
    updatedAt?: string;
    lastContextUsageRatio?: number;
    llmProvider?: string;
    model?: string;
  },
  totalTokens = 0,
) {
  return {
    id: session.id,
    title: session.title,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    totalTokens,
    lastContextUsageRatio: session.lastContextUsageRatio ?? null,
    llmProvider: session.llmProvider,
    model: session.model,
  };
}

export class ChatSessionsController {
  constructor(
    @repository(ChatSessionRepository) private chatSessionRepository: ChatSessionRepository,
    @repository(ChatSessionDirectoryRepository)
    private chatSessionDirectoryRepository: ChatSessionDirectoryRepository,
    @repository(DirectoryRepository) private directoryRepository: DirectoryRepository,
    @repository(DirectoryFileRepository)
    private directoryFileRepository: DirectoryFileRepository,
    @repository(UserRepository) private userRepository: UserRepository,
    @repository(LlmCallRepository) private llmCallRepository: LlmCallRepository,
    @repository(LlmModelRepository) private llmModelRepository: LlmModelRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  private async ownedSessionOrThrow(sessionId: number, userId: number) {
    const session = await this.chatSessionRepository.findOne({
      where: {id: sessionId, userId},
    });
    if (!session) {
      throw new HttpErrors.NotFound('Chat session not found.');
    }
    return session;
  }

  @post('/chat-sessions')
  @response(200, {description: 'Created chat session'})
  async create(@requestBody() body: SessionBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const defaultModel = process.env.OPENROUTER_DEFAULT_MODEL ?? 'google/gemini-3-flash-preview';
    const available = await this.llmModelRepository.findOne({
      where: {provider: 'openrouter', modelId: defaultModel, isActive: true},
    });
    if (!available) {
      throw new HttpErrors.ServiceUnavailable('The configured default model is temporarily unavailable.');
    }
    const session = await this.chatSessionRepository.create({
      userId: user.id,
      title: body.title?.trim() || 'New chat',
      llmProvider: 'openrouter',
      model: defaultModel,
    });
    return toSafeSession(session);
  }

  @patch('/chat-sessions/{id}/model')
  @response(200, {description: 'Persisted chat model selection'})
  async setModel(@param.path.number('id') id: number, @requestBody() body: SetModelBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const session = await this.ownedSessionOrThrow(id, user.id);
    if (body.provider !== 'openrouter' || !body.modelId?.trim()) {
      throw new HttpErrors.BadRequest('Choose an active OpenRouter model.');
    }
    const model = await this.llmModelRepository.findOne({where: {provider: 'openrouter', modelId: body.modelId.trim(), isActive: true}});
    if (!model) throw new HttpErrors.BadRequest('Selected model is not available.');
    await this.chatSessionRepository.updateById(id, {llmProvider: 'openrouter', model: model.modelId, updatedAt: new Date().toISOString()});
    return toSafeSession({...session, llmProvider: 'openrouter', model: model.modelId, updatedAt: new Date().toISOString()});
  }

  @get('/chat-sessions')
  @response(200, {description: 'List chat sessions for the current user'})
  async list() {
    const user = await getCurrentUser(this.request, this.userRepository);
    const [sessions, usageRows] = await Promise.all([
      this.chatSessionRepository.find({
        where: {userId: user.id},
        order: ['createdAt DESC'],
      }),
      this.llmCallRepository.usageTotalsPerSession(user.id),
    ]);
    const totalTokensBySession = new Map(
      usageRows.map(row => [Number(row.session_id), Number(row.total_tokens)]),
    );
    return sessions.map(session => toSafeSession(session, totalTokensBySession.get(session.id ?? -1) ?? 0));
  }

  @patch('/chat-sessions/{id}')
  @response(204, {description: 'Renamed chat session'})
  async rename(@param.path.number('id') id: number, @requestBody() body: SessionBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(id, user.id);
    const title = body.title?.trim();
    if (!title) throw new HttpErrors.BadRequest('title is required.');
    await this.chatSessionRepository.updateById(id, {title, updatedAt: new Date().toISOString()});
  }

  @del('/chat-sessions/{id}')
  @response(204, {description: 'Deleted chat session'})
  async delete(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(id, user.id);
    await this.chatSessionDirectoryRepository.deleteAll({sessionId: id});
    await this.chatSessionRepository.deleteById(id);
  }

  @get('/chat-sessions/{id}/directories')
  @response(200, {description: 'Directories linked to this chat session'})
  async listDirectories(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(id, user.id);
    const links = await this.chatSessionDirectoryRepository.find({where: {sessionId: id}});
    if (!links.length) return [];
    const directoryIds = links.map(l => l.directoryId);
    const directories = await this.directoryRepository.find({
      where: {id: {inq: directoryIds}},
    });
    return directories.map(toSafeDirectory);
  }

  /**
   * Replaces the full set of directories linked to this session. This is the
   * single point that decides what a session can see — anything not listed
   * here must never be visible to it (no implicit access to the user's other
   * directories).
   */
  @put('/chat-sessions/{id}/directories')
  @response(204, {description: 'Replaced the set of linked directories'})
  async setDirectories(
    @param.path.number('id') id: number,
    @requestBody() body: SetDirectoriesBody,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(id, user.id);

    const directoryIds = [...new Set(body.directoryIds ?? [])];
    if (directoryIds.length) {
      const found = await this.directoryRepository.find({
        where: {id: {inq: directoryIds}},
      });
      if (found.length !== directoryIds.length) {
        throw new HttpErrors.BadRequest('One or more directoryIds do not exist.');
      }
    }

    await this.chatSessionDirectoryRepository.deleteAll({sessionId: id});
    await Promise.all(
      directoryIds.map(directoryId =>
        this.chatSessionDirectoryRepository.create({sessionId: id, directoryId}),
      ),
    );
  }

  @get('/chat-sessions/{id}/files')
  @response(200, {description: 'Files visible to this session (only from linked directories)'})
  async listFiles(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedSessionOrThrow(id, user.id);

    const links = await this.chatSessionDirectoryRepository.find({where: {sessionId: id}});
    if (!links.length) return [];

    const directoryIds = links.map(l => l.directoryId);
    const files = await this.directoryFileRepository.find({
      where: {directoryId: {inq: directoryIds}},
      order: ['createdAt DESC'],
    });
    return files.map(file => ({...toSafeFile(file), directoryId: file.directoryId}));
  }
}
