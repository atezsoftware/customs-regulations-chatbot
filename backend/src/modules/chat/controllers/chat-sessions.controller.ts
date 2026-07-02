import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {del, get, HttpErrors, param, patch, post, put, Request, requestBody, response, RestBindings} from '@loopback/rest';
import {getCurrentUser} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {DirectoryFileRepository, DirectoryRepository} from '../../directories/repositories';
import {toSafeDirectory, toSafeFile} from '../../directories/transformers';
import {ChatSessionDirectoryRepository, ChatSessionRepository} from '../repositories';

interface SessionBody {
  title?: string;
}

interface SetDirectoriesBody {
  directoryIds: number[];
}

function toSafeSession(session: {id?: number; title?: string; createdAt?: string; updatedAt?: string}) {
  return {id: session.id, title: session.title, createdAt: session.createdAt, updatedAt: session.updatedAt};
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
    const session = await this.chatSessionRepository.create({
      userId: user.id,
      title: body.title?.trim() || 'New chat',
    });
    return toSafeSession(session);
  }

  @get('/chat-sessions')
  @response(200, {description: 'List chat sessions for the current user'})
  async list() {
    const user = await getCurrentUser(this.request, this.userRepository);
    const sessions = await this.chatSessionRepository.find({
      where: {userId: user.id},
      order: ['createdAt DESC'],
    });
    return sessions.map(toSafeSession);
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
