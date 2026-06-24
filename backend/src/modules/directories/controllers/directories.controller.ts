import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {
  del,
  get,
  HttpErrors,
  param,
  patch,
  post,
  Request,
  requestBody,
  response,
  RestBindings,
} from '@loopback/rest';
import {getCurrentUser} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {DirectoryFileRepository, DirectoryRepository} from '../repositories';
import {
  DirectoryChunkingCompletion,
  DirectoryEmbeddingCompletion,
  DirectoryIndexStatus,
  getDirectoryIndexStatus,
  startCorpusEmbedding,
  startDirectoryChunking,
  StorageService,
  virtualCorpusKey,
} from '../services';
import {toSafeDirectory, toSafeFile} from '../transformers';

interface DirectoryBody {
  name: string;
}

export class DirectoriesController {
  private storageService = new StorageService();

  constructor(
    @repository(DirectoryRepository) private directoryRepository: DirectoryRepository,
    @repository(DirectoryFileRepository)
    private directoryFileRepository: DirectoryFileRepository,
    @repository(UserRepository) private userRepository: UserRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  async ownedDirectoryOrThrow(directoryId: number, userId: number) {
    const directory = await this.directoryRepository.findOne({
      where: {id: directoryId, userId},
    });
    if (!directory) {
      throw new HttpErrors.NotFound('Directory not found.');
    }
    return directory;
  }

  @post('/directories')
  @response(200, {description: 'Created directory'})
  async create(@requestBody() body: DirectoryBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const name = body.name?.trim();
    if (!name) throw new HttpErrors.BadRequest('name is required.');
    const directory = await this.directoryRepository.create({userId: user.id, name});
    return toSafeDirectory(directory);
  }

  @get('/directories')
  @response(200, {description: 'List directories for the current user'})
  async list() {
    const user = await getCurrentUser(this.request, this.userRepository);
    const directories = await this.directoryRepository.find({
      where: {userId: user.id},
      order: ['createdAt DESC'],
    });
    return directories.map(toSafeDirectory);
  }

  @get('/directories/{id}')
  @response(200, {description: 'Directory with its files'})
  async getOne(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const directory = await this.ownedDirectoryOrThrow(id, user.id);
    const files = await this.directoryFileRepository.find({
      where: {directoryId: id},
      order: ['createdAt DESC'],
    });
    return {...toSafeDirectory(directory), files: files.map(toSafeFile)};
  }

  @get('/directories/{id}/index/status')
  @response(200, {description: 'Index status for this directory'})
  async indexStatus(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedDirectoryOrThrow(id, user.id);
    const serviceStatus = await getDirectoryIndexStatus(id);
    if (serviceStatus.status === 'chunking' || serviceStatus.status === 'indexing' || serviceStatus.status === 'error') {
      return serviceStatus;
    }

    const files = await this.directoryFileRepository.find({where: {directoryId: id}});
    return this.statusFromFiles(id, files, serviceStatus);
  }

  @post('/directories/{id}/chunks')
  @response(202, {description: 'Started generating chunks for this directory'})
  async startChunking(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const directory = await this.ownedDirectoryOrThrow(id, user.id);
    const files = await this.directoryFileRepository.find({
      where: {directoryId: id},
      order: ['createdAt ASC'],
    });
    if (!files.length) {
      throw new HttpErrors.BadRequest('Upload at least one file before generating chunks.');
    }
    const chunkableFiles = files.filter(
      file => file.storageStatus !== 'chunked' && file.storageStatus !== 'indexed',
    );
    for (const file of chunkableFiles) {
      const storedPath = await this.storageService.moveFileToDirectory({
        storedPath: file.storedPath,
        userName: user.email,
        directoryName: directory.name,
        originalName: file.originalName,
      });
      if (storedPath !== file.storedPath) {
        await this.directoryFileRepository.updateById(file.id, {
          storedPath,
          updatedAt: new Date().toISOString(),
        });
        file.storedPath = storedPath;
      }
    }
    const indexableFiles = chunkableFiles.map(file => ({
      id: file.id,
      originalName: file.originalName,
      storedPath: file.storedPath,
      storageStatus: file.storageStatus,
    }));

    if (!indexableFiles.length) {
      return this.statusFromFiles(id, files, {
        directoryId: id,
        status: 'chunked',
        progress: 50,
        message: 'Chunks are already up to date.',
        updatedAt: new Date().toISOString(),
      });
    }

    return startDirectoryChunking(id, indexableFiles, {
      onCompleted: completion => this.markDirectoryChunked(completion),
    });
  }

  @post('/directories/{id}/index')
  @response(202, {description: 'Started indexing (embedding) this directory'})
  async startIndex(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedDirectoryOrThrow(id, user.id);
    const files = await this.directoryFileRepository.find({where: {directoryId: id}});
    if (!files.length) {
      throw new HttpErrors.BadRequest('Upload at least one file before indexing.');
    }
    const unchunkedFiles = files.filter(
      file => file.storageStatus !== 'chunked' && file.storageStatus !== 'indexed' && file.storageStatus !== 'error',
    );
    if (unchunkedFiles.length) {
      throw new HttpErrors.BadRequest('Generate chunks before indexing.');
    }
    if (files.every(file => file.storageStatus === 'indexed' || file.storageStatus === 'error')) {
      return this.statusFromFiles(id, files, {
        directoryId: id,
        status: 'completed',
        progress: 100,
        message: 'Index is already complete.',
        updatedAt: new Date().toISOString(),
      });
    }

    return startCorpusEmbedding(id, virtualCorpusKey(id), {
      onCompleted: completion => this.markDirectoryEmbedded(completion),
    });
  }

  @patch('/directories/{id}')
  @response(204, {description: 'Renamed directory'})
  async rename(@param.path.number('id') id: number, @requestBody() body: DirectoryBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedDirectoryOrThrow(id, user.id);
    const name = body.name?.trim();
    if (!name) throw new HttpErrors.BadRequest('name is required.');
    const files = await this.directoryFileRepository.find({where: {directoryId: id}});
    for (const file of files.filter(file => file.storageStatus !== 'indexed')) {
      const storedPath = await this.storageService.moveFileToDirectory({
        storedPath: file.storedPath,
        userName: user.email,
        directoryName: name,
        originalName: file.originalName,
      });
      if (storedPath !== file.storedPath) {
        await this.directoryFileRepository.updateById(file.id, {
          storedPath,
          updatedAt: new Date().toISOString(),
        });
      }
    }
    await this.directoryRepository.updateById(id, {name, updatedAt: new Date().toISOString()});
  }

  @del('/directories/{id}')
  @response(204, {description: 'Deleted directory and its files'})
  async delete(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    await this.ownedDirectoryOrThrow(id, user.id);
    const files = await this.directoryFileRepository.find({where: {directoryId: id}});
    await this.directoryFileRepository.deleteAll({directoryId: id});
    await this.directoryRepository.deleteById(id);
    await this.storageService.deleteFiles(files.map(file => file.storedPath));
  }

  private async markDirectoryChunked(completion: DirectoryChunkingCompletion): Promise<void> {
    const now = new Date().toISOString();
    for (const skipped of completion.skippedFiles) {
      if (!skipped.file?.id) continue;
      await this.directoryFileRepository.updateById(skipped.file.id, {
        storageStatus: 'error',
        storageError: skipped.reason,
        updatedAt: now,
      });
    }

    for (const file of completion.indexedFiles) {
      if (!file.id) continue;
      // Chunk text is now safely in `core_chunks`/`core_documents` — the raw
      // upload is no longer needed even though embeddings haven't run yet.
      await this.storageService.deleteFile(file.storedPath);
      await this.directoryFileRepository.updateById(file.id, {
        storageStatus: 'chunked',
        chunkedAt: now,
        rawDeletedAt: new Date().toISOString(),
        storageError: null,
        updatedAt: new Date().toISOString(),
      });
    }
  }

  private async markDirectoryEmbedded(_completion: DirectoryEmbeddingCompletion): Promise<void> {
    const now = new Date().toISOString();
    const files = await this.directoryFileRepository.find({
      where: {directoryId: _completion.directoryId, storageStatus: 'chunked'},
    });
    for (const file of files) {
      await this.directoryFileRepository.updateById(file.id, {
        storageStatus: 'indexed',
        indexedAt: now,
        updatedAt: now,
      });
    }
  }

  private statusFromFiles(
    directoryId: number,
    files: Array<{
      storageStatus?: string;
      originalName: string;
      storageError?: string | null;
    }>,
    fallback: DirectoryIndexStatus,
  ): DirectoryIndexStatus {
    if (!files.length) return fallback;

    const indexedCount = files.filter(file => file.storageStatus === 'indexed').length;
    const chunkedCount = files.filter(file => file.storageStatus === 'chunked').length;
    const errorFiles = files.filter(file => file.storageStatus === 'error');
    const storedCount = files.length - indexedCount - chunkedCount - errorFiles.length;
    const skippedFiles = errorFiles.map(
      file => `${file.originalName}: ${file.storageError ?? 'not indexed'}`,
    );

    if (indexedCount > 0 && storedCount === 0 && chunkedCount === 0) {
      return {
        directoryId,
        status: 'completed',
        progress: 100,
        message: skippedFiles.length
          ? `Index is complete. Skipped ${skippedFiles.length} invalid file(s).`
          : 'Index is complete and raw upload files were removed.',
        documentCount: indexedCount,
        skippedFiles,
        updatedAt: new Date().toISOString(),
      };
    }

    if (indexedCount > 0 && (storedCount > 0 || chunkedCount > 0)) {
      return {
        directoryId,
        status: 'stale',
        progress: 65,
        message: 'Some files were uploaded or changed after the last index.',
        documentCount: indexedCount,
        skippedFiles,
        updatedAt: new Date().toISOString(),
      };
    }

    if (indexedCount === 0 && chunkedCount > 0 && storedCount === 0) {
      return {
        directoryId,
        status: 'chunked',
        progress: 50,
        message: skippedFiles.length
          ? `Chunks are ready. Skipped ${skippedFiles.length} invalid file(s).`
          : 'Chunks are ready. Start indexing to generate embeddings.',
        documentCount: chunkedCount,
        skippedFiles,
        updatedAt: new Date().toISOString(),
      };
    }

    if (errorFiles.length === files.length) {
      return {
        directoryId,
        status: 'error',
        progress: 0,
        message: 'No valid indexable files were found.',
        skippedFiles,
        updatedAt: new Date().toISOString(),
      };
    }

    return fallback;
  }
}
