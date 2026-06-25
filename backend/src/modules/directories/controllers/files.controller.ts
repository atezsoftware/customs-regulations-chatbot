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
  Response,
  RestBindings,
} from '@loopback/rest';
import multer from 'multer';
import {getCurrentUser} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {DirectoryFileRepository, DirectoryRepository} from '../repositories';
import {fetchDocumentChunks, StorageService, virtualCorpusKey} from '../services';
import {toSafeFile} from '../transformers';
import {DirectoriesController} from './directories.controller';

const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024;
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {fileSize: MAX_FILE_SIZE_BYTES},
});

interface UploadedFile {
  originalname: string;
  mimetype: string;
  buffer: Buffer;
}

interface RenameFileBody {
  name: string;
}

export class FilesController {
  private storageService = new StorageService();
  private directoriesController: DirectoriesController;

  constructor(
    @repository(DirectoryRepository) private directoryRepository: DirectoryRepository,
    @repository(DirectoryFileRepository)
    private directoryFileRepository: DirectoryFileRepository,
    @repository(UserRepository) private userRepository: UserRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
    @inject(RestBindings.Http.RESPONSE) private res: Response,
  ) {
    this.directoriesController = new DirectoriesController(
      directoryRepository,
      directoryFileRepository,
      userRepository,
      request,
    );
  }

  private async findOwnedFileOrThrow(directoryId: number, fileId: number, userId: number) {
    await this.directoriesController.ownedDirectoryOrThrow(directoryId, userId);
    const file = await this.directoryFileRepository.findOne({
      where: {id: fileId, directoryId},
    });
    if (!file) {
      throw new HttpErrors.NotFound('File not found.');
    }
    return file;
  }

  @get('/directories/{id}/files/{fileId}/chunks')
  @response(200, {description: 'Indexed chunks for a directory file'})
  async chunks(
    @param.path.number('id') directoryId: number,
    @param.path.number('fileId') fileId: number,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const file = await this.findOwnedFileOrThrow(directoryId, fileId, user.id);
    const {document, chunks} = await fetchDocumentChunks(
      virtualCorpusKey(directoryId),
      `${fileId}-`,
    );

    return {
      directoryId,
      file: toSafeFile(file),
      document: document
        ? {
            id: document.id,
            relativePath: document.relative_path,
            title: displayTitle(document.absolute_path || file.originalName),
          }
        : undefined,
      chunks: chunks.map(chunk => ({
        id: chunk.id,
        documentId: chunk.document_id,
        relativePath: chunk.relative_path,
        documentTitle: displayTitle(chunk.absolute_path || file.originalName),
        text: chunk.text,
        position: Number(chunk.position),
        startChar: Number(chunk.start_char),
        endChar: Number(chunk.end_char),
        chunkType: chunk.chunk_type,
        metadata: chunk.metadata,
        hasEmbedding: Boolean(chunk.has_embedding),
      })),
    };
  }

  @post('/directories/{id}/files')
  @response(200, {description: 'Uploaded files (multipart field name: "files", repeatable)'})
  async upload(@param.path.number('id') directoryId: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const directory = await this.directoriesController.ownedDirectoryOrThrow(
      directoryId,
      user.id,
    );

    const files = await new Promise<UploadedFile[]>((resolve, reject) => {
      upload.array('files')(this.request, this.res, err => {
        if (err) reject(err);
        else resolve(((this.request as unknown as {files?: UploadedFile[]}).files) ?? []);
      });
    });

    if (!files.length) {
      throw new HttpErrors.BadRequest(
        'No files were uploaded (expected multipart field "files").',
      );
    }

    const created = [];
    for (const file of files) {
      const {storedPath, sizeBytes} = await this.storageService.saveFile({
        userName: user.email,
        directoryName: directory.name,
        originalName: file.originalname,
        buffer: file.buffer,
      });
      const record = await this.directoryFileRepository.create({
        directoryId,
        originalName: file.originalname,
        storedPath,
        mimeType: file.mimetype,
        sizeBytes,
        storageStatus: 'stored',
      });
      created.push(toSafeFile(record));
    }
    return created;
  }

  @patch('/directories/{id}/files/{fileId}')
  @response(204, {description: 'Renamed file'})
  async rename(
    @param.path.number('id') directoryId: number,
    @param.path.number('fileId') fileId: number,
    @requestBody() body: RenameFileBody,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const directory = await this.directoriesController.ownedDirectoryOrThrow(
      directoryId,
      user.id,
    );
    const file = await this.findOwnedFileOrThrow(directoryId, fileId, user.id);
    const name = body.name?.trim();
    if (!name) throw new HttpErrors.BadRequest('name is required.');
    const storedPath =
      file.storageStatus !== 'indexed'
        ? await this.storageService.moveFileToDirectory({
            storedPath: file.storedPath,
            userName: user.email,
            directoryName: directory.name,
            originalName: name,
          })
        : file.storedPath;
    await this.directoryFileRepository.updateById(fileId, {
      originalName: name,
      storedPath,
      updatedAt: new Date().toISOString(),
    });
  }

  @del('/directories/{id}/files/{fileId}')
  @response(204, {description: 'Deleted file'})
  async delete(
    @param.path.number('id') directoryId: number,
    @param.path.number('fileId') fileId: number,
  ) {
    const user = await getCurrentUser(this.request, this.userRepository);
    const file = await this.findOwnedFileOrThrow(directoryId, fileId, user.id);
    await this.directoryFileRepository.deleteById(fileId);
    await this.storageService.deleteFile(file.storedPath);
  }
}

function displayTitle(value: string): string {
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
