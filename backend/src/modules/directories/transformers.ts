import {Directory} from './models/directory.model';
import {DirectoryFile} from './models/directory-file.model';

export function toSafeDirectory(directory: Directory) {
  return {id: directory.id, name: directory.name, createdAt: directory.createdAt};
}

export function toSafeFile(file: DirectoryFile) {
  return {
    id: file.id,
    name: file.originalName,
    mimeType: file.mimeType,
    sizeBytes: file.sizeBytes,
    storageStatus: file.storageStatus ?? 'stored',
    indexedAt: file.indexedAt,
    rawDeletedAt: file.rawDeletedAt,
    storageError: file.storageError,
    createdAt: file.createdAt,
  };
}
