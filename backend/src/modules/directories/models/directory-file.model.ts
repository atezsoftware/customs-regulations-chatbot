import {Entity, model, property} from '@loopback/repository';

export type DirectoryFileStorageStatus = 'stored' | 'indexed' | 'error';

@model({settings: {postgresql: {schema: 'public', table: 'directory_files'}}})
export class DirectoryFile extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'directory_id'}})
  directoryId: number;

  @property({type: 'string', required: true, postgresql: {columnName: 'original_name'}})
  originalName: string;

  @property({type: 'string', required: true, postgresql: {columnName: 'stored_path'}})
  storedPath: string;

  @property({type: 'string', postgresql: {columnName: 'mime_type'}})
  mimeType?: string;

  @property({type: 'number', required: true, postgresql: {columnName: 'size_bytes'}})
  sizeBytes: number;

  @property({
    type: 'string',
    default: 'stored',
    postgresql: {columnName: 'storage_status'},
  })
  storageStatus?: DirectoryFileStorageStatus;

  @property({type: 'date', postgresql: {columnName: 'indexed_at'}})
  indexedAt?: string | null;

  @property({type: 'date', postgresql: {columnName: 'raw_deleted_at'}})
  rawDeletedAt?: string | null;

  @property({type: 'string', postgresql: {columnName: 'storage_error'}})
  storageError?: string | null;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  @property({type: 'date', postgresql: {columnName: 'updated_at'}})
  updatedAt?: string;

  constructor(data?: Partial<DirectoryFile>) {
    super(data);
  }
}

export interface DirectoryFileRelations {
  // no relations defined yet
}

export type DirectoryFileWithRelations = DirectoryFile & DirectoryFileRelations;
