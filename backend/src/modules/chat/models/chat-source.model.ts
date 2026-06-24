import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'chat_sources'}}})
export class ChatSource extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'message_id'}})
  messageId: number;

  @property({type: 'string', required: true})
  title: string;

  @property({type: 'string'})
  snippet?: string;

  @property({type: 'string'})
  url?: string;

  @property({type: 'string', postgresql: {columnName: 'file_path'}})
  filePath?: string;

  @property({type: 'number'})
  page?: number;

  @property({type: 'string', postgresql: {columnName: 'chunk_id'}})
  chunkId?: string;

  @property({type: 'number'})
  score?: number;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  constructor(data?: Partial<ChatSource>) {
    super(data);
  }
}

export interface ChatSourceRelations {
  // no relations defined yet
}

export type ChatSourceWithRelations = ChatSource & ChatSourceRelations;
