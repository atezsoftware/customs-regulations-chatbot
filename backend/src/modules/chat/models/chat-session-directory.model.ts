import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'chat_session_directories'}}})
export class ChatSessionDirectory extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'session_id'}})
  sessionId: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'directory_id'}})
  directoryId: number;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  constructor(data?: Partial<ChatSessionDirectory>) {
    super(data);
  }
}

export interface ChatSessionDirectoryRelations {
  // no relations defined yet
}

export type ChatSessionDirectoryWithRelations = ChatSessionDirectory & ChatSessionDirectoryRelations;
