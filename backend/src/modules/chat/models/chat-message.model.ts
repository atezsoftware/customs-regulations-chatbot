import {Entity, model, property} from '@loopback/repository';

export type ChatMessageRole = 'user' | 'assistant';
export type ChatMessageStatus = 'pending' | 'streaming' | 'completed' | 'error' | 'cancelled';

@model({settings: {postgresql: {schema: 'public', table: 'chat_messages'}}})
export class ChatMessage extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'session_id'}})
  sessionId: number;

  @property({type: 'string', required: true})
  role: ChatMessageRole;

  @property({type: 'string'})
  content: string;

  @property({type: 'string', required: true})
  status: ChatMessageStatus;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  @property({type: 'date', postgresql: {columnName: 'updated_at'}})
  updatedAt?: string;

  constructor(data?: Partial<ChatMessage>) {
    super(data);
  }
}

export interface ChatMessageRelations {
  // no relations defined yet
}

export type ChatMessageWithRelations = ChatMessage & ChatMessageRelations;
