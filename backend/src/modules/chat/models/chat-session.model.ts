import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'chat_sessions'}}})
export class ChatSession extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'user_id'}})
  userId: number;

  @property({type: 'string'})
  title?: string;

  @property({type: 'string'})
  model?: string;

  @property({type: 'number'})
  temperature?: number;

  @property({type: 'number', postgresql: {columnName: 'last_context_usage_ratio'}})
  lastContextUsageRatio?: number;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  @property({type: 'date', postgresql: {columnName: 'updated_at'}})
  updatedAt?: string;

  constructor(data?: Partial<ChatSession>) {
    super(data);
  }
}

export interface ChatSessionRelations {
  // no relations defined yet
}

export type ChatSessionWithRelations = ChatSession & ChatSessionRelations;
