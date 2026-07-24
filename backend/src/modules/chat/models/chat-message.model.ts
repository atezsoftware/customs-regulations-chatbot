import {Entity, model, property} from '@loopback/repository';

export type ChatMessageRole = 'user' | 'assistant';
export type ChatMessageStatus = 'pending' | 'streaming' | 'completed' | 'error' | 'cancelled';
// Retrieval breadth the user picked for this message — mirrors core-api's
// EffortLevel (agent.py). Chosen client-side and forwarded straight through
// to core-api's /ws/explore `start` payload; not persisted on the message
// row since it's only needed for the one streamAssistantResponse() call
// that create() and stream() together make (see chat-messages.controller.ts).
export type ChatEffortLevel = 'low' | 'medium' | 'high';

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

  @property({type: 'string', postgresql: {columnName: 'error_message'}})
  errorMessage?: string | null;

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
