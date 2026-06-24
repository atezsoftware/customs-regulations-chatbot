import {Entity, model, property} from '@loopback/repository';

export type ResearchStepStatus = 'pending' | 'running' | 'completed' | 'error';

@model({settings: {postgresql: {schema: 'public', table: 'chat_research_steps'}}})
export class ChatResearchStep extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'message_id'}})
  messageId: number;

  @property({type: 'string', required: true, postgresql: {columnName: 'step_key'}})
  stepKey: string;

  @property({type: 'string', required: true})
  status: ResearchStepStatus;

  @property({type: 'string', required: true})
  title: string;

  @property({type: 'string'})
  preview?: string;

  @property({type: 'string'})
  details?: string;

  @property({type: 'object', postgresql: {dataType: 'jsonb'}})
  metadata?: object;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  @property({type: 'date', postgresql: {columnName: 'completed_at'}})
  completedAt?: string;

  constructor(data?: Partial<ChatResearchStep>) {
    super(data);
  }
}

export interface ChatResearchStepRelations {
  // no relations defined yet
}

export type ChatResearchStepWithRelations = ChatResearchStep & ChatResearchStepRelations;
