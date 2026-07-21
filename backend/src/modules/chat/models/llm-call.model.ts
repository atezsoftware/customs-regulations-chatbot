import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'llm_calls'}}})
export class LlmCall extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', postgresql: {columnName: 'message_id'}})
  messageId?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'session_id'}})
  sessionId: number;

  @property({type: 'string', required: true})
  provider: string;

  @property({type: 'string'})
  model?: string;

  @property({type: 'string', required: true})
  purpose: string;

  @property({type: 'number', required: true, postgresql: {columnName: 'input_tokens'}})
  inputTokens: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'output_tokens'}})
  outputTokens: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'thinking_tokens'}})
  thinkingTokens: number;

  @property({type: 'number', postgresql: {columnName: 'duration_ms'}})
  durationMs?: number;

  @property({type: 'string', postgresql: {columnName: 'generation_id'}})
  generationId?: string;

  @property({type: 'number', postgresql: {columnName: 'cached_input_tokens'}})
  cachedInputTokens?: number;

  @property({type: 'number', postgresql: {columnName: 'cache_write_tokens'}})
  cacheWriteTokens?: number;

  @property({type: 'string', postgresql: {columnName: 'billed_cost_usd'}})
  billedCostUsd?: string;

  @property({type: 'string', postgresql: {columnName: 'upstream_cost_usd'}})
  upstreamCostUsd?: string;

  @property({type: 'string', postgresql: {columnName: 'cost_source'}})
  costSource?: 'provider' | 'estimated';

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  constructor(data?: Partial<LlmCall>) {
    super(data);
  }
}

export interface LlmCallRelations {
  // no relations defined yet
}

export type LlmCallWithRelations = LlmCall & LlmCallRelations;
