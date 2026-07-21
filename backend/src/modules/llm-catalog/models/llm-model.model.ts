import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'llm_models'}}})
export class LlmModel extends Entity {
  @property({type: 'string', id: true}) provider: string;
  @property({type: 'string', id: true, postgresql: {columnName: 'model_id'}}) modelId: string;
  @property({type: 'string', required: true, postgresql: {columnName: 'display_name'}}) displayName: string;
  @property({type: 'string'}) description?: string;
  @property({type: 'number', required: true, postgresql: {columnName: 'context_length'}}) contextLength: number;
  @property({type: 'number', postgresql: {columnName: 'max_completion_tokens'}}) maxCompletionTokens?: number;
  @property({type: 'array', itemType: 'string', postgresql: {columnName: 'input_modalities'}}) inputModalities?: string[];
  @property({type: 'array', itemType: 'string', postgresql: {columnName: 'output_modalities'}}) outputModalities?: string[];
  @property({type: 'array', itemType: 'string', postgresql: {columnName: 'supported_parameters'}}) supportedParameters?: string[];
  @property({type: 'object', postgresql: {columnName: 'raw_pricing'}}) rawPricing?: object;
  @property({type: 'string', postgresql: {columnName: 'prompt_usd_per_token'}}) promptUsdPerToken?: string;
  @property({type: 'string', postgresql: {columnName: 'completion_usd_per_token'}}) completionUsdPerToken?: string;
  @property({type: 'string', postgresql: {columnName: 'request_usd'}}) requestUsd?: string;
  @property({type: 'boolean', required: true, postgresql: {columnName: 'is_active'}}) isActive: boolean;
  @property({type: 'date', postgresql: {columnName: 'last_synced_at'}}) lastSyncedAt?: string;
}
