import {Entity, model, property} from '@loopback/repository';

export type BenchmarkRunItemStatus = 'pending' | 'running' | 'completed' | 'error';

@model({settings: {postgresql: {schema: 'public', table: 'benchmark_run_items'}}})
export class BenchmarkRunItem extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'run_id'}})
  runId: number;

  @property({type: 'string', required: true})
  provider: string;

  @property({type: 'string', required: true, postgresql: {columnName: 'model_id'}})
  modelId: string;

  @property({type: 'number', required: true, postgresql: {columnName: 'question_id'}})
  questionId: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'repeat_index'}})
  repeatIndex: number;

  @property({type: 'string', required: true})
  status: BenchmarkRunItemStatus;

  @property({type: 'string', postgresql: {columnName: 'final_result'}})
  finalResult?: string;

  @property({type: 'string', postgresql: {columnName: 'error_message'}})
  errorMessage?: string;

  @property({type: 'boolean', required: true})
  incomplete: boolean;

  // Set only when the candidate run itself succeeded but the best-effort
  // judge call failed (rate limit, judge model outage, schema validation) —
  // see BenchmarkRunnerService.executeItem. Null/undefined means either the
  // item hasn't been judged yet or judging succeeded.
  @property({type: 'string', postgresql: {columnName: 'judge_error'}})
  judgeError?: string;

  @property({type: 'number'})
  steps?: number;

  @property({type: 'number', postgresql: {columnName: 'api_calls'}})
  apiCalls?: number;

  @property({type: 'number', postgresql: {columnName: 'prompt_tokens'}})
  promptTokens?: number;

  @property({type: 'number', postgresql: {columnName: 'completion_tokens'}})
  completionTokens?: number;

  @property({type: 'number', postgresql: {columnName: 'thinking_tokens'}})
  thinkingTokens?: number;

  @property({type: 'number', postgresql: {columnName: 'total_tokens'}})
  totalTokens?: number;

  @property({type: 'number', postgresql: {columnName: 'tool_result_chars'}})
  toolResultChars?: number;

  @property({type: 'number', postgresql: {columnName: 'context_summaries'}})
  contextSummaries?: number;

  @property({type: 'number', postgresql: {columnName: 'duration_ms'}})
  durationMs?: number;

  @property({type: 'string', postgresql: {columnName: 'cost_usd'}})
  costUsd?: string;

  @property({type: 'string', postgresql: {columnName: 'cost_source'}})
  costSource?: 'provider' | 'estimated';

  @property({type: 'array', itemType: 'string', postgresql: {columnName: 'cited_sources'}})
  citedSources?: string[];

  @property({type: 'array', itemType: 'string', postgresql: {columnName: 'step_path'}})
  stepPath?: string[];

  @property({type: 'date', postgresql: {columnName: 'started_at'}})
  startedAt?: string;

  @property({type: 'date', postgresql: {columnName: 'completed_at'}})
  completedAt?: string;

  constructor(data?: Partial<BenchmarkRunItem>) {
    super(data);
  }
}

export interface BenchmarkRunItemRelations {
  // no relations defined yet
}

export type BenchmarkRunItemWithRelations = BenchmarkRunItem & BenchmarkRunItemRelations;
