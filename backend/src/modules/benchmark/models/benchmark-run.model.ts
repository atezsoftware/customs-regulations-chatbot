import {Entity, model, property} from '@loopback/repository';

export type BenchmarkRunStatus = 'pending' | 'running' | 'completed' | 'error' | 'cancelled';

@model({settings: {postgresql: {schema: 'public', table: 'benchmark_runs'}}})
export class BenchmarkRun extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'string'})
  label?: string;

  @property({type: 'string', required: true})
  status: BenchmarkRunStatus;

  @property({type: 'string', required: true, postgresql: {columnName: 'judge_provider'}})
  judgeProvider: string;

  @property({type: 'string', required: true, postgresql: {columnName: 'judge_model'}})
  judgeModel: string;

  @property({type: 'number', postgresql: {columnName: 'created_by'}})
  createdBy?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'total_items'}})
  totalItems: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'completed_items'}})
  completedItems: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'failed_items'}})
  failedItems: number;

  @property({type: 'date', postgresql: {columnName: 'started_at'}})
  startedAt?: string;

  @property({type: 'date', postgresql: {columnName: 'completed_at'}})
  completedAt?: string;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  constructor(data?: Partial<BenchmarkRun>) {
    super(data);
  }
}

export interface BenchmarkRunRelations {
  // no relations defined yet
}

export type BenchmarkRunWithRelations = BenchmarkRun & BenchmarkRunRelations;
