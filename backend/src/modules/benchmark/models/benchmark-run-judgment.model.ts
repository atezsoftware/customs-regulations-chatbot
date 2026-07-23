import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'benchmark_run_judgments'}}})
export class BenchmarkRunJudgment extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'run_item_id'}})
  runItemId: number;

  @property({type: 'string', required: true, postgresql: {columnName: 'judge_provider'}})
  judgeProvider: string;

  @property({type: 'string', required: true, postgresql: {columnName: 'judge_model'}})
  judgeModel: string;

  @property({type: 'number', required: true, postgresql: {columnName: 'correctness_score'}})
  correctnessScore: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'groundedness_score'}})
  groundednessScore: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'completeness_score'}})
  completenessScore: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'clarity_score'}})
  clarityScore: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'overall_score'}})
  overallScore: number;

  @property({type: 'string'})
  rationale?: string;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  constructor(data?: Partial<BenchmarkRunJudgment>) {
    super(data);
  }
}

export interface BenchmarkRunJudgmentRelations {
  // no relations defined yet
}

export type BenchmarkRunJudgmentWithRelations = BenchmarkRunJudgment & BenchmarkRunJudgmentRelations;
