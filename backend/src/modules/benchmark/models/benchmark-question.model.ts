import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'benchmark_questions'}}})
export class BenchmarkQuestion extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'string', required: true})
  prompt: string;

  @property({type: 'string', postgresql: {columnName: 'reference_answer'}})
  referenceAnswer?: string;

  @property({type: 'array', itemType: 'string', postgresql: {columnName: 'expected_facts'}})
  expectedFacts?: string[];

  @property({type: 'string', postgresql: {columnName: 'rubric_notes'}})
  rubricNotes?: string;

  @property({type: 'array', itemType: 'string'})
  tags?: string[];

  @property({type: 'boolean', required: true, postgresql: {columnName: 'is_active'}})
  isActive: boolean;

  @property({type: 'number', postgresql: {columnName: 'created_by'}})
  createdBy?: number;

  @property({type: 'number', postgresql: {columnName: 'updated_by'}})
  updatedBy?: number;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  @property({type: 'date', postgresql: {columnName: 'updated_at'}})
  updatedAt?: string;

  constructor(data?: Partial<BenchmarkQuestion>) {
    super(data);
  }
}

export interface BenchmarkQuestionRelations {
  // no relations defined yet
}

export type BenchmarkQuestionWithRelations = BenchmarkQuestion & BenchmarkQuestionRelations;
