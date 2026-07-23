import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'benchmark_question_directories'}}})
export class BenchmarkQuestionDirectory extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'question_id'}})
  questionId: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'directory_id'}})
  directoryId: number;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  constructor(data?: Partial<BenchmarkQuestionDirectory>) {
    super(data);
  }
}

export interface BenchmarkQuestionDirectoryRelations {
  // no relations defined yet
}

export type BenchmarkQuestionDirectoryWithRelations = BenchmarkQuestionDirectory &
  BenchmarkQuestionDirectoryRelations;
