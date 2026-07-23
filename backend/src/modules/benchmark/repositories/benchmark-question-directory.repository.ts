import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {BenchmarkQuestionDirectory, BenchmarkQuestionDirectoryRelations} from '../models';

export class BenchmarkQuestionDirectoryRepository extends DefaultCrudRepository<
  BenchmarkQuestionDirectory,
  typeof BenchmarkQuestionDirectory.prototype.id,
  BenchmarkQuestionDirectoryRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(BenchmarkQuestionDirectory, dataSource);
  }
}
