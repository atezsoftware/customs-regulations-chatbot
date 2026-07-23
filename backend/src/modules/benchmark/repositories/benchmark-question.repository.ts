import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {BenchmarkQuestion, BenchmarkQuestionRelations} from '../models';

export class BenchmarkQuestionRepository extends DefaultCrudRepository<
  BenchmarkQuestion,
  typeof BenchmarkQuestion.prototype.id,
  BenchmarkQuestionRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(BenchmarkQuestion, dataSource);
  }

  async activeQuestions(): Promise<BenchmarkQuestion[]> {
    return this.find({where: {isActive: true}, order: ['id ASC']});
  }
}
