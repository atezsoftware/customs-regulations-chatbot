import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {BenchmarkRunJudgment, BenchmarkRunJudgmentRelations} from '../models';

export class BenchmarkRunJudgmentRepository extends DefaultCrudRepository<
  BenchmarkRunJudgment,
  typeof BenchmarkRunJudgment.prototype.id,
  BenchmarkRunJudgmentRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(BenchmarkRunJudgment, dataSource);
  }
}
