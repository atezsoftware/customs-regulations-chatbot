import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {LlmCall, LlmCallRelations} from '../models';

export class LlmCallRepository extends DefaultCrudRepository<
  LlmCall,
  typeof LlmCall.prototype.id,
  LlmCallRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(LlmCall, dataSource);
  }
}
