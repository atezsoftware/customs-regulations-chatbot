import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {ChatSource, ChatSourceRelations} from '../models';

export class ChatSourceRepository extends DefaultCrudRepository<
  ChatSource,
  typeof ChatSource.prototype.id,
  ChatSourceRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(ChatSource, dataSource);
  }
}
