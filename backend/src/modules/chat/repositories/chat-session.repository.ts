import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {ChatSession, ChatSessionRelations} from '../models';

export class ChatSessionRepository extends DefaultCrudRepository<
  ChatSession,
  typeof ChatSession.prototype.id,
  ChatSessionRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(ChatSession, dataSource);
  }
}
