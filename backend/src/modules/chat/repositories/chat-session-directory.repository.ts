import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {ChatSessionDirectory, ChatSessionDirectoryRelations} from '../models';

export class ChatSessionDirectoryRepository extends DefaultCrudRepository<
  ChatSessionDirectory,
  typeof ChatSessionDirectory.prototype.id,
  ChatSessionDirectoryRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(ChatSessionDirectory, dataSource);
  }
}
