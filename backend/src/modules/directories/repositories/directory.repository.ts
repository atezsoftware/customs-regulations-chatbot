import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {Directory, DirectoryRelations} from '../models';

export class DirectoryRepository extends DefaultCrudRepository<
  Directory,
  typeof Directory.prototype.id,
  DirectoryRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(Directory, dataSource);
  }
}
