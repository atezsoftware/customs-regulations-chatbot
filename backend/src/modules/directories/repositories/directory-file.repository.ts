import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {DirectoryFile, DirectoryFileRelations} from '../models';

export class DirectoryFileRepository extends DefaultCrudRepository<
  DirectoryFile,
  typeof DirectoryFile.prototype.id,
  DirectoryFileRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(DirectoryFile, dataSource);
  }
}
