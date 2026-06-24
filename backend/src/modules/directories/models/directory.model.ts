import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'directories'}}})
export class Directory extends Entity {
  @property({type: 'number', id: true, generated: true})
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'user_id'}})
  userId: number;

  @property({type: 'string', required: true})
  name: string;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  @property({type: 'date', postgresql: {columnName: 'updated_at'}})
  updatedAt?: string;

  constructor(data?: Partial<Directory>) {
    super(data);
  }
}

export interface DirectoryRelations {
  // no relations defined yet
}

export type DirectoryWithRelations = Directory & DirectoryRelations;
