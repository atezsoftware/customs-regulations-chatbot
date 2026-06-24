import {Entity, model, property} from '@loopback/repository';

export const USER_ROLES = ['user', 'admin'] as const;
export type UserRole = (typeof USER_ROLES)[number];

@model({settings: {postgresql: {schema: 'public', table: 'users'}}})
export class User extends Entity {
  @property({
    type: 'number',
    id: true,
    generated: true,
    postgresql: {columnName: 'id'},
  })
  id?: number;

  @property({type: 'string', required: true})
  email: string;

  @property({
    type: 'string',
    required: true,
    postgresql: {columnName: 'password_hash'},
  })
  passwordHash: string;

  @property({type: 'string', postgresql: {columnName: 'full_name'}})
  fullName?: string;

  @property({type: 'string', default: 'user'})
  role: UserRole;

  @property({type: 'boolean', default: true, postgresql: {columnName: 'is_active'}})
  isActive: boolean;

  @property({
    type: 'number',
    default: 0,
    postgresql: {columnName: 'failed_login_attempts'},
  })
  failedLoginAttempts: number;

  @property({type: 'date', postgresql: {columnName: 'locked_until'}})
  lockedUntil?: string | null;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  @property({type: 'date', postgresql: {columnName: 'updated_at'}})
  updatedAt?: string;

  constructor(data?: Partial<User>) {
    super(data);
  }
}

export interface UserRelations {
  // no relations defined yet
}

export type UserWithRelations = User & UserRelations;
