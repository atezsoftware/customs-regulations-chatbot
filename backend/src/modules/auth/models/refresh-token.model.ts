import {Entity, model, property} from '@loopback/repository';

@model({settings: {postgresql: {schema: 'public', table: 'refresh_tokens'}}})
export class RefreshToken extends Entity {
  @property({
    type: 'number',
    id: true,
    generated: true,
    postgresql: {columnName: 'id'},
  })
  id?: number;

  @property({type: 'number', required: true, postgresql: {columnName: 'user_id'}})
  userId: number;

  @property({
    type: 'string',
    required: true,
    postgresql: {columnName: 'token_hash'},
  })
  tokenHash: string;

  @property({type: 'date', required: true, postgresql: {columnName: 'expires_at'}})
  expiresAt: string;

  @property({type: 'date', postgresql: {columnName: 'revoked_at'}})
  revokedAt?: string;

  @property({
    type: 'string',
    postgresql: {columnName: 'replaced_by_token_hash'},
  })
  replacedByTokenHash?: string;

  @property({type: 'date', postgresql: {columnName: 'created_at'}})
  createdAt?: string;

  constructor(data?: Partial<RefreshToken>) {
    super(data);
  }
}

export interface RefreshTokenRelations {
  // no relations defined yet
}

export type RefreshTokenWithRelations = RefreshToken & RefreshTokenRelations;
