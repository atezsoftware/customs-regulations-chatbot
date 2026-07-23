import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {ChatSession, ChatSessionRelations} from '../models';

export interface AdminSessionRow {
  id: number | string;
  title: string | null;
  created_at: string | Date | null;
  updated_at: string | Date | null;
  user_id: number | string;
  email: string;
  full_name: string | null;
  role: string;
  llm_provider: string | null;
  model: string | null;
  message_count: number | string;
  total_tokens: number | string;
  last_message_at: string | Date | null;
  last_message_role: string | null;
  last_message_status: string | null;
  last_message_preview: string | null;
}

export class ChatSessionRepository extends DefaultCrudRepository<
  ChatSession,
  typeof ChatSession.prototype.id,
  ChatSessionRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(ChatSession, dataSource);
  }

  /**
   * Admin listing of chat sessions with owning user, message/token stats and
   * a preview of the last message. Aggregates across chat_sessions,
   * chat_messages and llm_calls, which the juggler query builder cannot
   * express, so this stays as raw SQL.
   */
  async findAdminSessions(search: string, limit: number): Promise<AdminSessionRow[]> {
    return (await this.dataSource.execute(
      `
        SELECT
          s.id,
          s.title,
          s.created_at,
          s.updated_at,
          u.id AS user_id,
          u.email,
          u.full_name,
          u.role,
          s.llm_provider,
          s.model,
          COALESCE(message_stats.message_count, 0) AS message_count,
          COALESCE(token_stats.total_tokens, 0) AS total_tokens,
          message_stats.last_message_at,
          last_message.role AS last_message_role,
          last_message.status AS last_message_status,
          last_message.content AS last_message_preview
        FROM chat_sessions s
        JOIN users u ON u.id = s.user_id
        LEFT JOIN (
          SELECT
            session_id,
            COUNT(id) AS message_count,
            MAX(created_at) AS last_message_at
          FROM chat_messages
          GROUP BY session_id
        ) message_stats ON message_stats.session_id = s.id
        LEFT JOIN (
          SELECT
            session_id,
            SUM(input_tokens + output_tokens + thinking_tokens) AS total_tokens
          FROM llm_calls
          GROUP BY session_id
        ) token_stats ON token_stats.session_id = s.id
        LEFT JOIN LATERAL (
          SELECT role, status, left(content, 220) AS content
          FROM chat_messages lm
          WHERE lm.session_id = s.id
          ORDER BY lm.created_at DESC, lm.id DESC
          LIMIT 1
        ) last_message ON true
        WHERE ($1 = '' OR u.email ILIKE $2 OR u.full_name ILIKE $2 OR s.title ILIKE $2)
        ORDER BY COALESCE(message_stats.last_message_at, s.updated_at, s.created_at) DESC
        LIMIT $3
      `,
      [search, `%${search}%`, limit],
    )) as AdminSessionRow[];
  }
}
