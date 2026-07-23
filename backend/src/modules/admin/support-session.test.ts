import assert from 'node:assert/strict';
import test from 'node:test';
import type {AdminSessionRow} from '../chat/repositories/chat-session.repository';
import {toAdminSupportSession} from './support-session';

test('includes the chat model selection in an admin support session', () => {
  const row: AdminSessionRow = {
    id: 119,
    title: 'Transit question',
    created_at: '2026-07-23T10:00:00.000Z',
    updated_at: '2026-07-23T10:01:00.000Z',
    user_id: 8,
    email: 'onur@example.com',
    full_name: 'Onur',
    role: 'user',
    llm_provider: 'openrouter',
    model: 'google/gemini-3-flash-preview',
    message_count: 2,
    total_tokens: 120,
    last_message_at: '2026-07-23T10:01:00.000Z',
    last_message_role: 'assistant',
    last_message_status: 'completed',
    last_message_preview: 'Answer',
  };

  assert.deepEqual(toAdminSupportSession(row).model, {
    provider: 'openrouter',
    modelId: 'google/gemini-3-flash-preview',
  });
});
