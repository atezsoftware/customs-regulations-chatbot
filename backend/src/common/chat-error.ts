import {stripNulBytes} from './text';

const MAX_CHAT_ERROR_LENGTH = 2000;

export function formatChatError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error ?? '');
  const message = stripNulBytes(raw).trim();
  return (message || 'Unknown chat error.').slice(0, MAX_CHAT_ERROR_LENGTH);
}
