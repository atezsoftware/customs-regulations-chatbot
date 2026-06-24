import type {AgentEvent} from '../types';
import {API_BASE_URL} from './api';
import {tokenStore} from './tokenStore';

export async function streamMessageEvents({
  sessionId,
  messageId,
  signal,
  onEvent,
}: {
  sessionId: number;
  messageId: number;
  signal: AbortSignal;
  onEvent: (event: AgentEvent) => void;
}) {
  const headers: Record<string, string> = {};
  const access = tokenStore.getAccess();
  if (access) headers.Authorization = `Bearer ${access}`;

  const res = await fetch(
    `${API_BASE_URL}/chat-sessions/${sessionId}/messages/${messageId}/stream`,
    {headers, signal},
  );
  if (!res.ok) {
    throw new Error(res.statusText || `Stream failed with ${res.status}`);
  }
  if (!res.body) {
    throw new Error('This browser does not support streaming responses.');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});

    let frameEnd = buffer.indexOf('\n\n');
    while (frameEnd >= 0) {
      const frame = buffer.slice(0, frameEnd);
      buffer = buffer.slice(frameEnd + 2);
      const data = frame
        .split('\n')
        .filter(line => line.startsWith('data:'))
        .map(line => line.slice('data:'.length).trimStart())
        .join('\n');
      if (data) onEvent(JSON.parse(data) as AgentEvent);
      frameEnd = buffer.indexOf('\n\n');
    }
  }
}
