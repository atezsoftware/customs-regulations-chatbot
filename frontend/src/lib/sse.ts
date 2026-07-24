import type {AgentEvent, ChatEffortLevel} from '../types';
import {API_BASE_URL, fetchWithAuthRetry} from './api';

export async function streamMessageEvents({
  sessionId,
  messageId,
  signal,
  onEvent,
  resumeRunId,
  effort,
}: {
  sessionId: number;
  messageId: number;
  signal: AbortSignal;
  onEvent: (event: AgentEvent) => void;
  // Continues a run core-api is still holding onto as resumable instead of
  // starting a brand-new one for this message — see AssistantMessage's
  // "Continue" button.
  resumeRunId?: string;
  // Retrieval breadth for a fresh run — ignored server-side on a resume,
  // since core-api already has the original run's effort.
  effort?: ChatEffortLevel;
}) {
  const url = new URL(
    `${API_BASE_URL}/chat-sessions/${sessionId}/messages/${messageId}/stream`,
  );
  if (resumeRunId) url.searchParams.set('resumeRunId', resumeRunId);
  if (effort) url.searchParams.set('effort', effort);

  const res = await fetchWithAuthRetry(url, {signal});
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
