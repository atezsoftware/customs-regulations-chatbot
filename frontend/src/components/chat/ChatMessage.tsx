import type {ChatMessageRecord} from '../../types';
import {AssistantMessage} from './AssistantMessage';

export function ChatMessage({
  message,
  previousUserContent,
  onStop,
  onRegenerate,
}: {
  message: ChatMessageRecord;
  previousUserContent?: string;
  onStop: () => void;
  onRegenerate: (content: string) => void;
}) {
  if (message.role === 'user') {
    return (
      <article className="ml-auto max-w-[82%] rounded-xl bg-indigo-600 px-4 py-3 text-sm leading-6 text-white shadow-sm shadow-indigo-600/20">
        <p className="whitespace-pre-wrap">{message.content}</p>
      </article>
    );
  }

  return (
    <AssistantMessage
      message={message}
      onStop={onStop}
      onRegenerate={() => previousUserContent && onRegenerate(previousUserContent)}
    />
  );
}
