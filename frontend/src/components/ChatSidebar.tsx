import type {ChatSession} from '../types';
import {Button} from './ui/Button';

export function ChatSidebar({
  sessions,
  selectedId,
  onSelect,
  onCreate,
  creating,
}: {
  sessions: ChatSession[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onCreate: () => void;
  creating: boolean;
}) {
  return (
    <aside className="flex w-72 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-100 p-3">
        <Button onClick={onCreate} disabled={creating} className="w-full">
          <PlusIcon />
          {creating ? 'Creating…' : 'New chat'}
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {sessions.length === 0 && (
          <p className="px-2 py-6 text-center text-sm text-slate-400">
            No chats yet — start one above.
          </p>
        )}
        <ul className="space-y-1">
          {sessions.map(session => {
            const createdAt = formatSessionDateTime(session.createdAt);
            return (
              <li key={session.id}>
                <button
                  onClick={() => onSelect(session.id)}
                  className={`w-full rounded-xl px-3 py-2.5 text-left text-sm transition-colors ${
                    session.id === selectedId
                      ? 'bg-indigo-50 font-medium text-indigo-700'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <span className="block truncate">{session.title || 'Untitled chat'}</span>
                  {createdAt && (
                    <span
                      className={`mt-0.5 block text-xs font-normal ${
                        session.id === selectedId ? 'text-indigo-500' : 'text-slate-400'
                      }`}
                    >
                      {createdAt}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}

function formatSessionDateTime(value?: string): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 5v14M5 12h14" strokeLinecap="round" />
    </svg>
  );
}
