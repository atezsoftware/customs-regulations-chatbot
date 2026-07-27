import {useCallback, useEffect, useRef, useState} from 'react';
import type {PointerEvent as ReactPointerEvent} from 'react';
import type {ChatSession} from '../types';
import {Button} from './ui/Button';

const WIDTH_KEY = 'cc_chat_sidebar_width';
const COLLAPSED_KEY = 'cc_chat_sidebar_collapsed';
const MIN_WIDTH = 220;
const MAX_WIDTH = 440;
const DEFAULT_WIDTH = 288;
const MOBILE_BREAKPOINT = 768;

function readStoredWidth(): number {
  const raw = Number(localStorage.getItem(WIDTH_KEY));
  return Number.isFinite(raw) && raw >= MIN_WIDTH && raw <= MAX_WIDTH ? raw : DEFAULT_WIDTH;
}

function readStoredCollapsed(): boolean {
  const stored = localStorage.getItem(COLLAPSED_KEY);
  if (stored !== null) return stored === 'true';
  // No explicit preference yet — default collapsed on narrow viewports so
  // the chat list doesn't eat the whole screen on first load. `innerWidth`
  // can transiently read 0 before the initial layout settles; treat that as
  // "unknown" rather than "narrow" so a slow first paint doesn't default to
  // collapsed on a normal desktop viewport.
  return typeof window !== 'undefined' && window.innerWidth > 0 && window.innerWidth < MOBILE_BREAKPOINT;
}

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
  const [width, setWidth] = useState(readStoredWidth);
  const [collapsed, setCollapsed] = useState(readStoredCollapsed);
  const draggingRef = useRef(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(width);

  useEffect(() => {
    localStorage.setItem(WIDTH_KEY, String(width));
  }, [width]);

  useEffect(() => {
    localStorage.setItem(COLLAPSED_KEY, String(collapsed));
  }, [collapsed]);

  // Listeners stay attached for the component's lifetime (a no-op while
  // `draggingRef` is false) rather than being added/removed per drag — that
  // sidesteps a self-referencing `stopDragging` callback needing to remove
  // its own listener from inside itself.
  useEffect(() => {
    function handlePointerMove(event: PointerEvent) {
      if (!draggingRef.current) return;
      const next = startWidthRef.current + (event.clientX - startXRef.current);
      setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, next)));
    }
    function stopDragging() {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      document.body.style.removeProperty('cursor');
      document.body.style.removeProperty('user-select');
    }
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopDragging);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopDragging);
    };
  }, []);

  const startDragging = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      draggingRef.current = true;
      startXRef.current = event.clientX;
      startWidthRef.current = width;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    },
    [width],
  );

  if (collapsed) {
    return (
      <aside className="flex w-14 shrink-0 flex-col items-center gap-2 border-r border-slate-200 bg-white py-3">
        <button
          onClick={() => setCollapsed(false)}
          title="Show chats"
          className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
        >
          <ChevronRightIcon />
        </button>
        <button
          onClick={onCreate}
          disabled={creating}
          title="New chat"
          className="flex h-9 w-9 items-center justify-center rounded-lg text-indigo-600 transition-colors hover:bg-indigo-50 disabled:opacity-50"
        >
          <PlusIcon />
        </button>
      </aside>
    );
  }

  return (
    <aside
      className="relative flex shrink-0 flex-col border-r border-slate-200 bg-white"
      style={{width, minWidth: MIN_WIDTH, maxWidth: MAX_WIDTH}}
    >
      <div className="flex items-center gap-2 border-b border-slate-100 p-3">
        <Button onClick={onCreate} disabled={creating} className="flex-1">
          <PlusIcon />
          {creating ? 'Creating…' : 'New chat'}
        </Button>
        <button
          onClick={() => setCollapsed(true)}
          title="Hide chats"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
        >
          <ChevronLeftIcon />
        </button>
      </div>
      <div className="min-w-0 flex-1 overflow-y-auto p-2">
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
                  className={`w-full min-w-0 rounded-xl px-3 py-2.5 text-left text-sm transition-colors ${
                    session.id === selectedId
                      ? 'bg-indigo-50 font-medium text-indigo-700'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <span className="block truncate">{session.title || 'Untitled chat'}</span>
                  {createdAt && (
                    <span
                      className={`mt-0.5 block truncate text-xs font-normal ${
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
      <div
        onPointerDown={startDragging}
        title="Drag to resize"
        className="absolute -right-1.5 top-0 z-10 h-full w-3 cursor-col-resize touch-none"
      >
        <div className="mx-auto h-full w-px bg-transparent transition-colors hover:bg-indigo-300" />
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

function ChevronLeftIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M15 6l-6 6 6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M9 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
