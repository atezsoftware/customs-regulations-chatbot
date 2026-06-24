import {useEffect, useRef, useState} from 'react';
import type {Source} from '../../types';

export function StreamingAnswer({
  content,
  streaming,
  sources = [],
}: {
  content: string;
  streaming?: boolean;
  sources?: Source[];
}) {
  const blocks = parseMarkdownBlocks(content);
  return (
    <div className="space-y-3 text-sm leading-6 text-slate-700">
      {blocks.map((block, index) => (
        <MarkdownBlock key={`${block.type}-${index}`} block={block} sources={sources} />
      ))}
      {streaming && <span className="ml-1 inline-block h-4 w-1 animate-pulse bg-indigo-500 align-middle" />}
    </div>
  );
}

type MarkdownBlock =
  | {type: 'heading'; level: number; text: string}
  | {type: 'paragraph'; text: string}
  | {type: 'ordered'; items: string[]}
  | {type: 'unordered'; items: string[]};

function MarkdownBlock({block, sources}: {block: MarkdownBlock; sources: Source[]}) {
  if (block.type === 'heading') {
    const className =
      block.level <= 2
        ? 'pt-1 text-base font-semibold text-slate-900'
        : 'pt-1 text-sm font-semibold text-slate-800';
    return <h3 className={className}>{renderInline(block.text, sources)}</h3>;
  }

  if (block.type === 'ordered') {
    return (
      <ol className="list-decimal space-y-2 pl-5">
        {block.items.map((item, index) => (
          <li key={index}>{renderInline(item, sources)}</li>
        ))}
      </ol>
    );
  }

  if (block.type === 'unordered') {
    return (
      <ul className="list-disc space-y-1.5 pl-5">
        {block.items.map((item, index) => (
          <li key={index}>{renderInline(item, sources)}</li>
        ))}
      </ul>
    );
  }

  return <p>{renderInline(block.text, sources)}</p>;
}

function parseMarkdownBlocks(content: string): MarkdownBlock[] {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const blocks: MarkdownBlock[] = [];
  let paragraph: string[] = [];
  let ordered: string[] = [];
  let unordered: string[] = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push({type: 'paragraph', text: paragraph.join(' ').trim()});
    paragraph = [];
  };
  const flushOrdered = () => {
    if (!ordered.length) return;
    blocks.push({type: 'ordered', items: ordered});
    ordered = [];
  };
  const flushUnordered = () => {
    if (!unordered.length) return;
    blocks.push({type: 'unordered', items: unordered});
    unordered = [];
  };
  const flushLists = () => {
    flushOrdered();
    flushUnordered();
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushLists();
      continue;
    }

    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushLists();
      blocks.push({type: 'heading', level: heading[1].length, text: heading[2]});
      continue;
    }

    const orderedMatch = /^\d+\.\s+(.+)$/.exec(line);
    if (orderedMatch) {
      flushParagraph();
      flushUnordered();
      ordered.push(orderedMatch[1]);
      continue;
    }

    const unorderedMatch = /^[-*]\s+(.+)$/.exec(line);
    if (unorderedMatch) {
      flushParagraph();
      flushOrdered();
      unordered.push(unorderedMatch[1]);
      continue;
    }

    flushLists();
    paragraph.push(line);
  }

  flushParagraph();
  flushLists();
  return blocks.length ? blocks : [{type: 'paragraph', text: content}];
}

function renderInline(text: string, sources: Source[]) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[[^\]]+\])/g).filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={index} className="font-semibold text-slate-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('[') && part.endsWith(']')) {
      return <CitationChip key={index} label={part.slice(1, -1)} sources={sources} />;
    }
    return <span key={index}>{part}</span>;
  });
}

function CitationChip({label, sources}: {label: string; sources: Source[]}) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);
  const matches = findSourcesForCitation(label, sources);
  const hasSources = matches.length > 0;

  useEffect(
    () => () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    },
    [],
  );

  function clearTimer() {
    if (!timerRef.current) return;
    window.clearTimeout(timerRef.current);
    timerRef.current = undefined;
  }

  function handleMouseEnter() {
    if (!hasSources) return;
    clearTimer();
    timerRef.current = window.setTimeout(() => setOpen(true), 1000);
  }

  function handleMouseLeave() {
    clearTimer();
    if (!pinned) setOpen(false);
  }

  function handleClick() {
    if (!hasSources) return;
    clearTimer();
    const nextPinned = !pinned;
    setPinned(nextPinned);
    setOpen(nextPinned);
  }

  return (
    <span
      className="relative mx-0.5 inline-flex align-baseline"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={() => hasSources && setOpen(true)}
      onBlur={() => !pinned && setOpen(false)}
    >
      <button
        type="button"
        className={[
          'inline rounded-md border-b border-indigo-200 bg-indigo-50 px-1.5 py-0.5 text-xs font-medium text-indigo-700 transition',
          hasSources ? 'cursor-pointer hover:bg-indigo-100' : 'cursor-default',
        ].join(' ')}
        onClick={handleClick}
      >
        {label}
      </button>
      {open && hasSources && (
        <span className="absolute left-1/2 top-full z-40 w-96 max-w-[calc(100vw-2rem)] -translate-x-1/2 pt-2">
          <span className="block max-h-80 overflow-y-auto rounded-lg border border-slate-200 bg-white p-3 text-left shadow-xl ring-1 ring-slate-900/5">
            {matches.map((source, index) => (
              <span
                key={source.id}
                className={index === 0 ? 'block' : 'mt-3 block border-t border-slate-100 pt-3'}
              >
                <span className="block text-xs font-semibold leading-5 text-slate-900">
                  {source.title}
                </span>
                <span className="mt-1 block whitespace-pre-wrap text-xs leading-5 text-slate-600">
                  {source.snippet || 'Chunk text is not available for this older source.'}
                </span>
              </span>
            ))}
          </span>
        </span>
      )}
    </span>
  );
}

function findSourcesForCitation(label: string, sources: Source[]): Source[] {
  const normalized = normalizeCitation(label);
  if (!normalized) return [];
  return sources
    .map(source => ({source, score: sourceMatchScore(normalized, source)}))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .map(item => item.source);
}

function sourceMatchScore(normalizedLabel: string, source: Source): number {
  const normalizedTitle = normalizeCitation(source.title);
  if (!normalizedTitle) return 0;
  if (normalizedLabel === normalizedTitle) return 100;
  if (normalizedLabel.includes(normalizedTitle)) return 90;
  if (normalizedTitle.includes(normalizedLabel)) return 80;

  const overlap = tokenOverlap(normalizedLabel, normalizedTitle);
  if (overlap >= 4) return 20 + overlap;
  return 0;
}

function tokenOverlap(a: string, b: string): number {
  const ignored = new Set(['source', 'sources', 'madde', 'article', 'section', 'chunk']);
  const aTokens = new Set(a.split(' ').filter(token => token && !ignored.has(token)));
  const bTokens = new Set(b.split(' ').filter(token => token && !ignored.has(token)));
  let overlap = 0;
  for (const token of aTokens) {
    if (bTokens.has(token)) overlap += 1;
  }
  return overlap;
}

function normalizeCitation(value: string): string {
  return value
    .trim()
    .replace(/^source:\s*/i, '')
    .replace(/ı/g, 'i')
    .replace(/İ/g, 'I')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}
