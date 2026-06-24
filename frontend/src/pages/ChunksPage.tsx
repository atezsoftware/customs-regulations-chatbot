import {useEffect, useMemo, useState} from 'react';
import {directoriesApi} from '../lib/endpoints';
import {formatBytes} from '../lib/format';
import type {DirectoryDetail, DirectoryFile, FileChunksResponse, IndexedChunk} from '../types';

interface SelectedFile {
  directoryId: number;
  file: DirectoryFile;
}

export function ChunksPage() {
  const [directories, setDirectories] = useState<DirectoryDetail[]>([]);
  const [selected, setSelected] = useState<SelectedFile | null>(null);
  const [chunkData, setChunkData] = useState<FileChunksResponse | null>(null);
  const [openChunkId, setOpenChunkId] = useState<string | null>(null);
  const [loadingTree, setLoadingTree] = useState(true);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    directoriesApi
      .list()
      .then(items => Promise.all(items.map(item => directoriesApi.get(item.id))))
      .then(details => {
        if (cancelled) return;
        setDirectories(details);
        const firstIndexed = details
          .flatMap(directory =>
            directory.files.map(file => ({directoryId: directory.id, file})),
          )
          .find(item => item.file.storageStatus === 'indexed');
        setSelected(firstIndexed ?? null);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoadingTree(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selected) {
      return;
    }
    let cancelled = false;
    Promise.resolve()
      .then(() => {
        if (cancelled) return null;
        setLoadingChunks(true);
        setOpenChunkId(null);
        return directoriesApi.fileChunks(selected.directoryId, selected.file.id);
      })
      .then(data => {
        if (!cancelled && data) setChunkData(data);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoadingChunks(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const stats = useMemo(() => {
    const chunks = chunkData?.chunks ?? [];
    return {
      chunks: chunks.length,
      embedded: chunks.filter(chunk => chunk.hasEmbedding).length,
      chars: chunks.reduce((sum, chunk) => sum + chunk.text.length, 0),
    };
  }, [chunkData]);

  return (
    <div className="flex min-h-0 flex-1">
      <aside className="flex w-80 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-100 px-4 py-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            View chunks
          </p>
          <h1 className="mt-1 text-lg font-semibold text-slate-900">Indexed documents</h1>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
          {loadingTree ? (
            <p className="px-2 py-6 text-sm text-slate-400">Loading directories...</p>
          ) : directories.length === 0 ? (
            <p className="px-2 py-6 text-sm text-slate-400">No directories yet.</p>
          ) : (
            <div className="space-y-3">
              {directories.map(directory => (
                <section key={directory.id}>
                  <div className="mb-1 flex items-center justify-between gap-2 px-2">
                    <p className="truncate text-sm font-semibold text-slate-700">
                      {directory.name}
                    </p>
                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-500">
                      {directory.files.length}
                    </span>
                  </div>
                  <div className="space-y-1">
                    {directory.files.map(file => {
                      const active =
                        selected?.directoryId === directory.id && selected.file.id === file.id;
                      const indexed = file.storageStatus === 'indexed';
                      return (
                        <button
                          key={file.id}
                          type="button"
                          onClick={() => setSelected({directoryId: directory.id, file})}
                          className={`w-full rounded-xl border px-3 py-2 text-left transition duration-200 ${
                            active
                              ? 'border-indigo-200 bg-indigo-50 shadow-sm'
                              : 'border-transparent hover:border-slate-200 hover:bg-slate-50'
                          }`}
                        >
                          <span className="flex items-center justify-between gap-2">
                            <span className="min-w-0 truncate text-sm font-medium text-slate-700">
                              {file.name}
                            </span>
                            <span
                              className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                                indexed
                                  ? 'bg-emerald-50 text-emerald-700'
                                  : file.storageStatus === 'error'
                                    ? 'bg-rose-50 text-rose-700'
                                    : 'bg-amber-50 text-amber-700'
                              }`}
                            >
                              {indexed ? 'indexed' : file.storageStatus ?? 'stored'}
                            </span>
                          </span>
                          <span className="mt-1 block text-xs text-slate-400">
                            {formatBytes(file.sizeBytes)}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-y-auto bg-slate-50">
        <div className="mx-auto max-w-6xl px-6 py-6">
          {error && (
            <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {error}
            </div>
          )}

          {!selected ? (
            <EmptyChunksState />
          ) : (
            <>
              <div className="mb-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Selected file
                    </p>
                    <h2 className="mt-1 truncate text-xl font-semibold text-slate-900">
                      {selected.file.name}
                    </h2>
                    {chunkData?.document && (
                      <p className="mt-1 text-sm text-slate-500">
                        {chunkData.document.title}
                      </p>
                    )}
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <Metric label="Chunks" value={loadingChunks ? '...' : String(stats.chunks)} />
                    <Metric label="Embedded" value={loadingChunks ? '...' : `${stats.embedded}/${stats.chunks}`} />
                    <Metric label="Chars" value={loadingChunks ? '...' : stats.chars.toLocaleString()} />
                  </div>
                </div>
              </div>

              {loadingChunks ? (
                <div className="rounded-xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-400">
                  Loading chunks...
                </div>
              ) : chunkData && chunkData.chunks.length > 0 ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-[88px_120px_160px_120px_1fr] gap-3 px-4 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    <span>Position</span>
                    <span>Type</span>
                    <span>Chars</span>
                    <span>Embedding</span>
                    <span>Preview</span>
                  </div>
                  {chunkData.chunks.map(chunk => (
                    <ChunkRow
                      key={chunk.id}
                      chunk={chunk}
                      open={openChunkId === chunk.id}
                      onToggle={() =>
                        setOpenChunkId(current => (current === chunk.id ? null : chunk.id))
                      }
                    />
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
                  <p className="text-sm font-medium text-slate-700">No chunks found.</p>
                  <p className="mt-1 text-sm text-slate-400">
                    Start indexing this directory first, then come back here.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function Metric({label, value}: {label: string; value: string}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="mt-0.5 text-sm font-semibold text-slate-800">{value}</p>
    </div>
  );
}

function ChunkRow({
  chunk,
  open,
  onToggle,
}: {
  chunk: IndexedChunk;
  open: boolean;
  onToggle: () => void;
}) {
  const json = {
    id: chunk.id,
    document_id: chunk.documentId,
    relative_path: chunk.relativePath,
    document_title: chunk.documentTitle,
    text: chunk.text,
    position: chunk.position,
    start_char: chunk.startChar,
    end_char: chunk.endChar,
    chunk_type: chunk.chunkType,
    metadata: chunk.metadata,
    has_embedding: chunk.hasEmbedding,
  };

  return (
    <article className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm transition duration-200 hover:-translate-y-0.5 hover:shadow-md">
      <button
        type="button"
        onClick={onToggle}
        className="grid w-full grid-cols-[88px_120px_160px_120px_1fr] gap-3 px-4 py-3 text-left"
      >
        <span className="text-sm font-semibold text-slate-800">#{chunk.position}</span>
        <span className="truncate text-sm text-slate-600">{chunk.chunkType ?? 'text'}</span>
        <span className="text-sm text-slate-500">
          {chunk.startChar}-{chunk.endChar}
        </span>
        <span
          className={`w-fit rounded-full px-2 py-0.5 text-xs font-medium ${
            chunk.hasEmbedding ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-500'
          }`}
        >
          {chunk.hasEmbedding ? 'indexed' : 'missing'}
        </span>
        <span className="line-clamp-2 text-sm leading-5 text-slate-600">
          {chunk.text}
        </span>
      </button>

      {open && (
        <div className="animate-[fadeIn_180ms_ease-out] border-t border-slate-100 bg-slate-950 p-4">
          <pre className="max-h-[520px] overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-100">
            {JSON.stringify(json, null, 2)}
          </pre>
        </div>
      )}
    </article>
  );
}

function EmptyChunksState() {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-10 text-center shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">Select an indexed file</h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">
        Pick a file from the directory tree to inspect the chunks stored in PostgreSQL.
        Embedding vectors stay hidden, but each row shows whether an embedding exists.
      </p>
    </div>
  );
}
