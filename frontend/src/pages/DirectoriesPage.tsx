import {useCallback, useEffect, useRef, useState} from 'react';
import type {ReactNode} from 'react';
import {DirectoryIndexProgress, DirectoryIndexStatusBadge} from '../components/DirectoryIndexStatusBadge';
import {Button} from '../components/ui/Button';
import {ConfirmModal} from '../components/ui/ConfirmModal';
import {NamePromptModal} from '../components/ui/NamePromptModal';
import {Spinner} from '../components/ui/Spinner';
import {useAuth} from '../context/useAuth';
import {directoriesApi} from '../lib/endpoints';
import {formatBytes} from '../lib/format';
import type {Directory, DirectoryDetail, DirectoryIndexStatus} from '../types';

export function DirectoriesPage() {
  const {user} = useAuth();
  const [directories, setDirectories] = useState<Directory[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<DirectoryDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [chunking, setChunking] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState<DirectoryIndexStatus | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [renaming, setRenaming] = useState<{kind: 'directory' | 'file'; id: number; name: string} | null>(
    null,
  );
  const [deleting, setDeleting] = useState<{kind: 'directory' | 'file'; id: number} | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadList = useCallback(async () => {
    setLoadingList(true);
    try {
      setDirectories(await directoriesApi.list());
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadList();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadList]);

  const loadDetail = useCallback(async (id: number) => {
    setLoadingDetail(true);
    try {
      const [nextDetail, nextIndexStatus] = await Promise.all([
        directoriesApi.get(id),
        directoriesApi.indexStatus(id),
      ]);
      setDetail(nextDetail);
      setIndexStatus(nextIndexStatus);
    } finally {
      setLoadingDetail(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (selectedId !== null) void loadDetail(selectedId);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selectedId, loadDetail]);

  async function handleCreate(name: string) {
    const directory = await directoriesApi.create(name);
    setDirectories(prev => [directory, ...prev]);
    setSelectedId(directory.id);
    setShowCreate(false);
  }

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0 || selectedId === null) return;
    setUploading(true);
    try {
      await directoriesApi.upload(selectedId, Array.from(files));
      await loadDetail(selectedId);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  useEffect(() => {
    const busy = indexStatus?.status === 'chunking' || indexStatus?.status === 'indexing';
    if (selectedId === null || !busy) return;
    const timer = window.setInterval(() => {
      directoriesApi
        .indexStatus(selectedId)
        .then(setIndexStatus)
        .catch(() => {
          // Keep the last visible status; transient core/backend misses should not blank the UI.
        });
    }, 1500);
    return () => window.clearInterval(timer);
  }, [selectedId, indexStatus?.status]);

  async function handleStartChunking() {
    if (selectedId === null) return;
    setChunking(true);
    setIndexError(null);
    try {
      const started = await directoriesApi.startChunking(selectedId);
      setIndexStatus(started);
    } catch (error) {
      setIndexError(error instanceof Error ? error.message : String(error));
    } finally {
      setChunking(false);
    }
  }

  async function handleStartIndexing() {
    if (selectedId === null) return;
    setIndexing(true);
    setIndexError(null);
    try {
      const started = await directoriesApi.startIndex(selectedId);
      setIndexStatus(started);
    } catch (error) {
      setIndexError(error instanceof Error ? error.message : String(error));
    } finally {
      setIndexing(false);
    }
  }

  async function handleRenameSubmit(name: string) {
    if (!renaming) return;
    if (renaming.kind === 'directory') {
      await directoriesApi.rename(renaming.id, name);
      setDirectories(prev => prev.map(d => (d.id === renaming.id ? {...d, name} : d)));
      setDetail(prev => (prev && prev.id === renaming.id ? {...prev, name} : prev));
    } else if (selectedId !== null) {
      await directoriesApi.renameFile(selectedId, renaming.id, name);
      await loadDetail(selectedId);
    }
    setRenaming(null);
  }

  async function handleDeleteConfirm() {
    if (!deleting) return;
    setDeleteBusy(true);
    try {
      if (deleting.kind === 'directory') {
        await directoriesApi.remove(deleting.id);
        setDirectories(prev => prev.filter(d => d.id !== deleting.id));
        if (selectedId === deleting.id) {
          setSelectedId(null);
          setDetail(null);
        }
      } else if (selectedId !== null) {
        await directoriesApi.removeFile(selectedId, deleting.id);
        await loadDetail(selectedId);
      }
      setDeleting(null);
    } finally {
      setDeleteBusy(false);
    }
  }

  return (
    <>
      <aside className="flex w-72 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-100 p-3">
          <Button onClick={() => setShowCreate(true)} className="w-full">
            <PlusIcon />
            New directory
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {loadingList ? (
            <div className="flex justify-center py-8 text-slate-400">
              <Spinner />
            </div>
          ) : directories.length === 0 ? (
            <p className="px-2 py-6 text-center text-sm text-slate-400">
              No directories yet — create one above.
            </p>
          ) : (
            <ul className="space-y-1">
              {directories.map(dir => (
                <li key={dir.id} className="group relative">
                  <button
                    onClick={() => setSelectedId(dir.id)}
                    className={`flex w-full items-center justify-between truncate rounded-xl px-3 py-2.5 text-left text-sm transition-colors ${
                      dir.id === selectedId
                        ? 'bg-indigo-50 font-medium text-indigo-700'
                        : 'text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    <span className="truncate">{dir.name}</span>
                  </button>
                  <div className="absolute right-1.5 top-1.5 hidden gap-1 group-hover:flex">
                    <IconButton
                      title="Rename"
                      onClick={() => setRenaming({kind: 'directory', id: dir.id, name: dir.name})}
                    >
                      <PencilIcon />
                    </IconButton>
                    <IconButton
                      title="Delete"
                      danger
                      onClick={() => setDeleting({kind: 'directory', id: dir.id})}
                    >
                      <TrashIcon />
                    </IconButton>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto">
        {selectedId === null ? (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center">
            <p className="text-lg font-medium text-slate-700">Pick a directory</p>
            <p className="mt-1 max-w-sm text-sm text-slate-400">
              Or create one to start uploading files.
            </p>
          </div>
        ) : loadingDetail || !detail ? (
          <div className="flex h-full items-center justify-center text-slate-400">
            <Spinner />
          </div>
        ) : (
          <div className="mx-auto max-w-2xl px-6 py-8">
            <h1 className="mb-6 text-xl font-semibold text-slate-900">{detail.name}</h1>

            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-slate-700">
                  Files ({detail.files.length})
                </h3>
                <DirectoryIndexStatusBadge status={indexStatus ?? undefined} />
              </div>
              {detail.files.length === 0 ? (
                <p className="text-sm text-slate-400">No files yet — add some below.</p>
              ) : (
                <ul className="max-h-80 divide-y divide-slate-100 overflow-y-auto pr-1">
                  {detail.files.map(file => (
                    <li
                      key={file.id}
                      className="group flex items-center justify-between py-2.5 text-sm"
                    >
                      <div className="flex min-w-0 items-center gap-2.5">
                        <FileIcon />
                        <span className="truncate text-slate-700">{file.name}</span>
                        <span className="shrink-0 text-xs text-slate-400">
                          {formatBytes(file.sizeBytes)}
                        </span>
                      </div>
                      <div className="hidden shrink-0 gap-1 group-hover:flex">
                        <IconButton
                          title="Rename"
                          onClick={() =>
                            setRenaming({kind: 'file', id: file.id, name: file.name})
                          }
                        >
                          <PencilIcon />
                        </IconButton>
                        <IconButton
                          title="Delete"
                          danger
                          onClick={() => setDeleting({kind: 'file', id: file.id})}
                        >
                          <TrashIcon />
                        </IconButton>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-700">Search index</h3>
                  <p className="mt-1 text-xs leading-5 text-slate-400">
                    Generate regulatory chunks first, then index them to build Gemini
                    embeddings for this directory.
                  </p>
                </div>
                <div className="flex shrink-0 gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    disabled={
                      chunking ||
                      indexStatus?.status === 'chunking' ||
                      indexStatus?.status === 'indexing' ||
                      detail.files.length === 0
                    }
                    onClick={() => void handleStartChunking()}
                  >
                    {indexStatus?.status === 'chunking'
                      ? 'Generating chunks...'
                      : indexStatus?.status === 'chunked' ||
                          indexStatus?.status === 'completed' ||
                          indexStatus?.status === 'stale'
                        ? 'Regenerate chunks'
                        : 'Generate chunks'}
                  </Button>
                  <Button
                    type="button"
                    variant={
                      indexStatus?.status === 'completed' || indexStatus?.status === 'indexing'
                        ? 'secondary'
                        : 'primary'
                    }
                    disabled={
                      indexing ||
                      chunking ||
                      indexStatus?.status === 'chunking' ||
                      indexStatus?.status === 'indexing' ||
                      !(
                        indexStatus?.status === 'chunked' ||
                        indexStatus?.status === 'completed' ||
                        indexStatus?.status === 'stale'
                      )
                    }
                    onClick={() => void handleStartIndexing()}
                  >
                    {indexStatus?.status === 'indexing'
                      ? 'Indexing...'
                      : indexStatus?.status === 'completed'
                        ? 'Re-index'
                        : 'Start indexing'}
                  </Button>
                </div>
              </div>
              <div className="mt-4">
                <DirectoryIndexProgress status={indexStatus ?? undefined} />
                {indexError && (
                  <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">
                    {indexError}
                  </p>
                )}
              </div>
            </div>

            {user?.uploadsEnabled && (
              <label className="mt-4 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/60 px-4 py-8 text-center transition-colors hover:border-indigo-300 hover:bg-indigo-50/40">
                {uploading ? (
                  <Spinner className="h-5 w-5 text-indigo-500" />
                ) : (
                  <UploadIcon />
                )}
                <span className="text-sm font-medium text-slate-600">
                  {uploading ? 'Uploading…' : 'Click to add files'}
                </span>
                <span className="text-xs text-slate-400">Any file type, multiple at once</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  disabled={uploading}
                  className="hidden"
                  onChange={e => void handleUpload(e.target.files)}
                />
              </label>
            )}
          </div>
        )}
      </main>

      {showCreate && (
        <NamePromptModal
          title="New directory"
          label="Name"
          confirmLabel="Create"
          onSubmit={handleCreate}
          onCancel={() => setShowCreate(false)}
        />
      )}

      {renaming && (
        <NamePromptModal
          title={renaming.kind === 'directory' ? 'Rename directory' : 'Rename file'}
          label="Name"
          initialValue={renaming.name}
          onSubmit={handleRenameSubmit}
          onCancel={() => setRenaming(null)}
        />
      )}

      {deleting && (
        <ConfirmModal
          title={deleting.kind === 'directory' ? 'Delete directory?' : 'Delete file?'}
          message={
            deleting.kind === 'directory'
              ? 'This deletes the directory and every file inside it. This cannot be undone.'
              : 'This file will be permanently deleted.'
          }
          busy={deleteBusy}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleting(null)}
        />
      )}
    </>
  );
}

function IconButton({
  children,
  title,
  danger,
  onClick,
}: {
  children: ReactNode;
  title: string;
  danger?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      title={title}
      onClick={e => {
        e.stopPropagation();
        onClick();
      }}
      className={`flex h-7 w-7 items-center justify-center rounded-lg transition-colors ${
        danger
          ? 'text-slate-400 hover:bg-rose-50 hover:text-rose-500'
          : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600'
      }`}
    >
      {children}
    </button>
  );
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 5v14M5 12h14" strokeLinecap="round" />
    </svg>
  );
}
function PencilIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path
        d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path
        d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function FileIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      className="shrink-0 text-slate-400"
    >
      <path
        d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M14 2v6h6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function UploadIcon() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      className="text-slate-400"
    >
      <path
        d="M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
