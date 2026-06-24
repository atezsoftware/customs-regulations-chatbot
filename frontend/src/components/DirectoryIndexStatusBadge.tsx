import type {DirectoryIndexStatus} from '../types';

const LABELS: Record<DirectoryIndexStatus['status'], string> = {
  not_indexed: 'Not indexed',
  chunking: 'Generating chunks',
  chunked: 'Chunks ready',
  indexing: 'Indexing',
  completed: 'Indexed',
  stale: 'Stale',
  error: 'Error',
  unavailable: 'Unavailable',
};

const CLASSES: Record<DirectoryIndexStatus['status'], string> = {
  not_indexed: 'bg-slate-100 text-slate-500',
  chunking: 'bg-indigo-50 text-indigo-600',
  chunked: 'bg-sky-50 text-sky-700',
  indexing: 'bg-indigo-50 text-indigo-600',
  completed: 'bg-emerald-50 text-emerald-700',
  stale: 'bg-amber-50 text-amber-700',
  error: 'bg-rose-50 text-rose-600',
  unavailable: 'bg-slate-100 text-slate-500',
};

export function DirectoryIndexStatusBadge({status}: {status?: DirectoryIndexStatus}) {
  if (!status) {
    return <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-400">Unknown</span>;
  }
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${CLASSES[status.status]}`}>
      {LABELS[status.status]}
    </span>
  );
}

export function DirectoryIndexProgress({status}: {status?: DirectoryIndexStatus}) {
  if (!status) return null;
  const progress = Math.max(0, Math.min(100, status.progress));
  return (
    <div className="space-y-2">
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full ${
            status.status === 'error' ? 'bg-rose-500' : 'bg-indigo-500'
          }`}
          style={{width: `${progress}%`}}
        />
      </div>
      <p className="text-xs leading-5 text-slate-500">{status.message}</p>
      {status.skippedFiles && status.skippedFiles.length > 0 && (
        <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-700">
          <p className="font-medium">Skipped files</p>
          <ul className="mt-1 list-disc space-y-1 pl-4">
            {status.skippedFiles.slice(0, 5).map(file => (
              <li key={file}>{file}</li>
            ))}
          </ul>
          {status.skippedFiles.length > 5 && (
            <p className="mt-1">+{status.skippedFiles.length - 5} more</p>
          )}
        </div>
      )}
    </div>
  );
}
