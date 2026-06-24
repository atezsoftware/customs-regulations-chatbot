import type {Source} from '../../types';

export function SourceList({sources}: {sources: Source[]}) {
  if (!sources.length) return null;
  return (
    <div className="mt-4">
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        Sources
      </h4>
      <div className="grid gap-2 sm:grid-cols-2">
        {sources.map(source => (
          <SourceCard key={source.id} source={source} />
        ))}
      </div>
    </div>
  );
}

function SourceCard({source}: {source: Source}) {
  const body = source.snippet;
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:border-indigo-200 hover:shadow-md">
      <p className="truncate text-sm font-medium text-slate-800">{source.title}</p>
      {(source.page || source.chunkId) && (
        <p className="mt-0.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
          {source.page ? `Page ${source.page}` : 'Indexed chunk'}
        </p>
      )}
      {body && <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-500">{body}</p>}
    </div>
  );
}
