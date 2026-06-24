import {useState} from 'react';
import type {Directory, DirectoryIndexStatus} from '../types';
import {DirectoryIndexStatusBadge} from './DirectoryIndexStatusBadge';
import {Button} from './ui/Button';

export function LinkedDirectoriesPanel({
  allDirectories,
  linkedIds,
  onSave,
  saving,
  indexStatuses,
}: {
  allDirectories: Directory[];
  linkedIds: number[];
  onSave: (ids: number[]) => void;
  saving: boolean;
  indexStatuses?: Record<number, DirectoryIndexStatus>;
}) {
  const linkedKey = [...linkedIds].sort((a, b) => a - b).join(',');
  const [draft, setDraft] = useState<{key: string; selected: Set<number>}>(() => ({
    key: linkedKey,
    selected: new Set(linkedIds),
  }));
  const selected = draft.key === linkedKey ? draft.selected : new Set(linkedIds);

  const dirty =
    selected.size !== linkedIds.length || linkedIds.some(id => !selected.has(id));

  function toggle(id: number) {
    setDraft(() => {
      const next = new Set(selected);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return {key: linkedKey, selected: next};
    });
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">Linked directories</h3>
        {dirty && (
          <Button
            variant="primary"
            className="px-3 py-1.5 text-xs"
            disabled={saving}
            onClick={() => onSave([...selected])}
          >
            {saving ? 'Saving…' : 'Save'}
          </Button>
        )}
      </div>
      {allDirectories.length === 0 ? (
        <p className="text-sm text-slate-400">
          You don't have any directories yet — create one first.
        </p>
      ) : (
        <ul className="space-y-1.5">
          {allDirectories.map(dir => (
            <li key={dir.id}>
              <label className="flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
                <input
                  type="checkbox"
                  checked={selected.has(dir.id)}
                  onChange={() => toggle(dir.id)}
                  className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-400"
                />
                <span className="min-w-0 flex-1 truncate">{dir.name}</span>
                {selected.has(dir.id) && (
                  <DirectoryIndexStatusBadge status={indexStatuses?.[dir.id]} />
                )}
              </label>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-3 text-xs leading-relaxed text-slate-400">
        Only files from the directories checked above are visible to this chat — nothing
        from your other directories is ever included.
      </p>
    </div>
  );
}
