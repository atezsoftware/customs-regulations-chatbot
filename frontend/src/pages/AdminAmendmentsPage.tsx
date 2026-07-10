import {useCallback, useEffect, useState} from 'react';
import {Button} from '../components/ui/Button';
import {Spinner} from '../components/ui/Spinner';
import {amendmentsApi, directoriesApi} from '../lib/endpoints';
import type {AmendmentProposal, Directory} from '../types';

export function AdminAmendmentsPage() {
  const [directories, setDirectories] = useState<Directory[]>([]);
  const [directoryId, setDirectoryId] = useState<number | null>(null);
  const [rawText, setRawText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [unmatched, setUnmatched] = useState<string[]>([]);
  const [lastBatchId, setLastBatchId] = useState<string | null>(null);

  const [pending, setPending] = useState<AmendmentProposal[]>([]);
  const [loadingPending, setLoadingPending] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [approving, setApproving] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    directoriesApi
      .list()
      .then(items => {
        setDirectories(items);
        setDirectoryId(current => current ?? items[0]?.id ?? null);
      })
      .catch(err => setError(errorMessage(err)));
  }, []);

  const refreshPending = useCallback((dirId: number) => {
    setLoadingPending(true);
    amendmentsApi
      .listProposals({status: 'pending', directoryId: dirId})
      .then(res => {
        setPending(res.proposals);
        setSelectedIds(new Set());
      })
      .catch(err => setError(errorMessage(err)))
      .finally(() => setLoadingPending(false));
  }, []);

  useEffect(() => {
    if (directoryId === null) return;
    const timer = window.setTimeout(() => refreshPending(directoryId), 0);
    return () => window.clearTimeout(timer);
  }, [directoryId, refreshPending]);

  async function handleAnalyze() {
    if (directoryId === null) return;
    const trimmed = rawText.trim();
    if (!trimmed) return;
    setAnalyzing(true);
    setError(null);
    setActionError(null);
    try {
      const result = await amendmentsApi.analyze(directoryId, trimmed);
      setUnmatched(result.unmatchedInstructions);
      setLastBatchId(result.batchId);
      refreshPending(directoryId);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setAnalyzing(false);
    }
  }

  function toggleSelected(id: string) {
    setSelectedIds(current => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleSelectAll() {
    setSelectedIds(current =>
      current.size === pending.length ? new Set() : new Set(pending.map(p => p.id)),
    );
  }

  async function handleApproveSelected() {
    if (!selectedIds.size || directoryId === null) return;
    setApproving(true);
    setActionError(null);
    try {
      const result = await amendmentsApi.approve(Array.from(selectedIds));
      if (result.failed.length) {
        setActionError(
          result.failed.map(f => `${f.proposalId.slice(0, 16)}...: ${f.reason}`).join(' | '),
        );
      }
      refreshPending(directoryId);
    } catch (err) {
      setActionError(errorMessage(err));
    } finally {
      setApproving(false);
    }
  }

  async function handleReject(id: string) {
    if (directoryId === null) return;
    setActionError(null);
    try {
      await amendmentsApi.reject(id);
      refreshPending(directoryId);
    } catch (err) {
      setActionError(errorMessage(err));
    }
  }

  async function handleDelete(id: string) {
    setActionError(null);
    try {
      await amendmentsApi.remove(id);
      setPending(current => current.filter(p => p.id !== id));
      setSelectedIds(current => {
        const next = new Set(current);
        next.delete(id);
        return next;
      });
    } catch (err) {
      setActionError(errorMessage(err));
    }
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto bg-slate-50">
      <div className="mx-auto max-w-5xl px-6 py-6">
        <div className="mb-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Mevzuat güncelleme
          </p>
          <h1 className="mt-1 text-lg font-semibold text-slate-900">
            Resmi Gazete değişiklik metni analizi
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Değişiklik metnini yapıştırın; sistem etkilenen chunk&apos;ları bulup taslak
            değişiklikler önerecek. Hiçbir şey, aşağıdan onaylamadan veritabanına yazılmaz.
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        <section className="mb-6 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <label className="block text-sm" htmlFor="directory-select">
            <span className="mb-1.5 block font-medium text-slate-700">Dizin</span>
            <select
              id="directory-select"
              value={directoryId ?? ''}
              onChange={event => setDirectoryId(Number(event.target.value))}
              className="w-full max-w-sm rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-slate-900 outline-none transition-shadow focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
            >
              {directories.length === 0 && <option value="">Dizin yok</option>}
              {directories.map(directory => (
                <option key={directory.id} value={directory.id}>
                  {directory.name}
                </option>
              ))}
            </select>
          </label>

          <label className="mt-4 block text-sm" htmlFor="raw-text">
            <span className="mb-1.5 block font-medium text-slate-700">
              Resmi Gazete değişiklik metni
            </span>
            <textarea
              id="raw-text"
              value={rawText}
              onChange={event => setRawText(event.target.value)}
              rows={8}
              placeholder="MADDE 1- ... değiştirilmiştir. ..."
              className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-900 outline-none transition-shadow placeholder:text-slate-400 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
            />
          </label>

          <div className="mt-4 flex items-center gap-3">
            <Button
              onClick={() => void handleAnalyze()}
              disabled={analyzing || directoryId === null || !rawText.trim()}
            >
              {analyzing && <Spinner className="h-4 w-4" />}
              Analiz Et
            </Button>
            {lastBatchId && !analyzing && (
              <span className="text-xs text-slate-400">
                Son analiz: {lastBatchId.slice(0, 20)}...
              </span>
            )}
          </div>

          {unmatched.length > 0 && (
            <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <p className="font-medium">
                {unmatched.length} talimat eşleştirilemedi (corpus&apos;ta ilgili içerik
                bulunamadı):
              </p>
              <ul className="mt-1 list-disc space-y-1 pl-5">
                {unmatched.map((text, index) => (
                  <li key={index} className="line-clamp-2">
                    {text}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <section>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-sm font-semibold text-slate-700">
              Bekleyen öneriler {loadingPending ? '' : `(${pending.length})`}
            </h2>
            {pending.length > 0 && (
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-600">
                  <input
                    type="checkbox"
                    checked={selectedIds.size > 0 && selectedIds.size === pending.length}
                    onChange={toggleSelectAll}
                  />
                  Tümünü seç
                </label>
                <Button
                  onClick={() => void handleApproveSelected()}
                  disabled={approving || selectedIds.size === 0}
                >
                  {approving && <Spinner className="h-4 w-4" />}
                  Seçilenleri Onayla ({selectedIds.size})
                </Button>
              </div>
            )}
          </div>

          {actionError && (
            <div className="mb-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {actionError}
            </div>
          )}

          {loadingPending ? (
            <div className="rounded-xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-400">
              Yükleniyor...
            </div>
          ) : pending.length === 0 ? (
            <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
              <p className="text-sm font-medium text-slate-700">Bekleyen öneri yok.</p>
              <p className="mt-1 text-sm text-slate-400">
                Yukarıdan bir değişiklik metni analiz edin.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {pending.map(proposal => (
                <ProposalCard
                  key={proposal.id}
                  proposal={proposal}
                  selected={selectedIds.has(proposal.id)}
                  open={openId === proposal.id}
                  onToggleSelected={() => toggleSelected(proposal.id)}
                  onToggleOpen={() =>
                    setOpenId(current => (current === proposal.id ? null : proposal.id))
                  }
                  onReject={() => void handleReject(proposal.id)}
                  onDelete={() => void handleDelete(proposal.id)}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function ProposalCard({
  proposal,
  selected,
  open,
  onToggleSelected,
  onToggleOpen,
  onReject,
  onDelete,
}: {
  proposal: AmendmentProposal;
  selected: boolean;
  open: boolean;
  onToggleSelected: () => void;
  onToggleOpen: () => void;
  onReject: () => void;
  onDelete: () => void;
}) {
  return (
    <article className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-start gap-3 px-4 py-3">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggleSelected}
          className="mt-1"
        />
        <button type="button" onClick={onToggleOpen} className="min-w-0 flex-1 text-left">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-slate-800">
              {proposal.oldChunkId ? 'Madde değişikliği' : 'Yeni madde'}
            </span>
            {proposal.matchConfidence !== null && (
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                güven: {Math.round(proposal.matchConfidence * 100)}%
              </span>
            )}
            {proposal.duplicateTarget && (
              <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
                aynı chunk&apos;ı hedefleyen başka öneri var
              </span>
            )}
          </div>
          <p className="mt-1 line-clamp-2 text-sm text-slate-600">{proposal.instructionText}</p>
          {proposal.matchRationale && (
            <p className="mt-1 text-xs text-slate-400">{proposal.matchRationale}</p>
          )}
          {proposal.dateRationale && (
            <p className="mt-0.5 text-xs text-slate-400">{proposal.dateRationale}</p>
          )}
        </button>
        <div className="flex shrink-0 gap-2">
          <Button variant="secondary" onClick={onReject}>
            Reddet
          </Button>
          <Button variant="danger" onClick={onDelete}>
            Sil
          </Button>
        </div>
      </div>

      {open && (
        <div className="grid grid-cols-1 gap-3 border-t border-slate-100 bg-slate-950 p-4 md:grid-cols-2">
          <div>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Eski chunk
            </p>
            <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-100">
              {proposal.oldChunkId
                ? JSON.stringify(proposal.oldChunkSnapshot, null, 2)
                : '(yok — yeni madde)'}
            </pre>
          </div>
          <div>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Yeni chunk (taslak)
            </p>
            <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap text-xs leading-5 text-emerald-200">
              {JSON.stringify(proposal.newChunkDraft, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </article>
  );
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
