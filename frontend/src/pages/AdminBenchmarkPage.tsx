import {useEffect, useMemo, useState} from 'react';
import {useNavigate, useParams} from 'react-router-dom';
import {BenchmarkMetricBars} from '../components/benchmark/BenchmarkMetricBars';
import {Button} from '../components/ui/Button';
import {ConfirmModal} from '../components/ui/ConfirmModal';
import {Modal} from '../components/ui/Modal';
import {benchmarkApi, directoriesApi, llmModelsApi} from '../lib/endpoints';
import {formatUsd} from '../lib/format';
import type {
  BenchmarkModelMetrics,
  BenchmarkQuestion,
  BenchmarkRun,
  BenchmarkRunItem,
  Directory,
  LlmModelOption,
} from '../types';

type Tab = 'questions' | 'runs';

const POLL_INTERVAL_MS = 3000;

export function AdminBenchmarkPage() {
  const {runId: runIdParam} = useParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>(runIdParam ? 'runs' : 'questions');
  const [error, setError] = useState<string | null>(null);

  const [questions, setQuestions] = useState<BenchmarkQuestion[]>([]);
  const [directories, setDirectories] = useState<Directory[]>([]);
  const [models, setModels] = useState<LlmModelOption[]>([]);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [loading, setLoading] = useState(true);

  const [editingQuestion, setEditingQuestion] = useState<BenchmarkQuestion | 'new' | null>(null);
  const [deletingQuestionId, setDeletingQuestionId] = useState<number | null>(null);

  const [selectedRunId, setSelectedRunId] = useState<number | null>(
    runIdParam ? Number(runIdParam) : null,
  );
  const [showNewRunForm, setShowNewRunForm] = useState(false);

  async function loadAll() {
    setLoading(true);
    setError(null);
    try {
      // Independent lists: the model catalog can legitimately be
      // unavailable (503, no OpenRouter sync yet) without that taking down
      // questions/directories/runs, which don't depend on it.
      const [questionsRes, directoriesRes, modelsRes, runsRes] = await Promise.allSettled([
        benchmarkApi.questions.list(),
        directoriesApi.list(),
        llmModelsApi.list(),
        benchmarkApi.runs.list(),
      ]);
      if (questionsRes.status === 'fulfilled') setQuestions(questionsRes.value.questions);
      if (directoriesRes.status === 'fulfilled') setDirectories(directoriesRes.value);
      if (modelsRes.status === 'fulfilled') setModels(modelsRes.value.models);
      if (runsRes.status === 'fulfilled') setRuns(runsRes.value.runs);
      const failures = [questionsRes, directoriesRes, modelsRes, runsRes].filter(
        (result): result is PromiseRejectedResult => result.status === 'rejected',
      );
      if (failures.length > 0) {
        setError(
          failures
            .map(failure => (failure.reason instanceof Error ? failure.reason.message : String(failure.reason)))
            .join(' · '),
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadAll();
  }, []);

  function selectRun(id: number) {
    setSelectedRunId(id);
    setShowNewRunForm(false);
    navigate(`/admin/benchmark/${id}`);
  }

  async function refreshRuns() {
    try {
      const runsRes = await benchmarkApi.runs.list();
      setRuns(runsRes.runs);
    } catch {
      // Best-effort refresh; the detail view surfaces its own errors.
    }
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto bg-slate-50">
      <div className="mx-auto max-w-6xl px-6 py-6">
        <header className="mb-5">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Admin</p>
          <h1 className="mt-1 text-xl font-semibold text-slate-900">Agentic benchmark</h1>
          <p className="mt-1 text-sm text-slate-500">
            Author benchmark questions, pick models to compare, and run them in the background.
          </p>
        </header>

        <div className="mb-5 flex gap-2 border-b border-slate-200">
          {(['questions', 'runs'] as const).map(candidate => (
            <button
              key={candidate}
              type="button"
              onClick={() => setTab(candidate)}
              className={`border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                tab === candidate
                  ? 'border-indigo-600 text-indigo-700'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              {candidate === 'questions' ? 'Questions' : 'Runs'}
            </button>
          ))}
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        {loading ? (
          <p className="text-sm text-slate-400">Loading…</p>
        ) : tab === 'questions' ? (
          <QuestionsPanel
            questions={questions}
            directories={directories}
            onAdd={() => setEditingQuestion('new')}
            onEdit={question => setEditingQuestion(question)}
            onDelete={id => setDeletingQuestionId(id)}
          />
        ) : (
          <RunsPanel
            runs={runs}
            questions={questions}
            models={models}
            selectedRunId={selectedRunId}
            showNewRunForm={showNewRunForm}
            onSelectRun={selectRun}
            onStartNewRun={() => setShowNewRunForm(true)}
            onCancelNewRun={() => setShowNewRunForm(false)}
            onRunCreated={id => {
              setShowNewRunForm(false);
              void refreshRuns();
              selectRun(id);
            }}
            onRunsChanged={refreshRuns}
          />
        )}
      </div>

      {editingQuestion !== null && (
        <QuestionFormModal
          question={editingQuestion === 'new' ? null : editingQuestion}
          directories={directories}
          onClose={() => setEditingQuestion(null)}
          onSaved={question => {
            setQuestions(current => {
              const others = current.filter(existing => existing.id !== question.id);
              return [question, ...others].sort((a, b) => b.id - a.id);
            });
            setEditingQuestion(null);
          }}
        />
      )}

      {deletingQuestionId !== null && (
        <DeleteQuestionModal
          questionId={deletingQuestionId}
          onCancel={() => setDeletingQuestionId(null)}
          onDeleted={id => {
            setQuestions(current => current.filter(question => question.id !== id));
            setDeletingQuestionId(null);
          }}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Questions tab
// ---------------------------------------------------------------------------

function QuestionsPanel({
  questions,
  directories,
  onAdd,
  onEdit,
  onDelete,
}: {
  questions: BenchmarkQuestion[];
  directories: Directory[];
  onAdd: () => void;
  onEdit: (question: BenchmarkQuestion) => void;
  onDelete: (id: number) => void;
}) {
  const directoryName = (id: number) => directories.find(dir => dir.id === id)?.name ?? `#${id}`;

  return (
    <div>
      <div className="mb-3 flex justify-end">
        <Button onClick={onAdd}>Add question</Button>
      </div>
      {questions.length === 0 ? (
        <EmptyState title="No benchmark questions yet" text="Add a question to start building a benchmark." />
      ) : (
        <ul className="space-y-2">
          {questions.map(question => (
            <li
              key={question.id}
              className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-2 text-sm font-medium text-slate-800">
                    {question.prompt}
                    {!question.isActive && (
                      <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                        Inactive
                      </span>
                    )}
                  </p>
                  {question.referenceAnswer && (
                    <p className="mt-1 line-clamp-2 text-xs text-slate-500">{question.referenceAnswer}</p>
                  )}
                  <div className="mt-2 flex flex-wrap gap-1.5 text-xs text-slate-500">
                    {question.directoryIds.map(id => (
                      <span key={id} className="rounded-full bg-indigo-50 px-2 py-0.5 text-indigo-700">
                        {directoryName(id)}
                      </span>
                    ))}
                    {question.tags.map(tag => (
                      <span key={tag} className="rounded-full bg-slate-100 px-2 py-0.5">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex shrink-0 gap-1">
                  <Button variant="ghost" onClick={() => onEdit(question)}>
                    Edit
                  </Button>
                  <Button variant="danger" onClick={() => onDelete(question.id)}>
                    Delete
                  </Button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function QuestionFormModal({
  question,
  directories,
  onClose,
  onSaved,
}: {
  question: BenchmarkQuestion | null;
  directories: Directory[];
  onClose: () => void;
  onSaved: (question: BenchmarkQuestion) => void;
}) {
  const [prompt, setPrompt] = useState(question?.prompt ?? '');
  const [referenceAnswer, setReferenceAnswer] = useState(question?.referenceAnswer ?? '');
  const [expectedFacts, setExpectedFacts] = useState((question?.expectedFacts ?? []).join(', '));
  const [rubricNotes, setRubricNotes] = useState(question?.rubricNotes ?? '');
  const [tags, setTags] = useState((question?.tags ?? []).join(', '));
  const [directoryIds, setDirectoryIds] = useState<number[]>(question?.directoryIds ?? []);
  const [isActive, setIsActive] = useState(question?.isActive ?? true);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  function toggleDirectory(id: number) {
    setDirectoryIds(current =>
      current.includes(id) ? current.filter(existing => existing !== id) : [...current, id],
    );
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!prompt.trim()) {
      setFormError('Prompt is required.');
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      const payload = {
        prompt: prompt.trim(),
        referenceAnswer: referenceAnswer.trim() || null,
        expectedFacts: splitList(expectedFacts),
        rubricNotes: rubricNotes.trim() || null,
        tags: splitList(tags),
        isActive,
        directoryIds,
      };
      const saved = question
        ? await benchmarkApi.questions.update(question.id, payload)
        : await benchmarkApi.questions.create(payload);
      onSaved(saved);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/30 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-white p-6 shadow-xl shadow-slate-900/10"
        onClick={event => event.stopPropagation()}
      >
        <h2 className="mb-4 text-lg font-semibold text-slate-900">
          {question ? 'Edit question' : 'Add question'}
        </h2>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <Field label="Prompt">
            <textarea
              value={prompt}
              onChange={event => setPrompt(event.target.value)}
              rows={3}
              className={INPUT_CLASS}
              placeholder="What is asked of the agent?"
            />
          </Field>
          <Field label="Reference answer (optional)">
            <textarea
              value={referenceAnswer}
              onChange={event => setReferenceAnswer(event.target.value)}
              rows={3}
              className={INPUT_CLASS}
              placeholder="The correct/expected answer, used by the LLM judge."
            />
          </Field>
          <Field label="Expected facts (comma-separated, optional)">
            <input
              value={expectedFacts}
              onChange={event => setExpectedFacts(event.target.value)}
              className={INPUT_CLASS}
              placeholder="241, ceza, mücbir sebep"
            />
          </Field>
          <Field label="Extra judge guidance (optional)">
            <textarea
              value={rubricNotes}
              onChange={event => setRubricNotes(event.target.value)}
              rows={2}
              className={INPUT_CLASS}
              placeholder="Anything the judge should specifically check or ignore for this question."
            />
          </Field>
          <Field label="Tags (comma-separated, optional)">
            <input
              value={tags}
              onChange={event => setTags(event.target.value)}
              className={INPUT_CLASS}
              placeholder="transit, penalty"
            />
          </Field>
          <Field label="Search against these directories">
            {directories.length === 0 ? (
              <p className="text-sm text-slate-400">No directories available.</p>
            ) : (
              <div className="grid max-h-40 grid-cols-2 gap-1.5 overflow-y-auto rounded-lg border border-slate-200 p-2">
                {directories.map(dir => (
                  <label key={dir.id} className="flex items-center gap-2 text-sm text-slate-700">
                    <input
                      type="checkbox"
                      checked={directoryIds.includes(dir.id)}
                      onChange={() => toggleDirectory(dir.id)}
                    />
                    {dir.name}
                  </label>
                ))}
              </div>
            )}
          </Field>
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input type="checkbox" checked={isActive} onChange={event => setIsActive(event.target.checked)} />
            Active (eligible for "all active" runs)
          </label>

          {formError && <p className="text-sm text-rose-600">{formError}</p>}

          <div className="flex justify-end gap-2 pt-2">
            <Button variant="secondary" type="button" onClick={onClose} disabled={saving}>
              Cancel
            </Button>
            <Button type="submit" disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

function DeleteQuestionModal({
  questionId,
  onCancel,
  onDeleted,
}: {
  questionId: number;
  onCancel: () => void;
  onDeleted: (id: number) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleConfirm() {
    setBusy(true);
    setError(null);
    try {
      await benchmarkApi.questions.remove(questionId);
      onDeleted(questionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  if (error) {
    return (
      <Modal title="Cannot delete question" onClose={onCancel}>
        <p className="mb-5 text-sm text-rose-600">{error}</p>
        <div className="flex justify-end">
          <Button variant="secondary" onClick={onCancel}>
            Close
          </Button>
        </div>
      </Modal>
    );
  }

  return (
    <ConfirmModal
      title="Delete question"
      message="This permanently deletes the question. Questions used in a past run cannot be deleted — deactivate them instead."
      onConfirm={handleConfirm}
      onCancel={onCancel}
      busy={busy}
    />
  );
}

// ---------------------------------------------------------------------------
// Runs tab
// ---------------------------------------------------------------------------

function RunsPanel({
  runs,
  questions,
  models,
  selectedRunId,
  showNewRunForm,
  onSelectRun,
  onStartNewRun,
  onCancelNewRun,
  onRunCreated,
  onRunsChanged,
}: {
  runs: BenchmarkRun[];
  questions: BenchmarkQuestion[];
  models: LlmModelOption[];
  selectedRunId: number | null;
  showNewRunForm: boolean;
  onSelectRun: (id: number) => void;
  onStartNewRun: () => void;
  onCancelNewRun: () => void;
  onRunCreated: (id: number) => void;
  onRunsChanged: () => void;
}) {
  return (
    <div className="flex min-h-0 gap-5">
      <aside className="w-72 shrink-0">
        <Button className="mb-3 w-full" onClick={onStartNewRun}>
          New run
        </Button>
        {runs.length === 0 ? (
          <p className="px-1 text-sm text-slate-400">No runs yet.</p>
        ) : (
          <ul className="space-y-1.5">
            {runs.map(run => (
              <li key={run.id}>
                <button
                  type="button"
                  onClick={() => onSelectRun(run.id)}
                  className={`w-full rounded-xl border px-3 py-2.5 text-left transition-colors ${
                    run.id === selectedRunId
                      ? 'border-indigo-200 bg-indigo-50'
                      : 'border-transparent bg-white hover:border-slate-200'
                  }`}
                >
                  <span className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium text-slate-800">
                      {run.label || `Run #${run.id}`}
                    </span>
                    <StatusBadge status={run.status} />
                  </span>
                  <span className="mt-1 block text-xs text-slate-400">
                    {run.completedItems + run.failedItems}/{run.totalItems} items
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <div className="min-w-0 flex-1">
        {showNewRunForm ? (
          <NewRunForm
            questions={questions}
            models={models}
            onCancel={onCancelNewRun}
            onCreated={onRunCreated}
          />
        ) : selectedRunId === null ? (
          <EmptyState title="Pick a run" text="Select a run from the list, or start a new one." />
        ) : (
          <RunDetail runId={selectedRunId} questions={questions} onRunsChanged={onRunsChanged} />
        )}
      </div>
    </div>
  );
}

function NewRunForm({
  questions,
  models,
  onCancel,
  onCreated,
}: {
  questions: BenchmarkQuestion[];
  models: LlmModelOption[];
  onCancel: () => void;
  onCreated: (id: number) => void;
}) {
  const activeQuestions = useMemo(() => questions.filter(question => question.isActive), [questions]);
  const [label, setLabel] = useState('');
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [useAllActive, setUseAllActive] = useState(true);
  const [selectedQuestionIds, setSelectedQuestionIds] = useState<number[]>([]);
  const [judgeModelId, setJudgeModelId] = useState(models[0]?.modelId ?? '');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  function toggleModel(modelId: string) {
    setSelectedModelIds(current =>
      current.includes(modelId) ? current.filter(id => id !== modelId) : [...current, modelId],
    );
  }

  function toggleQuestion(id: number) {
    setSelectedQuestionIds(current =>
      current.includes(id) ? current.filter(existing => existing !== id) : [...current, id],
    );
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (selectedModelIds.length === 0) {
      setFormError('Select at least one model.');
      return;
    }
    if (!useAllActive && selectedQuestionIds.length === 0) {
      setFormError('Select at least one question, or use all active questions.');
      return;
    }
    if (!judgeModelId) {
      setFormError('Select a judge model.');
      return;
    }
    setSubmitting(true);
    setFormError(null);
    try {
      const result = await benchmarkApi.runs.create({
        label: label.trim() || undefined,
        providerModelPairs: selectedModelIds.map(modelId => ({provider: 'openrouter', modelId})),
        questionIds: useAllActive ? 'all-active' : selectedQuestionIds,
        judgeProvider: 'openrouter',
        judgeModel: judgeModelId,
      });
      onCreated(result.runId);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <h2 className="text-base font-semibold text-slate-900">New benchmark run</h2>

      <Field label="Label (optional)">
        <input value={label} onChange={event => setLabel(event.target.value)} className={INPUT_CLASS} />
      </Field>

      <Field label={`Models under test (${selectedModelIds.length} selected)`}>
        {models.length === 0 ? (
          <p className="text-sm text-slate-400">No models available from the catalog.</p>
        ) : (
          <div className="grid max-h-48 grid-cols-1 gap-1 overflow-y-auto rounded-lg border border-slate-200 p-2 sm:grid-cols-2">
            {models.map(model => (
              <label key={model.modelId} className="flex items-start gap-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={selectedModelIds.includes(model.modelId)}
                  onChange={() => toggleModel(model.modelId)}
                />
                <span>
                  <span className="block">{model.displayName}</span>
                  <span className="block text-xs text-slate-400">{model.modelId}</span>
                </span>
              </label>
            ))}
          </div>
        )}
      </Field>

      <Field label="Judge model">
        <select
          value={judgeModelId}
          onChange={event => setJudgeModelId(event.target.value)}
          className={INPUT_CLASS}
        >
          {models.map(model => (
            <option key={model.modelId} value={model.modelId}>
              {model.displayName} ({model.modelId})
            </option>
          ))}
        </select>
      </Field>

      <Field label="Questions">
        <label className="mb-2 flex items-center gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            checked={useAllActive}
            onChange={event => setUseAllActive(event.target.checked)}
          />
          Use all active questions ({activeQuestions.length})
        </label>
        {!useAllActive &&
          (questions.length === 0 ? (
            <p className="text-sm text-slate-400">No questions available.</p>
          ) : (
            <div className="max-h-48 space-y-1 overflow-y-auto rounded-lg border border-slate-200 p-2">
              {questions.map(question => (
                <label key={question.id} className="flex items-start gap-2 text-sm text-slate-700">
                  <input
                    type="checkbox"
                    className="mt-0.5"
                    checked={selectedQuestionIds.includes(question.id)}
                    onChange={() => toggleQuestion(question.id)}
                  />
                  <span className="line-clamp-1">{question.prompt}</span>
                </label>
              ))}
            </div>
          ))}
      </Field>

      {formError && <p className="text-sm text-rose-600">{formError}</p>}

      <div className="flex justify-end gap-2 pt-2">
        <Button variant="secondary" type="button" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
        <Button type="submit" disabled={submitting}>
          {submitting ? 'Starting…' : 'Start run'}
        </Button>
      </div>
    </form>
  );
}

function RunDetail({
  runId,
  questions,
  onRunsChanged,
}: {
  runId: number;
  questions: BenchmarkQuestion[];
  onRunsChanged: () => void;
}) {
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [metrics, setMetrics] = useState<BenchmarkModelMetrics[]>([]);
  const [items, setItems] = useState<BenchmarkRunItem[] | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const detail = await benchmarkApi.runs.get(runId);
        if (cancelled) return;
        setRun(detail.run);
        setMetrics(detail.metrics);
        setError(null);
        if (detail.run.status === 'running') {
          timer = window.setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    setLoading(true);
    setRun(null);
    setItems(null);
    setSelectedModelKey(null);
    void poll();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [runId]);

  async function loadItems() {
    if (items) return;
    try {
      const result = await benchmarkApi.runs.items(runId);
      setItems(result.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleCancel() {
    setCancelling(true);
    try {
      await benchmarkApi.runs.cancel(runId);
      const detail = await benchmarkApi.runs.get(runId);
      setRun(detail.run);
      onRunsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCancelling(false);
    }
  }

  if (loading || !run) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-400">
        Loading run…
      </div>
    );
  }

  const progress = run.totalItems > 0 ? (run.completedItems + run.failedItems) / run.totalItems : 0;
  const selectedItems = selectedModelKey
    ? (items ?? []).filter(item => `${item.provider}/${item.modelId}` === selectedModelKey)
    : [];

  return (
    <div className="space-y-4">
      {error && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">{run.label || `Run #${run.id}`}</h2>
            <p className="mt-1 text-sm text-slate-500">
              Judge: {run.judgeProvider}/{run.judgeModel}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge status={run.status} />
            {(run.status === 'pending' || run.status === 'running') && (
              <Button variant="secondary" onClick={handleCancel} disabled={cancelling}>
                {cancelling ? 'Cancelling…' : 'Cancel'}
              </Button>
            )}
          </div>
        </div>
        <div className="mt-4">
          <div className="h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-2 rounded-full bg-indigo-500 transition-all"
              style={{width: `${Math.round(progress * 100)}%`}}
            />
          </div>
          <p className="mt-1.5 text-xs text-slate-500">
            {run.completedItems} completed · {run.failedItems} failed · {run.totalItems} total
          </p>
        </div>
      </div>

      <BenchmarkMetricBars metrics={metrics} />

      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        <table className="w-full min-w-[1100px] text-left text-sm">
          <thead className="border-b border-slate-100 text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-3 py-2.5">Model</th>
              <th className="px-3 py-2.5">Items</th>
              <th className="px-3 py-2.5">Avg steps</th>
              <th className="px-3 py-2.5">Tokens/step</th>
              <th className="px-3 py-2.5">Avg tokens</th>
              <th className="px-3 py-2.5">Avg duration</th>
              <th className="px-3 py-2.5">Duration/step</th>
              <th className="px-3 py-2.5">p50 / p95</th>
              <th className="px-3 py-2.5">Avg cost</th>
              <th className="px-3 py-2.5">Success</th>
              <th className="px-3 py-2.5">Citations</th>
              <th className="px-3 py-2.5">Judge score</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(model => {
              const key = `${model.provider}/${model.modelId}`;
              return (
                <tr
                  key={key}
                  className={`cursor-pointer border-b border-slate-50 last:border-0 hover:bg-slate-50 ${
                    selectedModelKey === key ? 'bg-indigo-50' : ''
                  }`}
                  onClick={() => {
                    setSelectedModelKey(current => (current === key ? null : key));
                    void loadItems();
                  }}
                >
                  <td className="px-3 py-2.5">
                    <span className="font-medium text-slate-800">{model.modelId}</span>
                    <span className="ml-1.5 text-xs text-slate-400">
                      {model.completedCount}/{model.totalCount}
                      {model.errorCount > 0 ? ` · ${model.errorCount} errored` : ''}
                    </span>
                  </td>
                  <td className="px-3 py-2.5">{model.totalCount}</td>
                  <td className="px-3 py-2.5">{formatNumber(model.avgSteps)}</td>
                  <td className="px-3 py-2.5">{formatNumber(model.avgTokensPerStep)}</td>
                  <td className="px-3 py-2.5">{formatNumber(model.avgTotalTokens)}</td>
                  <td className="px-3 py-2.5">{formatMs(model.avgDurationMs)}</td>
                  <td className="px-3 py-2.5">{formatMs(model.avgDurationPerStepMs)}</td>
                  <td className="px-3 py-2.5">
                    {formatMs(model.p50DurationMs)} / {formatMs(model.p95DurationMs)}
                  </td>
                  <td className="px-3 py-2.5">{formatUsd(model.avgCostUsd) ?? '—'}</td>
                  <td className="px-3 py-2.5">{formatPercent(model.successRate)}</td>
                  <td className="px-3 py-2.5">{formatPercent(model.citationRate)}</td>
                  <td className="px-3 py-2.5">
                    {model.judgeOverallScore === null ? '—' : `${Math.round(model.judgeOverallScore)}/100`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {selectedModelKey && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h3 className="mb-3 text-sm font-semibold text-slate-800">
            {selectedModelKey} — per-question results
          </h3>
          {items === null ? (
            <p className="text-sm text-slate-400">Loading items…</p>
          ) : selectedItems.length === 0 ? (
            <p className="text-sm text-slate-400">No items for this model yet.</p>
          ) : (
            <ul className="space-y-3">
              {selectedItems.map(item => (
                <li key={item.id} className="rounded-lg border border-slate-100 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-sm font-medium text-slate-700">
                      {questions.find(question => question.id === item.questionId)?.prompt ??
                        `Question #${item.questionId}`}
                    </p>
                    <ItemStatusBadge item={item} />
                  </div>
                  {item.status === 'error' ? (
                    <p className="mt-2 text-xs text-rose-600">{item.errorMessage}</p>
                  ) : (
                    <>
                      {item.finalResult && (
                        <p className="mt-2 line-clamp-3 text-xs text-slate-500">{item.finalResult}</p>
                      )}
                      <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
                        {item.citedSources.map(source => (
                          <span key={source} className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-500">
                            {source}
                          </span>
                        ))}
                      </div>
                      {item.judgment && (
                        <p className="mt-2 text-xs text-slate-500">
                          Judge: {item.judgment.overallScore}/100 — {item.judgment.rationale}
                        </p>
                      )}
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function ItemStatusBadge({item}: {item: BenchmarkRunItem}) {
  if (item.status === 'error') {
    return <span className="shrink-0 rounded-full bg-rose-100 px-2 py-0.5 text-xs text-rose-700">Error</span>;
  }
  if (item.status === 'completed' && item.incomplete) {
    return (
      <span className="shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700">
        Incomplete
      </span>
    );
  }
  if (item.status === 'completed') {
    return (
      <span className="shrink-0 rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700">
        Completed
      </span>
    );
  }
  return <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">{item.status}</span>;
}

// ---------------------------------------------------------------------------
// Shared bits
// ---------------------------------------------------------------------------

const INPUT_CLASS =
  'w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none transition focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100';

function Field({label, children}: {label: string; children: React.ReactNode}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-600">{label}</span>
      {children}
    </label>
  );
}

function StatusBadge({status}: {status: BenchmarkRun['status']}) {
  const styles: Record<BenchmarkRun['status'], string> = {
    pending: 'bg-slate-100 text-slate-500',
    running: 'bg-indigo-100 text-indigo-700',
    completed: 'bg-emerald-100 text-emerald-700',
    error: 'bg-rose-100 text-rose-700',
    cancelled: 'bg-amber-100 text-amber-700',
  };
  return (
    <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${styles[status]}`}>{status}</span>
  );
}

function EmptyState({title, text}: {title: string; text: string}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
      <p className="text-sm font-semibold text-slate-700">{title}</p>
      <p className="mt-1 text-sm text-slate-400">{text}</p>
    </div>
  );
}

function splitList(value: string): string[] {
  return value
    .split(',')
    .map(item => item.trim())
    .filter(Boolean);
}

function formatNumber(value: number | null): string {
  if (value === null) return '—';
  return new Intl.NumberFormat(undefined, {maximumFractionDigits: 1}).format(value);
}

function formatMs(value: number | null): string {
  if (value === null) return '—';
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`;
}

function formatPercent(value: number | null): string {
  if (value === null) return '—';
  return `${Math.round(value * 100)}%`;
}
