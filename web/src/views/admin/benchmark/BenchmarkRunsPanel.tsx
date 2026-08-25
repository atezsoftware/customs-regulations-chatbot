"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  Button,
  Card,
  Checkbox,
  InputSelect,
  InputTypeIn,
  ProgressBar,
  SelectCard,
  Tag,
  Text,
} from "@opal/components";
import { toast } from "@opal/layouts";

import {
  type BenchmarkAvailableModel,
  type BenchmarkQuestion,
  type BenchmarkRun,
  type BenchmarkSearchMode,
  cancelBenchmarkRun,
  createBenchmarkRun,
  listBenchmarkModels,
  listBenchmarkQuestions,
  listBenchmarkRuns,
  startBenchmarkRun,
} from "@/lib/regulatory/benchmark";
import {
  FormField,
  formatCost,
  formatDuration,
  formatPercent,
  Metric,
  modelIdentityLabel,
  modelKey,
  StatusBadge,
} from "@/views/admin/benchmark/BenchmarkPresentation";
import BenchmarkRunItemDetails from "@/views/admin/benchmark/BenchmarkRunItemDetails";

const panelClass = "rounded-xl border border-border-02 bg-background p-5";
const Field = FormField;

const withUpdatedRun = (runs: BenchmarkRun[], updatedRun: BenchmarkRun) => [
  updatedRun,
  ...runs.filter((run) => run.id !== updatedRun.id),
];

export function processedItemCount(run: BenchmarkRun): number {
  const cancelledItems = run.items.filter(
    (item) => item.status === "cancelled"
  ).length;
  return run.completed_items + run.failed_items + cancelledItems;
}

function mergeRunSnapshots(
  currentRuns: BenchmarkRun[],
  incomingRuns: BenchmarkRun[],
  protectedRunIds: ReadonlySet<number>
): BenchmarkRun[] {
  const currentById = new Map(currentRuns.map((run) => [run.id, run]));
  const mergedRuns = incomingRuns.map((incomingRun) => {
    const currentRun = currentById.get(incomingRun.id);
    if (currentRun && protectedRunIds.has(incomingRun.id)) return currentRun;
    return incomingRun;
  });
  const incomingIds = new Set(incomingRuns.map((run) => run.id));
  return [
    ...currentRuns.filter(
      (run) => protectedRunIds.has(run.id) && !incomingIds.has(run.id)
    ),
    ...mergedRuns,
  ];
}

interface ProtectedRunUpdate {
  expected: BenchmarkRun["status"];
  previous: BenchmarkRun["status"] | null;
  attemptQueuedAt: string | null;
}

function isSameOrNewerAttempt(
  incomingQueuedAt: string | null,
  expectedQueuedAt: string | null
) {
  if (!incomingQueuedAt || !expectedQueuedAt) return false;
  const incomingTime = Date.parse(incomingQueuedAt);
  const expectedTime = Date.parse(expectedQueuedAt);
  return (
    Number.isFinite(incomingTime) &&
    Number.isFinite(expectedTime) &&
    incomingTime >= expectedTime
  );
}

function hasObservedRunUpdate(
  update: ProtectedRunUpdate,
  incomingRun: BenchmarkRun
) {
  const incomingStatus = incomingRun.status;
  if (incomingStatus === update.expected) return true;
  if (
    update.previous === "error" &&
    update.expected === "queued" &&
    incomingStatus === "error"
  )
    return isSameOrNewerAttempt(incomingRun.queued_at, update.attemptQueuedAt);
  if (update.expected === "queued" && incomingStatus === "running")
    return isSameOrNewerAttempt(incomingRun.queued_at, update.attemptQueuedAt);
  if (update.expected === "pending") return true;
  return (
    (update.expected === "queued" || update.expected === "running") &&
    (incomingStatus === "completed" ||
      incomingStatus === "error" ||
      incomingStatus === "cancelled")
  );
}

export default function BenchmarkRunsPanel() {
  const [questions, setQuestions] = useState<BenchmarkQuestion[]>([]);
  const [models, setModels] = useState<BenchmarkAvailableModel[]>([]);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [selectedQuestionIds, setSelectedQuestionIds] = useState<number[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [judgeKey, setJudgeKey] = useState<string>("");
  const [modelSearch, setModelSearch] = useState("");
  const [judgeSearch, setJudgeSearch] = useState("");
  const [questionSearch, setQuestionSearch] = useState("");
  const [label, setLabel] = useState("");
  const [deepResearch, setDeepResearch] = useState(false);
  const [searchMode, setSearchMode] = useState<BenchmarkSearchMode>("v2");
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [launching, setLaunching] = useState(false);
  const [startingRunIds, setStartingRunIds] = useState<Set<number>>(new Set());
  const runRequestSequence = useRef(0);
  const protectedRunUpdates = useRef(new Map<number, ProtectedRunUpdate>());

  const setRunStarting = useCallback((runId: number, starting: boolean) => {
    setStartingRunIds((current) => {
      const next = new Set(current);
      if (starting) next.add(runId);
      else next.delete(runId);
      return next;
    });
  }, []);

  const refreshConfiguration = useCallback(async () => {
    void listBenchmarkQuestions()
      .then((nextQuestions) =>
        setQuestions(nextQuestions.filter((question) => question.is_active))
      )
      .catch((error: unknown) =>
        toast.error(
          error instanceof Error ? error.message : "Questions failed."
        )
      );
    void listBenchmarkModels()
      .then(setModels)
      .catch((error: unknown) =>
        toast.error(error instanceof Error ? error.message : "Models failed.")
      );
  }, []);

  const refreshRuns = useCallback(async () => {
    const requestSequence = ++runRequestSequence.current;
    try {
      const nextRuns = await listBenchmarkRuns();
      if (requestSequence !== runRequestSequence.current) return;
      for (const nextRun of nextRuns) {
        const protectedUpdate = protectedRunUpdates.current.get(nextRun.id);
        if (protectedUpdate && hasObservedRunUpdate(protectedUpdate, nextRun)) {
          protectedRunUpdates.current.delete(nextRun.id);
        }
      }
      setRuns((current) =>
        mergeRunSnapshots(
          current,
          nextRuns,
          new Set(protectedRunUpdates.current.keys())
        )
      );
      setSelectedRunId((current) => current ?? nextRuns[0]?.id ?? null);
    } catch (error) {
      if (requestSequence !== runRequestSequence.current) return;
      toast.error(error instanceof Error ? error.message : "Runs failed.");
    }
  }, []);

  useEffect(() => {
    void refreshConfiguration();
    void refreshRuns();
  }, [refreshConfiguration, refreshRuns]);
  useEffect(() => {
    if (
      !runs.some((run) => run.status === "queued" || run.status === "running")
    )
      return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      await refreshRuns();
      if (!cancelled) timer = window.setTimeout(() => void poll(), 3000);
    };
    timer = window.setTimeout(() => void poll(), 3000);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [refreshRuns, runs]);

  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const modelMap = useMemo(
    () => new Map(models.map((model) => [modelKey(model), model])),
    [models]
  );
  const filteredModels = useMemo(() => {
    const query = modelSearch.trim().toLowerCase();
    return models
      .filter(
        (model) =>
          !query ||
          `${model.display_name} ${model.model_id}`
            .toLowerCase()
            .includes(query)
      )
      .slice(0, 80);
  }, [modelSearch, models]);
  const filteredJudgeModels = useMemo(() => {
    const query = judgeSearch.trim().toLowerCase();
    return models.filter(
      (model) =>
        !query ||
        `${model.display_name} ${model.model_id}`.toLowerCase().includes(query)
    );
  }, [judgeSearch, models]);
  const filteredQuestions = useMemo(() => {
    const query = questionSearch.trim().toLocaleLowerCase("tr");
    return questions.filter(
      (question) =>
        !query ||
        `${question.title} ${question.prompt} ${question.tags.join(" ")}`
          .toLocaleLowerCase("tr")
          .includes(query)
    );
  }, [questionSearch, questions]);

  const launch = useCallback(async () => {
    const judge = modelMap.get(judgeKey);
    const candidates = selectedModels
      .map((key) => modelMap.get(key))
      .filter((model): model is BenchmarkAvailableModel => Boolean(model));
    if (!judge || candidates.length === 0 || selectedQuestionIds.length === 0)
      return;
    setLaunching(true);
    try {
      const run = await createBenchmarkRun({
        label: label.trim() || null,
        question_ids: selectedQuestionIds,
        candidates: candidates.map(({ provider, provider_id, model_id }) => ({
          provider,
          provider_id,
          model_id,
        })),
        judge: {
          provider: judge.provider,
          provider_id: judge.provider_id,
          model_id: judge.model_id,
        },
        deep_research: deepResearch,
        search_mode: searchMode,
      });
      runRequestSequence.current += 1;
      protectedRunUpdates.current.set(run.id, {
        expected: run.status,
        previous: null,
        attemptQueuedAt: run.queued_at,
      });
      setRuns((current) => withUpdatedRun(current, run));
      setSelectedRunId(run.id);
      setRunStarting(run.id, true);
      try {
        const startedRun = await startBenchmarkRun(run.id);
        runRequestSequence.current += 1;
        protectedRunUpdates.current.set(run.id, {
          expected: startedRun.status,
          previous: run.status,
          attemptQueuedAt: startedRun.queued_at,
        });
        setRuns((current) => withUpdatedRun(current, startedRun));
        toast.success("Benchmark run queued through the production chat flow.");
      } catch (error) {
        toast.error(
          error instanceof Error
            ? `Run created, but could not start: ${error.message}`
            : "Run created, but could not start."
        );
      } finally {
        setRunStarting(run.id, false);
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Run creation failed."
      );
    } finally {
      setLaunching(false);
    }
  }, [
    deepResearch,
    judgeKey,
    label,
    modelMap,
    selectedModels,
    selectedQuestionIds,
    searchMode,
    setRunStarting,
  ]);

  const startRun = useCallback(
    async (run: BenchmarkRun) => {
      setSelectedRunId(run.id);
      setRunStarting(run.id, true);
      try {
        const startedRun = await startBenchmarkRun(run.id);
        runRequestSequence.current += 1;
        protectedRunUpdates.current.set(run.id, {
          expected: startedRun.status,
          previous: run.status,
          attemptQueuedAt: startedRun.queued_at,
        });
        setRuns((current) => withUpdatedRun(current, startedRun));
        toast.success(
          run.status === "error"
            ? "Benchmark run queued for retry."
            : "Benchmark run queued."
        );
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Run could not start."
        );
      } finally {
        setRunStarting(run.id, false);
      }
    },
    [setRunStarting]
  );

  const cancelRun = useCallback(async (run: BenchmarkRun) => {
    try {
      const cancelledRun = await cancelBenchmarkRun(run.id);
      runRequestSequence.current += 1;
      protectedRunUpdates.current.set(run.id, {
        expected: cancelledRun.status,
        previous: run.status,
        attemptQueuedAt: cancelledRun.queued_at,
      });
      setRuns((current) => withUpdatedRun(current, cancelledRun));
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Run could not cancel."
      );
    }
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <Card border="solid" padding="lg">
        <div className="flex flex-col gap-1 pb-5">
          <Text as="h2" font="heading-h3" color="text-05">
            Create a run
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            Choose the questions, candidate models, and independent judge used
            for a production-chat benchmark.
          </Text>
        </div>
        <div className="grid items-start gap-6 xl:grid-cols-3">
          <div className="flex min-w-0 flex-col gap-3">
            <Field label="Run label">
              <InputTypeIn
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                placeholder="July legal QA baseline"
              />
            </Field>
            <Field label={`Questions · ${selectedQuestionIds.length} selected`}>
              <InputTypeIn
                searchIcon
                value={questionSearch}
                onChange={(event) => setQuestionSearch(event.target.value)}
                placeholder="Search questions"
              />
            </Field>
            <div className="flex gap-2">
              <Button
                size="sm"
                prominence="tertiary"
                onClick={() =>
                  setSelectedQuestionIds(
                    filteredQuestions.map((question) => question.id)
                  )
                }
              >
                Select visible
              </Button>
              <Button
                size="sm"
                prominence="tertiary"
                onClick={() => setSelectedQuestionIds([])}
              >
                Clear
              </Button>
            </div>
            <div className="h-72 overflow-y-auto rounded-lg border border-border-02">
              {filteredQuestions.length === 0 ? (
                <div className="flex h-full items-center justify-center p-4 text-center">
                  <Text font="main-ui-body" color="text-02">
                    No matching questions.
                  </Text>
                </div>
              ) : (
                filteredQuestions.map((question) => (
                  <label
                    key={question.id}
                    className="flex cursor-pointer gap-2 border-b border-border-01 p-3 last:border-0 hover:bg-background-01"
                  >
                    <Checkbox
                      checked={selectedQuestionIds.includes(question.id)}
                      onCheckedChange={(checked) =>
                        setSelectedQuestionIds((current) =>
                          checked
                            ? current.includes(question.id)
                              ? current
                              : [...current, question.id]
                            : current.filter((id) => id !== question.id)
                        )
                      }
                      aria-label={`${question.title}: ${question.prompt}`}
                    />
                    <span className="flex min-w-0 flex-col gap-1">
                      <Text font="main-ui-action" color="text-04" nowrap>
                        {question.title}
                      </Text>
                      <Text font="secondary-body" color="text-02" nowrap>
                        {question.prompt}
                      </Text>
                    </span>
                  </label>
                ))
              )}
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-3">
            <Field
              label={`Candidate models · ${selectedModels.length} selected`}
              hint={`${models.length} OpenRouter models available`}
            >
              <InputTypeIn
                searchIcon
                value={modelSearch}
                onChange={(event) => setModelSearch(event.target.value)}
                placeholder="Search OpenRouter catalog"
              />
            </Field>
            <div className="flex h-72 min-h-0 flex-col overflow-hidden rounded-lg border border-border-02">
              {selectedModels.length > 0 && (
                <div className="max-h-24 shrink-0 overflow-y-auto border-b border-border-02 bg-background-01 p-2">
                  <div className="flex flex-wrap gap-1.5">
                    {selectedModels.map((key) => {
                      const model = modelMap.get(key);
                      const displayName = model?.display_name ?? key;
                      return (
                        <Tag
                          key={key}
                          title={displayName}
                          truncate
                          onRemove={() =>
                            setSelectedModels((current) =>
                              current.filter((item) => item !== key)
                            )
                          }
                        />
                      );
                    })}
                  </div>
                </div>
              )}
              <div className="min-h-0 flex-1 overflow-y-auto">
                {filteredModels.length === 0 ? (
                  <div className="flex h-full items-center justify-center p-4 text-center">
                    <Text font="main-ui-body" color="text-02">
                      No matching OpenRouter models.
                    </Text>
                  </div>
                ) : (
                  filteredModels.map((model) => {
                    const key = modelKey(model);
                    return (
                      <label
                        key={key}
                        className="flex cursor-pointer items-start gap-2 border-b border-border-01 p-3 last:border-0 hover:bg-background-01"
                      >
                        <Checkbox
                          checked={selectedModels.includes(key)}
                          onCheckedChange={(checked) =>
                            setSelectedModels((current) =>
                              checked
                                ? current.includes(key)
                                  ? current
                                  : [...current, key]
                                : current.filter((item) => item !== key)
                            )
                          }
                          aria-label={`${model.display_name}: ${model.model_id}`}
                        />
                        <span className="flex min-w-0 flex-col gap-1">
                          <Text font="main-ui-action" color="text-04" nowrap>
                            {model.display_name}
                          </Text>
                          <Text font="secondary-body" color="text-02" nowrap>
                            {`${model.model_id}${
                              model.max_input_tokens
                                ? ` · ${model.max_input_tokens.toLocaleString()} context`
                                : ""
                            }`}
                          </Text>
                        </span>
                      </label>
                    );
                  })
                )}
              </div>
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-4">
            <Field
              label="Judge model"
              hint="Use a strong model; it scores every answer and writes the final report."
            >
              <InputTypeIn
                searchIcon
                value={judgeSearch}
                onChange={(event) => setJudgeSearch(event.target.value)}
                placeholder="Filter judge models"
              />
              <InputSelect value={judgeKey} onValueChange={setJudgeKey}>
                <InputSelect.Trigger
                  aria-label="Judge model"
                  placeholder="Select OpenRouter judge"
                />
                <InputSelect.Content>
                  {filteredJudgeModels.map((model) => (
                    <InputSelect.Item
                      key={modelKey(model)}
                      value={modelKey(model)}
                      description={model.model_id}
                    >
                      {model.display_name}
                    </InputSelect.Item>
                  ))}
                </InputSelect.Content>
              </InputSelect>
            </Field>
            <Field
              label="Search mode"
              hint="Choose the production regulatory search workflow for every run item."
            >
              <InputSelect
                value={searchMode}
                onValueChange={(value) =>
                  setSearchMode(value as BenchmarkSearchMode)
                }
              >
                <InputSelect.Trigger aria-label="Search mode" />
                <InputSelect.Content>
                  <InputSelect.Item
                    value="v2"
                    description="Current regulatory search workflow"
                  >
                    Atez Search V2
                  </InputSelect.Item>
                  <InputSelect.Item
                    value="v1"
                    description="Legacy regulatory search workflow"
                  >
                    Atez Search V1
                  </InputSelect.Item>
                </InputSelect.Content>
              </InputSelect>
            </Field>
            <Card border="solid" padding="sm">
              <label className="flex cursor-pointer items-start gap-3">
                <Checkbox
                  id="benchmark-deep-research"
                  aria-label="Deep research"
                  checked={deepResearch}
                  onCheckedChange={setDeepResearch}
                />
                <span className="flex flex-col gap-1">
                  <Text font="main-ui-action" color="text-04">
                    Deep research
                  </Text>
                  <Text font="secondary-body" color="text-02">
                    Uses the production deep-research loop; no web-search tool
                    is enabled.
                  </Text>
                </span>
              </label>
            </Card>
            <Card background="heavy" padding="sm">
              <div className="flex flex-col gap-1">
                <Text font="main-ui-body" color="text-03">
                  {`${selectedQuestionIds.length} questions × ${selectedModels.length} models`}
                </Text>
                <Text font="main-ui-action" color="text-05">
                  {`${selectedQuestionIds.length * selectedModels.length} total run items`}
                </Text>
              </div>
            </Card>
            <Button
              disabled={
                launching ||
                selectedQuestionIds.length === 0 ||
                selectedModels.length === 0 ||
                !judgeKey
              }
              onClick={() => void launch()}
            >
              {launching ? "Creating…" : "Create and start"}
            </Button>
          </div>
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[300px_1fr]">
        <aside className="flex flex-col gap-2">
          <Text as="h2" font="main-ui-action" color="text-04">
            Run history
          </Text>
          {runs.map((run) => (
            <SelectCard
              key={run.id}
              state={run.id === selectedRunId ? "selected" : "empty"}
              padding="sm"
              role="button"
              aria-pressed={run.id === selectedRunId}
              tabIndex={0}
              onClick={() => setSelectedRunId(run.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setSelectedRunId(run.id);
                }
              }}
            >
              <div className="flex items-center justify-between gap-2">
                <Text font="main-ui-action" color="inherit" nowrap>
                  {run.label ?? `Run #${run.id}`}
                </Text>
                <StatusBadge status={run.status} />
              </div>
              <Text as="p" font="secondary-body" color="text-02">
                {`${processedItemCount(run)}/${run.total_items} items · ${run.judge_model}`}
              </Text>
            </SelectCard>
          ))}
        </aside>
        <main className="min-w-0">
          {!selectedRun && (
            <Card border="solid">
              <Text as="p" font="main-ui-body" color="text-03">
                Create a run or select one from the history to inspect it.
              </Text>
            </Card>
          )}
          {selectedRun && (
            <div className="flex flex-col gap-4">
              <Card border="solid" padding="lg">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="flex flex-col gap-1">
                    <Text as="h2" font="heading-h3" color="text-05">
                      {selectedRun.label ?? `Run #${selectedRun.id}`}
                    </Text>
                    <Text as="p" font="secondary-body" color="text-02">
                      {`Judge: ${selectedRun.judge_provider} / ${selectedRun.judge_model} · ${
                        selectedRun.deep_research
                          ? "Deep research"
                          : "Internal search"
                      } · Atez Search ${selectedRun.search_mode.toUpperCase()}`}
                    </Text>
                  </div>
                  <div className="flex items-center gap-2">
                    <StatusBadge status={selectedRun.status} />
                    {(selectedRun.status === "pending" ||
                      selectedRun.status === "error") && (
                      <Button
                        size="sm"
                        disabled={startingRunIds.has(selectedRun.id)}
                        onClick={() => void startRun(selectedRun)}
                      >
                        {startingRunIds.has(selectedRun.id)
                          ? "Starting…"
                          : selectedRun.status === "error"
                            ? "Retry"
                            : "Start"}
                      </Button>
                    )}
                    {(selectedRun.status === "queued" ||
                      selectedRun.status === "running") && (
                      <Button
                        size="sm"
                        variant="danger"
                        prominence="secondary"
                        onClick={() => void cancelRun(selectedRun)}
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                </div>
                <div className="flex flex-col gap-2 pt-4">
                  <div className="flex items-center justify-between gap-3">
                    <Text font="main-ui-action" color="text-04">
                      Overall progress
                    </Text>
                    <Text font="secondary-body" color="text-03">
                      {`${processedItemCount(selectedRun)} of ${selectedRun.total_items} items`}
                    </Text>
                  </div>
                  <ProgressBar
                    aria-label="Run progress"
                    value={processedItemCount(selectedRun)}
                    max={selectedRun.total_items}
                    color={selectedRun.failed_items > 0 ? "red" : "blue"}
                  />
                </div>
                <div className="grid gap-2 pt-4 sm:grid-cols-2 lg:grid-cols-4">
                  <Metric
                    label="Progress"
                    value={`${processedItemCount(selectedRun)}/${selectedRun.total_items}`}
                  />
                  <Metric
                    label="Completed"
                    value={String(selectedRun.completed_items)}
                  />
                  <Metric
                    label="Failed"
                    value={String(selectedRun.failed_items)}
                  />
                  <Metric
                    label="Report cost"
                    value={formatCost(selectedRun.report_cost_cents)}
                  />
                </div>
              </Card>
              {selectedRun.failure_message && (
                <div className="rounded-lg border border-status-error-03 bg-status-error-01 p-3">
                  <div className="flex flex-col gap-1">
                    <Text font="main-ui-action" color="status-error-05">
                      Benchmark execution failed
                    </Text>
                    <Text as="p" font="main-ui-body" color="status-error-05">
                      {selectedRun.failure_message}
                    </Text>
                    {selectedRun.failure_code && (
                      <Text font="secondary-body" color="status-error-05">
                        {`Failure code: ${selectedRun.failure_code}`}
                      </Text>
                    )}
                  </div>
                </div>
              )}
              {selectedRun.report && (
                <details open className={panelClass}>
                  <summary className="cursor-pointer">
                    <Text font="heading-h3" color="text-05">
                      Judge comparison report
                    </Text>
                  </summary>
                  <div className="mt-4">
                    <Text as="p" font="main-ui-body" color="text-03">
                      {selectedRun.report.executive_summary}
                    </Text>
                    <div className="mt-4 grid gap-3 md:grid-cols-2">
                      {[...selectedRun.report.model_reports]
                        .sort((a, b) => a.rank - b.rank)
                        .map((model) => (
                          <div
                            key={modelKey(model)}
                            className="rounded-lg border border-border-01 p-3"
                          >
                            <Text font="main-ui-action" color="text-05">
                              {`#${model.rank} · ${modelIdentityLabel(model)}`}
                            </Text>
                            <div className="mt-1">
                              <Text as="p" font="main-ui-body" color="text-03">
                                {model.summary}
                              </Text>
                            </div>
                            <div className="mt-2">
                              <Text font="secondary-action" color="text-04">
                                Recommended use
                              </Text>
                            </div>
                            <Text as="p" font="main-ui-body" color="text-03">
                              {model.recommended_use}
                            </Text>
                          </div>
                        ))}
                    </div>
                    <div className="mt-4 rounded-lg bg-background-01 p-3">
                      <Text font="main-ui-action" color="text-04">
                        Recommendation
                      </Text>
                      <div className="mt-1">
                        <Text as="p" font="main-ui-body" color="text-03">
                          {selectedRun.report.recommendation}
                        </Text>
                      </div>
                    </div>
                  </div>
                </details>
              )}
              {selectedRun.report_error && (
                <div className="rounded-lg border border-status-error-03 bg-status-error-01 p-3">
                  <Text font="main-ui-body" color="status-error-05">
                    {`Run report failed: ${selectedRun.report_error}`}
                  </Text>
                </div>
              )}
              <section className="grid gap-3 md:grid-cols-2">
                {selectedRun.aggregates.map((aggregate) => (
                  <div key={modelKey(aggregate)} className={panelClass}>
                    <Text font="main-ui-action" color="text-05">
                      {modelIdentityLabel(aggregate)}
                    </Text>
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      <Text color="text-02">Score</Text>
                      <Text color="text-04">
                        {aggregate.average_score?.toFixed(1) ?? "—"}
                      </Text>
                      <Text color="text-02">Citation recall</Text>
                      <Text color="text-04">
                        {formatPercent(aggregate.average_citation_recall)}
                      </Text>
                      <Text color="text-02">Avg tokens</Text>
                      <Text color="text-04">
                        {aggregate.average_tokens?.toFixed(0) ?? "—"}
                      </Text>
                      <Text color="text-02">Avg latency</Text>
                      <Text color="text-04">
                        {formatDuration(aggregate.average_duration_ms)}
                      </Text>
                      <Text color="text-02">Answer + judge cost</Text>
                      <Text color="text-04">
                        {formatCost(aggregate.total_cost_cents)}
                      </Text>
                    </div>
                  </div>
                ))}
              </section>
              <section className="flex flex-col gap-3">
                <Text as="h3" font="heading-h3" color="text-05">
                  Item results
                </Text>
                {selectedRun.items.map((item) => (
                  <BenchmarkRunItemDetails key={item.id} item={item} />
                ))}
              </section>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
