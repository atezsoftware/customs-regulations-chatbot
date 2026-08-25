import {
  act,
  deferred,
  render,
  screen,
  setupUser,
  waitFor,
} from "@tests/setup/test-utils";

import {
  BenchmarkAvailableModel,
  BenchmarkQuestion,
  BenchmarkRun,
  BenchmarkRunItem,
  createBenchmarkQuestion,
  createBenchmarkRun,
  getBenchmarkLimits,
  getBenchmarkRun,
  listBenchmarkCitationOptions,
  listBenchmarkModels,
  listBenchmarkQuestions,
  listBenchmarkRunItems,
  listBenchmarkRuns,
  startBenchmarkRun,
} from "@/lib/regulatory/benchmark";
import BenchmarkPage from "@/views/admin/BenchmarkPage";
import { formatCost } from "@/views/admin/benchmark/BenchmarkPresentation";
import { processedItemCount } from "@/views/admin/benchmark/BenchmarkRunsPanel";

test("tiny non-zero benchmark costs are not rendered as zero", () => {
  expect(formatCost(0.001)).toBe("<$0.0001");
  expect(formatCost(0)).toBe("$0.0000");
  expect(formatCost(null)).toBe("Unavailable");
});

test("cancelled benchmark items count as processed progress", () => {
  const run = buildRun(8, "cancelled");

  expect(processedItemCount(run)).toBe(run.total_items);
});

let mockDocumentSets: Array<{ id: number; name: string }> = [];

jest.mock("next/navigation", () => ({
  useRouter: () => ({ back: jest.fn(), push: jest.fn(), replace: jest.fn() }),
}));

jest.mock("@/lib/hooks/useDocumentSets", () => ({
  useDocumentSets: () => ({ documentSets: mockDocumentSets }),
}));

jest.mock("@/lib/regulatory/benchmark", () => ({
  cancelBenchmarkRun: jest.fn(),
  createBenchmarkQuestion: jest.fn(),
  createBenchmarkRun: jest.fn(),
  deleteBenchmarkQuestion: jest.fn(),
  getBenchmarkLimits: jest.fn(),
  getBenchmarkRun: jest.fn(),
  listBenchmarkCitationOptions: jest.fn(),
  listBenchmarkModels: jest.fn(),
  listBenchmarkQuestions: jest.fn(),
  listBenchmarkRunItems: jest.fn(),
  listBenchmarkRuns: jest.fn(),
  startBenchmarkRun: jest.fn(),
  updateBenchmarkQuestion: jest.fn(),
}));

const question: BenchmarkQuestion = {
  id: 11,
  title: "Transit declaration",
  prompt: "Which declaration is required?",
  reference_answer: null,
  expected_facts: [],
  expected_citations: [],
  as_of_date: null,
  rubric_notes: null,
  tags: ["transit"],
  document_set_id: 7,
  document_set_name: "Transit rules",
  is_active: true,
  created_at: "2026-08-10T09:00:00Z",
  updated_at: "2026-08-10T09:00:00Z",
};

const model: BenchmarkAvailableModel = {
  provider: "openrouter",
  model_id: "openai/gpt-5",
  provider_id: 3,
  display_name: "GPT-5",
  max_input_tokens: 128_000,
  is_visible: true,
};

const mockRunDetails = new Map<number, BenchmarkRun>();

function buildRun(
  id: number,
  status: BenchmarkRun["status"],
  label = `Run ${id}`
): BenchmarkRun {
  const run: BenchmarkRun = {
    id,
    label,
    status,
    judge_provider: "openrouter",
    judge_provider_id: 3,
    judge_model: "openai/gpt-5",
    deep_research: false,
    search_mode: "v2",
    total_items: 4,
    completed_items: status === "completed" ? 4 : 1,
    failed_items: status === "error" ? 1 : 0,
    queued_at: status === "pending" ? null : "2026-08-10T09:00:30Z",
    started_at:
      status === "pending" || status === "queued"
        ? null
        : "2026-08-10T09:01:00Z",
    heartbeat_at: status === "running" ? "2026-08-10T09:01:30Z" : null,
    completed_at:
      status === "completed" || status === "error"
        ? "2026-08-10T09:02:00Z"
        : null,
    created_at: "2026-08-10T09:00:00Z",
    failure_code: status === "error" ? "execution_failed" : null,
    failure_message:
      status === "error" ? "One or more benchmark items failed" : null,
    report: null,
    report_error: null,
    report_input_tokens: null,
    report_output_tokens: null,
    report_cost_cents: null,
    aggregates: [],
  };
  mockRunDetails.set(id, run);
  return run;
}

function buildRunningItem(): BenchmarkRunItem {
  return {
    id: 101,
    provider: "openrouter",
    provider_id: 3,
    model_id: "openai/gpt-5",
    question_id: question.id,
    question_prompt: question.prompt,
    question_title: question.title,
    question_snapshot: {},
    status: "running",
    execution_phase: "researching",
    heartbeat_at: "2026-08-10T09:01:30Z",
    started_at: "2026-08-10T09:01:00Z",
    completed_at: null,
    final_result: null,
    error_message: null,
    input_tokens: null,
    output_tokens: null,
    total_tokens: null,
    duration_ms: null,
    cost_cents: null,
    cost_source: "unavailable",
    cited_chunk_ids: [],
    cited_sources: [],
    execution_steps: [],
    llm_calls: [],
    answer_reasoning: null,
    chat_session_id: null,
    assistant_message_id: null,
    citation_recall: null,
    citation_precision: null,
    judge_error: null,
    judgment: null,
  };
}

const mockedCreateRun = jest.mocked(createBenchmarkRun);
const mockedCreateQuestion = jest.mocked(createBenchmarkQuestion);
const mockedGetLimits = jest.mocked(getBenchmarkLimits);
const mockedGetRun = jest.mocked(getBenchmarkRun);
const mockedListCitationOptions = jest.mocked(listBenchmarkCitationOptions);
const mockedListModels = jest.mocked(listBenchmarkModels);
const mockedListQuestions = jest.mocked(listBenchmarkQuestions);
const mockedListRunItems = jest.mocked(listBenchmarkRunItems);
const mockedListRuns = jest.mocked(listBenchmarkRuns);
const mockedStartRun = jest.mocked(startBenchmarkRun);

beforeAll(() => {
  window.HTMLElement.prototype.scrollIntoView = jest.fn();
});

async function openRuns() {
  const user = setupUser();
  await screen.findByText(question.title);
  await user.click(screen.getByRole("tab", { name: "Runs" }));
  await screen.findByRole("heading", { name: "Create a run" });
  return user;
}

async function configureRun() {
  const user = await openRuns();
  await user.click(
    screen.getByRole("checkbox", { name: /Transit declaration/i })
  );
  await user.click(screen.getByRole("checkbox", { name: /GPT-5/i }));
  act(() => {
    screen.getByRole("combobox", { name: "Judge model" }).focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "GPT-5" });
  await user.keyboard("{Enter}");
  return user;
}

beforeEach(() => {
  mockDocumentSets = [];
  mockRunDetails.clear();
  mockedListQuestions.mockResolvedValue([question]);
  mockedListRuns.mockResolvedValue([]);
  mockedListModels.mockResolvedValue([model]);
  mockedListCitationOptions.mockResolvedValue([]);
  mockedGetLimits.mockResolvedValue({
    max_questions: 100,
    max_candidates: 6,
    max_run_items: 300,
    default_item_page_size: 20,
    max_item_page_size: 50,
  });
  mockedGetRun.mockImplementation(async (runId) => {
    const run = mockRunDetails.get(runId);
    if (!run) throw new Error(`Missing run detail fixture ${runId}`);
    return run;
  });
  mockedListRunItems.mockResolvedValue({
    items: [],
    total: 0,
    offset: 0,
    limit: 20,
  });
});

afterEach(() => {
  jest.useRealTimers();
  jest.resetAllMocks();
});

test("switches between the Questions and Runs workspaces", async () => {
  const user = setupUser();
  render(<BenchmarkPage />);

  expect(
    await screen.findByRole("heading", { name: "Question library" })
  ).toBeInTheDocument();
  await screen.findByText(question.title);

  await user.click(screen.getByRole("tab", { name: "Runs" }));
  expect(
    await screen.findByRole("heading", { name: "Create a run" })
  ).toBeInTheDocument();

  await user.click(screen.getByRole("tab", { name: "Questions" }));
  expect(
    await screen.findByRole("heading", { name: "Question library" })
  ).toBeInTheDocument();
});

test("selects a created run before its start request finishes", async () => {
  const createdRun = buildRun(42, "pending", "August baseline");
  let finishStart: (run: BenchmarkRun) => void = () => undefined;
  mockedCreateRun.mockResolvedValue(createdRun);
  mockedStartRun.mockImplementation(
    () =>
      new Promise((resolve) => {
        finishStart = resolve;
      })
  );

  render(<BenchmarkPage />);
  const user = await configureRun();
  await user.click(screen.getByRole("button", { name: "Create and start" }));

  expect(
    await screen.findByRole("heading", { name: "August baseline" })
  ).toBeInTheDocument();
  expect(screen.getAllByText("Pending").length).toBeGreaterThan(0);

  await act(async () => {
    finishStart({
      ...createdRun,
      status: "queued",
      queued_at: "2026-08-10T09:00:30Z",
    });
  });
  expect(screen.getAllByText("Queued").length).toBeGreaterThan(0);
});

test("sends the selected provider row for candidates and the judge", async () => {
  const createdRun = buildRun(45, "pending", "Provider identity run");
  mockedCreateRun.mockResolvedValue(createdRun);
  mockedStartRun.mockResolvedValue({
    ...createdRun,
    status: "queued",
    queued_at: "2026-08-10T09:00:30Z",
  });

  render(<BenchmarkPage />);
  const user = await configureRun();
  await user.click(screen.getByRole("button", { name: "Create and start" }));

  await waitFor(() =>
    expect(mockedCreateRun).toHaveBeenCalledWith({
      label: null,
      question_ids: [question.id],
      candidates: [
        {
          provider: model.provider,
          provider_id: model.provider_id,
          model_id: model.model_id,
        },
      ],
      judge: {
        provider: model.provider,
        provider_id: model.provider_id,
        model_id: model.model_id,
      },
      deep_research: false,
      search_mode: "v2",
    })
  );
});

test("sends the search mode selected for a benchmark run", async () => {
  const createdRun = buildRun(46, "pending", "V1 comparison");
  mockedCreateRun.mockResolvedValue(createdRun);
  mockedStartRun.mockResolvedValue({
    ...createdRun,
    status: "queued",
    queued_at: "2026-08-10T09:00:30Z",
  });

  render(<BenchmarkPage />);
  const user = await configureRun();
  act(() => {
    screen.getByRole("combobox", { name: "Search mode" }).focus();
  });
  await user.keyboard("{ArrowDown}");
  await waitFor(() =>
    expect(screen.getByRole("option", { name: "Atez Search V2" })).toHaveFocus()
  );
  await user.keyboard("{End}");
  await waitFor(() =>
    expect(screen.getByRole("option", { name: "Atez Search V1" })).toHaveFocus()
  );
  await user.keyboard("{Enter}");
  await user.click(screen.getByRole("button", { name: "Create and start" }));

  await waitFor(() =>
    expect(mockedCreateRun).toHaveBeenCalledWith(
      expect.objectContaining({ search_mode: "v1" })
    )
  );
});

test("blocks a question and model combination above the run item capacity", async () => {
  const questions = Array.from({ length: 100 }, (_, index) => ({
    ...question,
    id: index + 1,
    title: `Question ${index + 1}`,
    prompt: `Prompt ${index + 1}`,
  }));
  const models = Array.from({ length: 4 }, (_, index) => ({
    ...model,
    provider_id: index + 1,
    model_id: `candidate/model-${index + 1}`,
    display_name: `Candidate ${index + 1}`,
  }));
  mockedListQuestions.mockResolvedValue(questions);
  mockedListModels.mockResolvedValue(models);

  render(<BenchmarkPage />);
  const user = setupUser();
  await screen.findByText("Question 1");
  await user.click(screen.getByRole("tab", { name: "Runs" }));
  await screen.findByRole("heading", { name: "Create a run" });
  await user.click(screen.getByRole("button", { name: "Select visible" }));
  for (const candidate of models) {
    await user.click(
      screen.getByRole("checkbox", {
        name: `${candidate.display_name}: ${candidate.model_id}`,
      })
    );
  }
  act(() => {
    screen.getByRole("combobox", { name: "Judge model" }).focus();
  });
  await user.keyboard("{ArrowDown}");
  await user.click(await screen.findByRole("option", { name: "Candidate 1" }));

  expect(screen.getByText("400 total run items")).toBeInTheDocument();
  expect(screen.getByText(/configured maximum is 300/i)).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Create and start" })
  ).toBeDisabled();
});

test("retains a newly created run when starting it fails", async () => {
  const createdRun = buildRun(43, "pending", "Start failure run");
  mockedCreateRun.mockResolvedValue(createdRun);
  mockedStartRun.mockRejectedValue(new Error("Queue is unavailable"));

  render(<BenchmarkPage />);
  const user = await configureRun();
  await user.click(screen.getByRole("button", { name: "Create and start" }));

  expect(
    await screen.findByRole("heading", { name: "Start failure run" })
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument();
});

test("retains an optimistic run until the run list observes it", async () => {
  jest.useFakeTimers();
  const createdRun = buildRun(44, "pending", "Awaiting publication");
  mockedCreateRun.mockResolvedValue(createdRun);
  mockedStartRun.mockRejectedValue(new Error("Queue is unavailable"));
  mockedListRuns.mockResolvedValue([]);

  render(<BenchmarkPage />);
  const user = await configureRun();
  await user.click(screen.getByRole("button", { name: "Create and start" }));
  expect(
    await screen.findByRole("heading", { name: "Awaiting publication" })
  ).toBeInTheDocument();

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });

  expect(
    screen.getByRole("heading", { name: "Awaiting publication" })
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument();
});

test.each([
  ["pending", "Start"],
  ["error", "Retry"],
] as const)(
  "offers %s runs a %s action and refreshes their status",
  async (status, actionLabel) => {
    const initialRun = buildRun(51, status, "Recoverable run");
    const queuedRun = {
      ...initialRun,
      status: "queued" as const,
      queued_at: "2026-08-10T10:00:00Z",
      started_at: null,
    };
    mockedListRuns
      .mockResolvedValueOnce([initialRun])
      .mockResolvedValue([queuedRun]);
    mockedStartRun.mockResolvedValue(queuedRun);

    render(<BenchmarkPage />);
    const user = await openRuns();
    await user.click(screen.getByRole("button", { name: actionLabel }));

    await waitFor(() => {
      expect(mockedStartRun).toHaveBeenCalledWith(initialRun.id);
      expect(screen.getAllByText("Queued").length).toBeGreaterThan(0);
      expect(
        screen.queryByRole("button", { name: actionLabel })
      ).not.toBeInTheDocument();
    });
  }
);

test("keeps a second run disabled while its start request is still pending", async () => {
  const firstRun = buildRun(54, "pending", "First pending run");
  const secondRun = buildRun(55, "pending", "Second pending run");
  const firstStart = deferred<BenchmarkRun>();
  const secondStart = deferred<BenchmarkRun>();
  mockedListRuns.mockResolvedValue([firstRun, secondRun]);
  mockedStartRun.mockImplementation((runId) =>
    runId === firstRun.id ? firstStart.promise : secondStart.promise
  );

  render(<BenchmarkPage />);
  const user = await openRuns();
  await user.click(screen.getByRole("button", { name: "Start" }));
  await user.click(screen.getByRole("button", { name: /Second pending run/i }));
  await user.click(screen.getByRole("button", { name: "Start" }));

  const secondStartingButton = screen.getByRole("button", {
    name: "Starting…",
  });
  expect(secondStartingButton).toBeDisabled();

  await act(async () => {
    firstStart.resolve({
      ...firstRun,
      status: "queued",
      queued_at: "2026-08-10T10:00:00Z",
    });
  });

  expect(screen.getByRole("button", { name: "Starting…" })).toBeDisabled();
  expect(
    screen.queryByRole("button", { name: "Start" })
  ).not.toBeInTheDocument();

  await act(async () => {
    secondStart.resolve({
      ...secondRun,
      status: "queued",
      queued_at: "2026-08-10T10:00:01Z",
    });
  });
});

test("shows a retried run that fails before its first poll", async () => {
  jest.useFakeTimers();
  const failedRun = buildRun(52, "error", "Fast retry failure");
  const activeRun = buildRun(53, "running", "Keeps polling");
  const retriedRun = {
    ...failedRun,
    status: "queued" as const,
    queued_at: "2026-08-10T10:00:00Z",
    started_at: null,
    heartbeat_at: null,
    completed_at: null,
    failure_code: null,
    failure_message: null,
  };
  const failedRetry = {
    ...failedRun,
    queued_at: retriedRun.queued_at,
    completed_at: "2026-08-10T10:00:01Z",
    failure_message: "Retry worker failed",
  };
  mockedListRuns
    .mockResolvedValueOnce([failedRun, activeRun])
    .mockResolvedValueOnce([failedRun, activeRun])
    .mockResolvedValue([failedRetry, activeRun]);
  mockedStartRun.mockResolvedValue(retriedRun);

  render(<BenchmarkPage />);
  const user = await openRuns();
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(screen.getAllByText("Queued").length).toBeGreaterThan(0);

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  expect(
    screen.queryByRole("button", { name: "Retry" })
  ).not.toBeInTheDocument();
  expect(screen.getAllByText("Queued").length).toBeGreaterThan(0);

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });

  expect((await screen.findAllByText("Failed")).length).toBeGreaterThan(0);
  expect(screen.getByText("Retry worker failed")).toBeInTheDocument();
  expect(
    screen.getByText("Failure code: execution_failed")
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
});

test("creates a question from the Questions workspace", async () => {
  const createdQuestion = {
    ...question,
    id: 12,
    title: "New transit question",
    prompt: "What must the carrier submit?",
  };
  mockDocumentSets = [{ id: 7, name: "Transit rules" }];
  mockedListQuestions
    .mockResolvedValueOnce([])
    .mockResolvedValue([createdQuestion]);
  mockedCreateQuestion.mockResolvedValue(createdQuestion);

  const user = setupUser();
  render(<BenchmarkPage />);

  await user.type(
    screen.getByRole("textbox", { name: "Question title" }),
    createdQuestion.title
  );
  act(() => {
    screen.getByRole("combobox", { name: "Document Set" }).focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.type(
    screen.getByRole("textbox", { name: "Question prompt" }),
    createdQuestion.prompt
  );
  await user.click(screen.getByRole("button", { name: "Create question" }));

  expect(await screen.findByText(createdQuestion.title)).toBeInTheDocument();
});

test("polls run status even when configuration refreshes fail", async () => {
  jest.useFakeTimers();
  const runningRun = buildRun(61, "running", "Independent poll");
  const completedRun = {
    ...runningRun,
    status: "completed" as const,
    completed_items: runningRun.total_items,
  };
  const catalogRequest = deferred<BenchmarkAvailableModel[]>();
  mockedListModels.mockReturnValue(catalogRequest.promise);
  mockedListRuns
    .mockResolvedValueOnce([runningRun])
    .mockResolvedValue([completedRun]);

  render(<BenchmarkPage />);
  await openRuns();
  expect((await screen.findAllByText("Running")).length).toBeGreaterThan(0);

  await act(async () => {
    catalogRequest.reject(new Error("Catalog unavailable"));
  });

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });

  expect((await screen.findAllByText("Completed")).length).toBeGreaterThan(0);
});

test("ignores an older poll that resolves after a run has started", async () => {
  jest.useFakeTimers();
  const pendingRun = buildRun(62, "pending", "Ordered run");
  const queuedRun = buildRun(62, "queued", "Ordered run");
  const runningRun = buildRun(62, "running", "Ordered run");
  const olderPoll = deferred<BenchmarkRun[]>();
  mockedListRuns
    .mockResolvedValueOnce([pendingRun])
    .mockReturnValueOnce(olderPoll.promise)
    .mockResolvedValue([runningRun]);
  mockedStartRun.mockResolvedValue(queuedRun);

  render(<BenchmarkPage />);
  const user = await openRuns();
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  await user.click(screen.getByRole("button", { name: "Start" }));
  expect((await screen.findAllByText("Queued")).length).toBeGreaterThan(0);

  await act(async () => {
    olderPoll.resolve([pendingRun]);
  });

  expect(screen.getAllByText("Queued").length).toBeGreaterThan(0);
  expect(
    screen.queryByRole("button", { name: "Start" })
  ).not.toBeInTheDocument();
});

test("continues polling after a transient run-list error", async () => {
  jest.useFakeTimers();
  const runningRun = buildRun(63, "running", "Resilient poll");
  const completedRun = {
    ...runningRun,
    status: "completed" as const,
    completed_items: runningRun.total_items,
  };
  mockedListRuns
    .mockResolvedValueOnce([runningRun])
    .mockRejectedValueOnce(new Error("Temporary polling error"))
    .mockResolvedValue([completedRun]);

  render(<BenchmarkPage />);
  await openRuns();
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  expect(screen.getAllByText("Running").length).toBeGreaterThan(0);

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  expect((await screen.findAllByText("Completed")).length).toBeGreaterThan(0);
});

test("waits for a slow poll to settle before scheduling the next one", async () => {
  jest.useFakeTimers();
  const runningRun = buildRun(64, "running", "Slow poll");
  const completedRun = {
    ...runningRun,
    status: "completed" as const,
    completed_items: runningRun.total_items,
  };
  const firstSlowPoll = deferred<BenchmarkRun[]>();
  const secondSlowPoll = deferred<BenchmarkRun[]>();
  mockedListRuns
    .mockResolvedValueOnce([runningRun])
    .mockReturnValueOnce(firstSlowPoll.promise)
    .mockReturnValueOnce(secondSlowPoll.promise)
    .mockResolvedValue([completedRun]);

  render(<BenchmarkPage />);
  await openRuns();
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  expect(mockedListRuns).toHaveBeenCalledTimes(2);

  await act(async () => {
    jest.advanceTimersByTime(6000);
  });
  expect(mockedListRuns).toHaveBeenCalledTimes(2);

  await act(async () => {
    firstSlowPoll.resolve([runningRun]);
  });
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  expect(mockedListRuns).toHaveBeenCalledTimes(3);

  await act(async () => {
    jest.advanceTimersByTime(6000);
  });
  expect(mockedListRuns).toHaveBeenCalledTimes(3);

  await act(async () => {
    secondSlowPoll.resolve([completedRun]);
  });
  expect((await screen.findAllByText("Completed")).length).toBeGreaterThan(0);
});

test("accepts an error run retried from another tab", async () => {
  jest.useFakeTimers();
  const failedRun = buildRun(65, "error", "External retry");
  const activeRun = buildRun(66, "running", "Keeps polling");
  const retriedRun = { ...failedRun, status: "running" as const };
  mockedListRuns
    .mockResolvedValueOnce([failedRun, activeRun])
    .mockResolvedValue([retriedRun, activeRun]);

  render(<BenchmarkPage />);
  await openRuns();
  expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });

  expect(
    screen.queryByRole("button", { name: "Retry" })
  ).not.toBeInTheDocument();
  expect(screen.getAllByText("Running").length).toBeGreaterThan(0);
});

test("distinguishes provider rows in report and aggregate cards without reordering the response", async () => {
  const completedRun: BenchmarkRun = {
    ...buildRun(67, "completed", "Provider identity report"),
    report: {
      executive_summary: "Provider comparison",
      model_reports: [
        {
          provider: "openrouter",
          provider_id: 9,
          model_id: "openai/gpt-5",
          rank: 2,
          summary: "Second provider row",
          strengths: [],
          weaknesses: [],
          recommended_use: "Fallback",
        },
        {
          provider: "openrouter",
          provider_id: 3,
          model_id: "openai/gpt-5",
          rank: 1,
          summary: "First provider row",
          strengths: [],
          weaknesses: [],
          recommended_use: "Primary",
        },
      ],
      common_failure_patterns: [],
      recommendation: "Prefer the first provider row.",
    },
    aggregates: [
      {
        provider: "openrouter",
        provider_id: 3,
        model_id: "openai/gpt-5",
        item_count: 1,
        completed_count: 1,
        failed_count: 0,
        average_score: 90,
        average_tokens: 100,
        average_duration_ms: 1_000,
        total_cost_cents: 1,
        average_citation_recall: 1,
        average_citation_precision: 1,
      },
      {
        provider: "openrouter",
        provider_id: 9,
        model_id: "openai/gpt-5",
        item_count: 1,
        completed_count: 1,
        failed_count: 0,
        average_score: 80,
        average_tokens: 120,
        average_duration_ms: 1_200,
        total_cost_cents: 2,
        average_citation_recall: 0.8,
        average_citation_precision: 0.9,
      },
    ],
  };
  mockedListRuns.mockResolvedValue([completedRun]);
  mockedGetRun.mockResolvedValue(completedRun);
  const originalReportOrder = completedRun.report!.model_reports.map(
    (report) => report.provider_id
  );

  render(<BenchmarkPage />);
  await openRuns();

  expect(
    screen.getByText("#1 · openrouter #3 · openai/gpt-5")
  ).toBeInTheDocument();
  expect(
    screen.getByText("#2 · openrouter #9 · openai/gpt-5")
  ).toBeInTheDocument();
  expect(screen.getByText("openrouter #3 · openai/gpt-5")).toBeInTheDocument();
  expect(screen.getByText("openrouter #9 · openai/gpt-5")).toBeInTheDocument();
  expect(
    completedRun.report!.model_reports.map((report) => report.provider_id)
  ).toEqual(originalReportOrder);
});

test("adds an expected citation and expands it in the question library", async () => {
  const citationOption = {
    chunk_id: "chunk-5",
    user_file_id: "file-1",
    file_name: "rules.pdf",
    heading_path: ["Article 5"],
    text_excerpt: "The carrier must submit the transit declaration.",
    status: "active" as const,
    validity_start_date: null,
    validity_end_date: null,
  };
  const createdQuestion: BenchmarkQuestion = {
    ...question,
    id: 13,
    title: "Citation-backed question",
    expected_citations: [
      {
        chunk_id: citationOption.chunk_id,
        requirement: "required",
        notes: null,
        file_name: citationOption.file_name,
        heading_path: citationOption.heading_path,
        text_excerpt: citationOption.text_excerpt,
      },
    ],
  };
  mockDocumentSets = [{ id: 7, name: "Transit rules" }];
  mockedListQuestions
    .mockResolvedValueOnce([])
    .mockResolvedValue([createdQuestion]);
  mockedListCitationOptions.mockResolvedValue([citationOption]);
  mockedCreateQuestion.mockResolvedValue(createdQuestion);

  const user = setupUser();
  render(<BenchmarkPage />);
  await user.type(
    screen.getByRole("textbox", { name: "Question title" }),
    createdQuestion.title
  );
  act(() => screen.getByRole("combobox", { name: "Document Set" }).focus());
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.type(
    screen.getByRole("textbox", { name: "Question prompt" }),
    question.prompt
  );
  await user.type(
    screen.getByRole("textbox", {
      name: /Expected citations/i,
    }),
    "Article 5"
  );
  await user.click(
    await screen.findByRole("button", { name: "Add citation rules.pdf" })
  );
  await user.click(screen.getByRole("button", { name: "Create question" }));

  expect(mockedCreateQuestion).toHaveBeenCalledWith(
    expect.objectContaining({
      expected_citations: [
        { chunk_id: "chunk-5", requirement: "required", notes: null },
      ],
    })
  );
  await user.click(await screen.findByText(createdQuestion.title));
  expect(
    await screen.findByText(/rules\.pdf · Article 5 \(required\)/)
  ).toBeInTheDocument();
});

test("ignores citation options from a previously selected document set", async () => {
  const oldRequest =
    deferred<Awaited<ReturnType<typeof listBenchmarkCitationOptions>>>();
  const currentRequest =
    deferred<Awaited<ReturnType<typeof listBenchmarkCitationOptions>>>();
  const currentQuestion = {
    ...question,
    id: 12,
    title: "Current customs declaration",
    document_set_id: 8,
    document_set_name: "Current customs rules",
  };
  mockedListQuestions.mockResolvedValue([question, currentQuestion]);
  mockedListCitationOptions.mockImplementation((documentSetId) => {
    if (documentSetId === 7) return oldRequest.promise;
    return currentRequest.promise;
  });

  const user = setupUser();
  render(<BenchmarkPage />);
  await user.click(await screen.findByText(question.title));
  await user.click(screen.getAllByRole("button", { name: "Edit" })[0]!);
  await waitFor(() =>
    expect(mockedListCitationOptions).toHaveBeenCalledWith(7)
  );

  await user.click(screen.getByText(currentQuestion.title));
  await user.click(screen.getAllByRole("button", { name: "Edit" })[1]!);
  await waitFor(() =>
    expect(mockedListCitationOptions).toHaveBeenCalledWith(8)
  );

  await act(async () => {
    currentRequest.resolve([
      {
        chunk_id: "current-chunk",
        user_file_id: "current-file",
        file_name: "current.pdf",
        heading_path: ["Article 8"],
        text_excerpt: "Current document-set evidence.",
        status: "active",
        validity_start_date: null,
        validity_end_date: null,
      },
    ]);
  });
  await act(async () => {
    oldRequest.resolve([
      {
        chunk_id: "stale-chunk",
        user_file_id: "stale-file",
        file_name: "stale.pdf",
        heading_path: ["Article 7"],
        text_excerpt: "Stale document-set evidence.",
        status: "active",
        validity_start_date: null,
        validity_end_date: null,
      },
    ]);
  });

  await user.type(
    screen.getByRole("textbox", { name: /Expected citations/i }),
    "pdf"
  );
  expect(
    await screen.findByRole("button", { name: "Add citation current.pdf" })
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Add citation stale.pdf" })
  ).not.toBeInTheDocument();
});

test("names the deep-research control and exposes run selection to keyboards", async () => {
  const firstRun = buildRun(71, "completed", "First run");
  const secondRun = buildRun(72, "completed", "Second run");
  mockedListRuns.mockResolvedValue([firstRun, secondRun]);

  render(<BenchmarkPage />);
  const user = await openRuns();

  const deepResearch = screen.getByRole("checkbox", {
    name: "Deep research",
  });
  deepResearch.focus();
  await user.keyboard(" ");
  expect(deepResearch).toHaveAttribute("aria-checked", "true");

  const secondRunCard = screen.getByRole("button", { name: /Second run/i });
  expect(secondRunCard).toHaveAttribute("aria-pressed", "false");
  secondRunCard.focus();
  const spaceEvent = new KeyboardEvent("keydown", {
    key: " ",
    bubbles: true,
    cancelable: true,
  });
  act(() => {
    secondRunCard.dispatchEvent(spaceEvent);
  });

  expect(spaceEvent.defaultPrevented).toBe(true);
  expect(secondRunCard).toHaveAttribute("aria-pressed", "true");
  expect(
    screen.getByRole("heading", { name: "Second run" })
  ).toBeInTheDocument();
});

test("shows the active phase and heartbeat for a running item", async () => {
  const run = buildRun(73, "running", "Live progress");
  mockedListRuns.mockResolvedValue([run]);
  mockedListRunItems.mockResolvedValue({
    items: [buildRunningItem()],
    total: 1,
    offset: 0,
    limit: 20,
  });

  render(<BenchmarkPage />);
  await openRuns();

  expect(
    screen.getByText("Deep research · heartbeat 09:01:30 UTC")
  ).toBeInTheDocument();
});

test("loads only the selected benchmark item page", async () => {
  const run = buildRun(74, "completed", "Paged results");
  run.total_items = 45;
  run.completed_items = 45;
  mockedListRuns.mockResolvedValue([run]);
  mockedListRunItems.mockResolvedValue({
    items: [buildRunningItem()],
    total: 45,
    offset: 0,
    limit: 20,
  });

  render(<BenchmarkPage />);
  const user = await openRuns();
  await waitFor(() =>
    expect(mockedListRunItems).toHaveBeenCalledWith(run.id, 0, 20)
  );
  await user.click(screen.getByRole("button", { name: "Next items" }));

  await waitFor(() =>
    expect(mockedListRunItems).toHaveBeenCalledWith(run.id, 20, 20)
  );
});

test("never renders item results from the previously selected run", async () => {
  const firstRun = buildRun(75, "completed", "First run");
  const secondRun = buildRun(76, "completed", "Second run");
  const firstItem = {
    ...buildRunningItem(),
    question_title: "Only belongs to first run",
  };
  mockedListRuns.mockResolvedValue([firstRun, secondRun]);
  mockedListRunItems.mockImplementation(async (runId) => {
    if (runId === firstRun.id) {
      return { items: [firstItem], total: 1, offset: 0, limit: 20 };
    }
    throw new Error("Second run page failed");
  });

  render(<BenchmarkPage />);
  const user = await openRuns();
  expect(await screen.findByText(firstItem.question_title)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /Second run/ }));

  expect(screen.queryByText(firstItem.question_title)).not.toBeInTheDocument();
});

test("refreshes terminal detail after observing the terminal summary", async () => {
  jest.useFakeTimers();
  const runningRun = buildRun(77, "running", "Finishing run");
  const completedRun = {
    ...runningRun,
    status: "completed" as const,
    completed_items: runningRun.total_items,
    report_error: "final report marker",
  };
  let terminalCommitObserved = false;
  mockedListRuns.mockResolvedValueOnce([runningRun]).mockImplementation(
    () =>
      new Promise((resolve) => {
        window.setTimeout(() => {
          terminalCommitObserved = true;
          resolve([completedRun]);
        }, 10);
      })
  );
  mockedGetRun.mockImplementation(async () =>
    terminalCommitObserved ? completedRun : runningRun
  );

  render(<BenchmarkPage />);
  await openRuns();
  await act(async () => {
    await jest.advanceTimersByTimeAsync(3010);
  });

  expect(
    await screen.findByText("Run report failed: final report marker")
  ).toBeInTheDocument();
});

test("does not let a cancelled poll refresh the previously selected run", async () => {
  jest.useFakeTimers();
  const firstRun = buildRun(78, "running", "First running run");
  const secondRun = buildRun(79, "running", "Second running run");
  const obsoletePoll = deferred<BenchmarkRun[]>();
  mockedListRuns
    .mockResolvedValueOnce([firstRun, secondRun])
    .mockReturnValueOnce(obsoletePoll.promise)
    .mockResolvedValue([firstRun, secondRun]);

  render(<BenchmarkPage />);
  const user = await openRuns();
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  await user.click(screen.getByRole("button", { name: /Second running run/ }));
  await waitFor(() =>
    expect(mockedListRunItems).toHaveBeenCalledWith(secondRun.id, 0, 20)
  );
  mockedListRunItems.mockClear();

  await act(async () => {
    obsoletePoll.resolve([firstRun, secondRun]);
  });

  expect(mockedListRunItems).not.toHaveBeenCalledWith(firstRun.id, 0, 20);
});
