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
  createBenchmarkQuestion,
  createBenchmarkRun,
  listBenchmarkCitationOptions,
  listBenchmarkModels,
  listBenchmarkQuestions,
  listBenchmarkRuns,
  startBenchmarkRun,
} from "@/lib/regulatory/benchmark";
import BenchmarkPage from "@/views/admin/BenchmarkPage";

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
  listBenchmarkCitationOptions: jest.fn(),
  listBenchmarkModels: jest.fn(),
  listBenchmarkQuestions: jest.fn(),
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

function buildRun(
  id: number,
  status: BenchmarkRun["status"],
  label = `Run ${id}`
): BenchmarkRun {
  return {
    id,
    label,
    status,
    judge_provider: "openrouter",
    judge_provider_id: 3,
    judge_model: "openai/gpt-5",
    deep_research: false,
    total_items: 4,
    completed_items: status === "completed" ? 4 : 1,
    failed_items: status === "error" ? 1 : 0,
    started_at: status === "pending" ? null : "2026-08-10T09:01:00Z",
    completed_at:
      status === "completed" || status === "error"
        ? "2026-08-10T09:02:00Z"
        : null,
    created_at: "2026-08-10T09:00:00Z",
    report: null,
    report_error: null,
    report_input_tokens: null,
    report_output_tokens: null,
    report_cost_cents: null,
    items: [],
    aggregates: [],
  };
}

const mockedCreateRun = jest.mocked(createBenchmarkRun);
const mockedCreateQuestion = jest.mocked(createBenchmarkQuestion);
const mockedListCitationOptions = jest.mocked(listBenchmarkCitationOptions);
const mockedListModels = jest.mocked(listBenchmarkModels);
const mockedListQuestions = jest.mocked(listBenchmarkQuestions);
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
  mockedListQuestions.mockResolvedValue([question]);
  mockedListRuns.mockResolvedValue([]);
  mockedListModels.mockResolvedValue([model]);
  mockedListCitationOptions.mockResolvedValue([]);
});

afterEach(() => {
  jest.useRealTimers();
  jest.clearAllMocks();
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
    finishStart({ ...createdRun, status: "running" });
  });
});

test("sends the selected provider row for candidates and the judge", async () => {
  const createdRun = buildRun(45, "pending", "Provider identity run");
  mockedCreateRun.mockResolvedValue(createdRun);
  mockedStartRun.mockResolvedValue({ ...createdRun, status: "running" });

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
    })
  );
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
    const runningRun = { ...initialRun, status: "running" as const };
    mockedListRuns
      .mockResolvedValueOnce([initialRun])
      .mockResolvedValue([runningRun]);
    mockedStartRun.mockResolvedValue(runningRun);

    render(<BenchmarkPage />);
    const user = await openRuns();
    await user.click(screen.getByRole("button", { name: actionLabel }));

    await waitFor(() => {
      expect(mockedStartRun).toHaveBeenCalledWith(initialRun.id);
      expect(screen.getAllByText("Running").length).toBeGreaterThan(0);
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
    firstStart.resolve({ ...firstRun, status: "running" });
  });

  expect(screen.getByRole("button", { name: "Starting…" })).toBeDisabled();
  expect(
    screen.queryByRole("button", { name: "Start" })
  ).not.toBeInTheDocument();

  await act(async () => {
    secondStart.resolve({ ...secondRun, status: "running" });
  });
});

test("shows a retried run that fails before its first poll", async () => {
  jest.useFakeTimers();
  const failedRun = buildRun(52, "error", "Fast retry failure");
  const activeRun = buildRun(53, "running", "Keeps polling");
  const retriedRun = {
    ...failedRun,
    status: "running" as const,
    started_at: "2026-08-10T10:00:00Z",
    completed_at: null,
  };
  const failedRetry = {
    ...failedRun,
    started_at: retriedRun.started_at,
    completed_at: "2026-08-10T10:00:01Z",
    error_message: "Retry worker failed",
  };
  mockedListRuns
    .mockResolvedValueOnce([failedRun, activeRun])
    .mockResolvedValueOnce([failedRun, activeRun])
    .mockResolvedValue([failedRetry, activeRun]);
  mockedStartRun.mockResolvedValue(retriedRun);

  render(<BenchmarkPage />);
  const user = await openRuns();
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(screen.getAllByText("Running").length).toBeGreaterThan(0);

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  expect(
    screen.queryByRole("button", { name: "Retry" })
  ).not.toBeInTheDocument();
  expect(screen.getAllByText("Running").length).toBeGreaterThan(0);

  await act(async () => {
    jest.advanceTimersByTime(3000);
  });

  expect((await screen.findAllByText("Failed")).length).toBeGreaterThan(0);
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
  const runningRun = { ...pendingRun, status: "running" as const };
  const olderPoll = deferred<BenchmarkRun[]>();
  mockedListRuns
    .mockResolvedValueOnce([pendingRun])
    .mockReturnValueOnce(olderPoll.promise)
    .mockResolvedValue([runningRun]);
  mockedStartRun.mockResolvedValue(runningRun);

  render(<BenchmarkPage />);
  const user = await openRuns();
  await act(async () => {
    jest.advanceTimersByTime(3000);
  });
  await user.click(screen.getByRole("button", { name: "Start" }));
  expect((await screen.findAllByText("Running")).length).toBeGreaterThan(0);

  await act(async () => {
    olderPoll.resolve([pendingRun]);
  });

  expect(screen.getAllByText("Running").length).toBeGreaterThan(0);
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
