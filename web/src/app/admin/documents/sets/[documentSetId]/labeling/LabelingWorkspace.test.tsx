import {
  act,
  render,
  screen,
  setupUser,
  waitFor,
} from "@tests/setup/test-utils";

import LabelingWorkspace from "@/app/admin/documents/sets/[documentSetId]/labeling/LabelingWorkspace";
import type {
  LabelingRun,
  LabelingSetup,
} from "@/lib/documentSetLabeling/interfaces";

const baseUrl = "/api/manage/admin/document-set/7/labeling";

const readySetup: LabelingSetup = {
  model: "gemini-3.8-flash",
  default_label_count: 255,
  taxonomies: [
    {
      id: "taxonomy-v1",
      name: "Customs scope v1",
      version_hash: "abc123def456",
      label_count: 3,
      created_at: "2026-09-13T08:00:00Z",
    },
  ],
  providers: [{ id: 41, name: "Google batch service account" }],
  counts: { files: 2, canonical_chunks: 12, derived_chunks: 20 },
  active_run_id: null,
  warnings: [],
};

function buildRun(overrides: Partial<LabelingRun> = {}): LabelingRun {
  return {
    id: "run-1",
    document_set_id: 7,
    taxonomy_id: "taxonomy-v1",
    taxonomy_name: "Customs scope v1",
    model: "gemini-3.8-flash",
    status: "running",
    stage: "waiting",
    total_chunks: 12,
    completed_chunks: 0,
    failed_chunks: 0,
    stale_chunks: 0,
    derived_chunks: 20,
    unresolved_derived_chunks: 0,
    created_at: "2026-09-13T09:00:00Z",
    updated_at: "2026-09-13T09:01:00Z",
    finished_at: null,
    error: null,
    cancel_requested: false,
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function installFetchRouter(options?: {
  setup?: LabelingSetup;
  runs?: LabelingRun[];
  items?: unknown;
}) {
  const setup = options?.setup ?? readySetup;
  const runs = options?.runs ?? [];
  return jest.spyOn(global, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url === `${baseUrl}/setup`) return jsonResponse(setup);
    if (url === `${baseUrl}/runs`) return jsonResponse(runs);
    if (url.includes("/items?"))
      return jsonResponse(options?.items ?? { items: [], total: 0 });
    const run = runs.find(
      (candidate) => url === `${baseUrl}/runs/${candidate.id}`
    );
    if (run) return jsonResponse(run);
    throw new Error(`Unexpected request: ${url}`);
  });
}

afterEach(() => {
  jest.useRealTimers();
  jest.restoreAllMocks();
  window.localStorage.clear();
});

test("shows bundled labels and keeps Start Labeling unavailable until a provider and canonical chunks exist", async () => {
  installFetchRouter({
    setup: {
      ...readySetup,
      taxonomies: [],
      providers: [],
      counts: { files: 2, canonical_chunks: 0, derived_chunks: 0 },
    },
  });

  render(<LabelingWorkspace documentSetId={7} />);

  expect(await screen.findByText("255 labels ready")).toBeInTheDocument();
  expect(
    screen.getByText("No Gemini connection configured")
  ).toBeInTheDocument();
  expect(screen.getByText("No canonical chunks ready")).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Upload taxonomy" })
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("combobox", { name: "Taxonomy version" })
  ).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Start Labeling" })).toBeDisabled();
});

test("prompts for a provider selection when multiple providers are available", async () => {
  installFetchRouter({
    setup: {
      ...readySetup,
      providers: [
        ...readySetup.providers,
        { id: 42, name: "Google workload identity" },
      ],
    },
  });

  render(<LabelingWorkspace documentSetId={7} />);

  expect(
    await screen.findByRole("combobox", { name: "Gemini connection" })
  ).toHaveTextContent("Select a Gemini connection");
});

test("shows batch configuration errors and prevents starting while preserving label settings", async () => {
  const configurationError =
    "Gemini Batch storage is not configured. Configure its Cloud Storage location before starting labeling.";
  installFetchRouter({
    setup: {
      ...readySetup,
      providers: readySetup.providers.map((provider) => ({
        ...provider,
        configuration_error: configurationError,
      })),
    },
  });
  render(<LabelingWorkspace documentSetId={7} />);

  expect(await screen.findByText(configurationError)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Start Labeling" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Label Settings" })).toBeEnabled();
});

test("allows a new run when history confirms the setup's active run has finished", async () => {
  const run = buildRun({ status: "completed", stage: "finished" });
  installFetchRouter({
    setup: { ...readySetup, active_run_id: run.id },
    runs: [run],
  });

  render(<LabelingWorkspace documentSetId={7} />);

  const startButton = await screen.findByRole("button", {
    name: "Start Labeling",
  });
  await waitFor(() => expect(startButton).toBeEnabled());
});

test.each(["network", 502, 503, 408] as const)(
  "reuses the same idempotency key after an uncertain %s start response",
  async (failure) => {
    const user = setupUser();
    let startRequests = 0;
    const fetchSpy = jest
      .spyOn(global, "fetch")
      .mockImplementation(async (input, init) => {
        const url = String(input);
        if (url === `${baseUrl}/setup`) return jsonResponse(readySetup);
        if (url === `${baseUrl}/runs` && init?.method === "POST") {
          startRequests += 1;
          if (startRequests === 1) {
            if (failure === "network")
              throw new TypeError("Network connection lost");
            return jsonResponse(
              { detail: "The start request could not be confirmed" },
              failure
            );
          }
          return jsonResponse(buildRun({ status: "queued" }));
        }
        if (url === `${baseUrl}/runs`) return jsonResponse([]);
        throw new Error(`Unexpected request: ${url}`);
      });

    render(<LabelingWorkspace documentSetId={7} />);
    const startButton = await screen.findByRole("button", {
      name: "Start Labeling",
    });
    await waitFor(() => expect(startButton).toBeEnabled());

    await user.click(startButton);
    expect(
      await screen.findByText(/The start request could not be confirmed/i)
    ).toBeInTheDocument();
    await user.click(startButton);

    const startCalls = fetchSpy.mock.calls.filter(
      ([input, init]) =>
        String(input) === `${baseUrl}/runs` && init?.method === "POST"
    );
    expect(startCalls).toHaveLength(2);
    const firstBody = JSON.parse(String(startCalls[0]?.[1]?.body));
    const secondBody = JSON.parse(String(startCalls[1]?.[1]?.body));
    expect(firstBody).toEqual({
      model_configuration_id: 41,
      idempotency_key: expect.stringMatching(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
      ),
    });
    expect(secondBody).toEqual(firstBody);
  }
);

test("does not reuse a pending key created for a custom taxonomy request", async () => {
  const user = setupUser();
  window.localStorage.setItem(
    "document-set-labeling:7:pending-start",
    JSON.stringify({
      key: "83775c4e-0726-4a36-88ad-ec33e612f5cb",
      taxonomyId: "taxonomy-v1",
      providerId: 41,
    })
  );
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === `${baseUrl}/setup`) return jsonResponse(readySetup);
      if (url === `${baseUrl}/runs` && init?.method === "POST")
        return jsonResponse(buildRun({ status: "queued" }));
      if (url === `${baseUrl}/runs`) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    });

  render(<LabelingWorkspace documentSetId={7} />);
  const startButton = await screen.findByRole("button", {
    name: "Start Labeling",
  });
  await waitFor(() => expect(startButton).toBeEnabled());
  await user.click(startButton);

  const startCall = fetchSpy.mock.calls.find(
    ([input, init]) =>
      String(input) === `${baseUrl}/runs` && init?.method === "POST"
  );
  const body = JSON.parse(String(startCall?.[1]?.body));
  expect(body).toEqual({
    model_configuration_id: 41,
    idempotency_key: expect.any(String),
  });
  expect(body.idempotency_key).not.toBe("83775c4e-0726-4a36-88ad-ec33e612f5cb");
});

test("opens Label Settings beside Start and refreshes the saved label count", async () => {
  const user = setupUser();
  let setupRequests = 0;
  let currentSettings = {
    revision: 1,
    taxonomy_id: "taxonomy-v1",
    labels: [{ id: "CUS.VALUE", name: "Value", description: "Valuation." }],
    updated_at: "2026-09-13T10:00:00Z",
  };
  jest.spyOn(global, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    if (url === `${baseUrl}/setup`) {
      setupRequests += 1;
      return jsonResponse({
        ...readySetup,
        default_label_count: setupRequests === 1 ? 255 : 256,
      });
    }
    if (url === `${baseUrl}/runs`) return jsonResponse([]);
    if (url === `${baseUrl}/label-settings` && init?.method === "PUT") {
      currentSettings = {
        revision: 2,
        taxonomy_id: "taxonomy-v2",
        labels: [
          { id: "CUS.VALUE", name: "Value", description: "Valuation." },
          { id: "NEW", name: "New", description: "New definition." },
        ],
        updated_at: "2026-09-13T11:00:00Z",
      };
      return jsonResponse(currentSettings);
    }
    if (url === `${baseUrl}/label-settings`)
      return jsonResponse(currentSettings);
    throw new Error(`Unexpected request: ${url}`);
  });

  render(<LabelingWorkspace documentSetId={7} />);
  await user.click(
    await screen.findByRole("button", { name: "Label Settings" })
  );
  await screen.findByLabelText("Label description");
  await user.click(screen.getByRole("button", { name: "Add label" }));
  await user.type(screen.getByLabelText("Label ID"), "NEW");
  await user.type(screen.getByLabelText("Label name"), "New");
  await user.type(
    screen.getByLabelText("Label description"),
    "New definition."
  );
  await user.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await screen.findByText("256 labels ready")).toBeInTheDocument();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Label Settings" }));
  expect(
    await screen.findByRole("button", { name: /NEW.*New/ })
  ).toBeInTheDocument();
});

test("polls active runs every five seconds and shows provider waiting without a fake percentage", async () => {
  jest.useFakeTimers();
  const runningRun = buildRun();
  let runListRequests = 0;
  jest.spyOn(global, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url === `${baseUrl}/setup`)
      return jsonResponse({ ...readySetup, active_run_id: runningRun.id });
    if (url === `${baseUrl}/runs`) {
      runListRequests += 1;
      return jsonResponse([runningRun]);
    }
    if (url === `${baseUrl}/runs/${runningRun.id}`)
      return jsonResponse(runningRun);
    if (url.includes("/items?")) return jsonResponse({ items: [], total: 0 });
    throw new Error(`Unexpected request: ${url}`);
  });

  render(<LabelingWorkspace documentSetId={7} />);
  expect(
    await screen.findByText("Waiting for Gemini Batch")
  ).toBeInTheDocument();
  expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();

  await act(async () => {
    jest.advanceTimersByTime(5000);
  });
  await waitFor(() => expect(runListRequests).toBeGreaterThanOrEqual(2));
});

test("cancels an active run and reports a server error without hiding the run", async () => {
  const user = setupUser();
  const run = buildRun();
  const fetchSpy = installFetchRouter({ runs: [run] });
  fetchSpy.mockImplementation(async (input, init) => {
    const url = String(input);
    if (url === `${baseUrl}/setup`) return jsonResponse(readySetup);
    if (url === `${baseUrl}/runs`) return jsonResponse([run]);
    if (url === `${baseUrl}/runs/${run.id}`) return jsonResponse(run);
    if (url.includes("/items?")) return jsonResponse({ items: [], total: 0 });
    if (url === `${baseUrl}/runs/${run.id}/cancel` && init?.method === "POST")
      return jsonResponse({ detail: "Provider rejected cancellation" }, 409);
    throw new Error(`Unexpected request: ${url}`);
  });

  render(<LabelingWorkspace documentSetId={7} />);
  await user.click(await screen.findByRole("button", { name: "Cancel run" }));

  expect(
    await screen.findByText("Provider rejected cancellation")
  ).toBeInTheDocument();
  expect(screen.getByText("Waiting for Gemini Batch")).toBeInTheDocument();
});

test("resumes the same run and replaces its cached failure immediately", async () => {
  const user = setupUser();
  const failedRun = buildRun({
    id: "failed-run",
    status: "failed",
    stage: "finished",
    error: "Labeling worker failed: ValueError",
  });
  const replacementRun = buildRun({
    id: failedRun.id,
    status: "queued",
    stage: "preparing",
  });
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === `${baseUrl}/setup`) return jsonResponse(readySetup);
      if (url === `${baseUrl}/runs`) return jsonResponse([failedRun]);
      if (url === `${baseUrl}/runs/${failedRun.id}`)
        return jsonResponse(failedRun);
      if (url.includes("/items?")) return jsonResponse({ items: [], total: 0 });
      if (
        url === `${baseUrl}/runs/${failedRun.id}/resume` &&
        init?.method === "POST"
      )
        return jsonResponse(replacementRun);
      throw new Error(`Unexpected request: ${url}`);
    });

  render(<LabelingWorkspace documentSetId={7} />);
  await user.click(
    await screen.findByRole("button", { name: "Resume labeling" })
  );

  expect(fetchSpy).toHaveBeenCalledWith(
    `${baseUrl}/runs/${failedRun.id}/resume`,
    expect.objectContaining({ method: "POST" })
  );
  expect(
    await screen.findByText(
      "Run resumed; completed chunks and submitted batches are preserved"
    )
  ).toBeInTheDocument();
  expect(
    await screen.findByRole("button", { name: "Cancel run" })
  ).toBeInTheDocument();
  expect(
    screen.queryByText("Labeling worker failed: ValueError")
  ).not.toBeInTheDocument();
});

test("pages through chunk outcomes and keeps item errors visible", async () => {
  const user = setupUser();
  const run = buildRun({ status: "completed_with_errors", stage: "finished" });
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockImplementation(async (input) => {
      const url = String(input);
      if (url === `${baseUrl}/setup`) return jsonResponse(readySetup);
      if (url === `${baseUrl}/runs`) return jsonResponse([run]);
      if (url === `${baseUrl}/runs/${run.id}`) return jsonResponse(run);
      if (url.endsWith("items?offset=0&limit=25"))
        return jsonResponse({
          items: [
            {
              chunk_id: "chunk-1",
              file_id: "file-1",
              status: "completed",
              labels: ["import-duty"],
              error: null,
            },
          ],
          total: 30,
        });
      if (url.endsWith("items?offset=25&limit=25"))
        return jsonResponse({
          items: [
            {
              chunk_id: "chunk-26",
              file_id: "file-2",
              status: "failed",
              labels: [],
              error: "Invalid provider output",
            },
          ],
          total: 30,
        });
      throw new Error(`Unexpected request: ${url}`);
    });

  render(<LabelingWorkspace documentSetId={7} />);
  expect(await screen.findByText("import-duty")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Next" }));

  expect(
    await screen.findByText("Invalid provider output")
  ).toBeInTheDocument();
  expect(fetchSpy).toHaveBeenCalledWith(
    `${baseUrl}/runs/${run.id}/items?offset=25&limit=25`
  );
});

test("shows unresolved derived chunks on a completed-with-errors run", async () => {
  const run = {
    ...buildRun({ status: "completed_with_errors", stage: "finished" }),
    unresolved_derived_chunks: 2,
  } as LabelingRun;
  installFetchRouter({ runs: [run] });

  render(<LabelingWorkspace documentSetId={7} />);

  expect(await screen.findByText("Unresolved derived")).toBeInTheDocument();
  expect(screen.getAllByText("Completed With Errors").length).toBeGreaterThan(
    0
  );
});

test("does not expose setup details when labeling is forbidden or unavailable", async () => {
  jest.spyOn(global, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url === `${baseUrl}/setup`)
      return jsonResponse({ detail: "Secret" }, 403);
    if (url === `${baseUrl}/runs`)
      return jsonResponse({ detail: "Secret" }, 403);
    throw new Error(`Unexpected request: ${url}`);
  });

  render(<LabelingWorkspace documentSetId={7} />);

  expect(
    await screen.findByText("Chunk labeling is unavailable")
  ).toBeInTheDocument();
  expect(screen.queryByText("Secret")).not.toBeInTheDocument();
});
