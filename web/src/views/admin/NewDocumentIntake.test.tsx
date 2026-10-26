import {
  act,
  deferred,
  render,
  screen,
  setupUser,
  waitFor,
  within,
} from "@tests/setup/test-utils";
import type { DocumentIntakeBatch } from "@/lib/regulatory/documentIntake";
import type { AmendmentSourcePackage } from "@/lib/regulatory/amendments";
import NewDocumentIntake from "@/views/admin/NewDocumentIntake";

const base = "/api/regulatory/document-intake";
const sourceBase = "/api/regulatory/amendments/source-packages";
const sourcePackage: AmendmentSourcePackage = {
  id: "source-1",
  document_set_id: 7,
  status: "ready",
  asset_count: 2,
  total_bytes: 100,
  manifest_sha256: "a".repeat(64),
  issues: [],
  created_at: "2026-10-26",
  updated_at: "2026-10-26",
  assets: [
    {
      id: "root-asset",
      display_name: "Main regulation.html",
      mime_type: "text/html",
      byte_count: 50,
      sha256: "b".repeat(64),
      text_sha256: "c".repeat(64),
      original_url: null,
      final_url: null,
    },
    {
      id: "annex-asset",
      display_name: "EK-1.pdf",
      mime_type: "application/pdf",
      byte_count: 50,
      sha256: "d".repeat(64),
      text_sha256: "e".repeat(64),
      original_url: "https://example.gov/EK-1.pdf",
      final_url: "https://example.gov/EK-1.pdf",
    },
  ],
};
const readyBatch: DocumentIntakeBatch = {
  id: "intake-1",
  source_package_id: sourcePackage.id,
  status: "ready",
  stage: "review",
  target_document_set_id: 7,
  target_document_set_name: "PC Külliyatı",
  review_sha256: "f".repeat(64),
  issues: [],
  documents: [
    {
      id: "root",
      asset_id: "root-asset",
      display_name: "Main regulation.html",
      role: "root",
      parent_document_id: null,
      source_url: null,
      status: "ready",
      chunk_count: 1,
      labels: { topic: ["customs"] },
      error: null,
      user_file_id: "file-1",
    },
    {
      id: "annex",
      asset_id: "annex-asset",
      display_name: "EK-1.pdf",
      role: "annex",
      parent_document_id: "root",
      source_url: "https://example.gov/EK-1.pdf",
      status: "ready",
      chunk_count: 1,
      labels: { topic: ["duty"] },
      error: null,
      user_file_id: "file-2",
    },
  ],
};

function json(value: unknown): Response {
  return { ok: true, status: 200, json: async () => value } as Response;
}

function mockApi(
  options: {
    source?: AmendmentSourcePackage;
    history?: DocumentIntakeBatch[];
    enabled?: boolean;
    prepareResponse?: Promise<Response>;
    configurationErrors?: string[];
    polledBatches?: DocumentIntakeBatch[];
    polledSources?: AmendmentSourcePackage[];
    chunksByDocument?: Record<string, string>;
    predecessors?: {
      user_file_id: string;
      title: string;
      reason: string;
      score: number;
      status: string;
    }[];
    editResponse?: Response;
    lineageResponse?: Response;
  } = {}
) {
  let batch = readyBatch;
  let pollIndex = 0;
  let sourcePollIndex = 0;
  const fetchMock = jest.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === `${base}/source-packages/source-1/links`)
        return json([
          {
            source_field: "html:0",
            label: "Ekler için tıklayınız",
            original_url: "ekler/ek-1.html",
            requested_url: null,
            final_url: null,
            target_asset_hash: null,
          },
        ]);
      // Intake capability and PC labeling setup.
      if (url === `${base}/capabilities`)
        return json({
          enabled: options.enabled ?? true,
          target_document_set_id: 7,
          target_document_set_name: "PC Külliyatı",
          error:
            options.enabled === false
              ? "PC Külliyatı is not configured."
              : null,
        });
      if (url === `${base}/labeling-setup`)
        return json({
          model: "gemini-3.5-flash",
          default_label_count: 255,
          providers: [{ id: 41, name: "Google batch service account" }],
          taxonomies: [],
          counts: { files: 100, canonical_chunks: 2000, derived_chunks: 0 },
          active_run_id: null,
          warnings: [],
          configuration_errors: options.configurationErrors ?? [],
        });
      // Frozen source-package creation and status polling.
      if (
        url === `${sourceBase}/upload` ||
        url === sourceBase ||
        url === `${base}/source-packages/markdown`
      )
        return options.prepareResponse ?? json(options.source ?? sourcePackage);
      if (url.startsWith(`${sourceBase}/source-1?`)) {
        const polled = options.polledSources;
        return json(
          polled?.length
            ? polled[Math.min(sourcePollIndex++, polled.length - 1)]
            : (options.source ?? sourcePackage)
        );
      }
      // Intake history, preparation, approval and status polling.
      if (url === `${base}/batches` && init?.method === "POST")
        return json(batch);
      if (url === `${base}/batches`) return json(options.history ?? []);
      if (url === `${base}/batches/intake-1/approve`) {
        batch = { ...batch, status: "approving", stage: "indexing" };
        return json(batch);
      }
      if (url === `${base}/batches/intake-1`) {
        const polled = options.polledBatches;
        return json(
          polled?.length
            ? polled[Math.min(pollIndex++, polled.length - 1)]
            : (options.history?.[0] ?? batch)
        );
      }
      if (url.includes("/predecessor-candidates"))
        return json(options.predecessors ?? []);
      if (url.endsWith("/predecessor-selection") && init?.method === "PUT") {
        if (options.lineageResponse) return options.lineageResponse;
        const selection = JSON.parse(String(init.body));
        batch = {
          ...batch,
          review_sha256: "a".repeat(64),
          documents: batch.documents.map((document) =>
            document.id === "root"
              ? {
                  ...document,
                  lineage_review: {
                    ...selection,
                    title: options.predecessors?.[0]?.title ?? "Previous law",
                    status: "draft",
                  },
                }
              : document
          ),
        };
        return json(batch);
      }
      if (/\/chunks\/[^?]+$/.test(url) && init?.method === "PATCH") {
        if (options.editResponse) return options.editResponse;
        batch = {
          ...batch,
          status: "preparing",
          stage: "labeling",
          review_sha256: null,
        };
        return json(batch);
      }
      // Bounded chunk preview for one newly imported file.
      const chunkDocumentId = url.match(/\/documents\/([^/]+)\/chunks\?/)?.[1];
      if (chunkDocumentId && options.chunksByDocument?.[chunkDocumentId])
        return json({
          items: [
            {
              id: `chunk-${chunkDocumentId}`,
              chunk_type: "madde",
              validity_start_date: null,
              validity_end_date: null,
              position: 0,
              heading_path: [],
              text: options.chunksByDocument[chunkDocumentId],
              chunk_metadata: { labels: ["customs"] },
            },
          ],
          total: 1,
          offset: 0,
          limit: 10,
        });
      if (url.includes("/documents/annex/chunks?"))
        return json({
          items: [
            {
              id: "chunk-1",
              position: 0,
              heading_path: ["EK-1"],
              text: "Attachment duty table",
              chunk_metadata: { labels: ["duty"] },
            },
          ],
          total: 1,
          offset: 0,
          limit: 10,
        });
      if (chunkDocumentId)
        return json({
          items: [
            {
              id: `chunk-${chunkDocumentId}`,
              chunk_type: "madde",
              validity_start_date: null,
              validity_end_date: null,
              position: 0,
              heading_path: ["MADDE 1"],
              text: "Main document rule",
              chunk_metadata: { labels: ["customs"] },
            },
          ],
          total: 1,
          offset: 0,
          limit: 10,
        });
      throw new Error(`Unexpected request: ${url}`);
    }
  );
  global.fetch = fetchMock;
  return fetchMock;
}

test("previews separate main and attachment documents and approves only the reviewed new import", async () => {
  const fetchMock = mockApi();
  const user = setupUser();
  render(<NewDocumentIntake />);
  await screen.findByText(/will be added to PC Külliyatı/);
  expect(screen.getByLabelText("New document source file")).toHaveAttribute(
    "accept",
    ".pdf,.docx,.html,.htm,.xlsx,.png,.jpg,.jpeg,.webp,.tif,.tiff"
  );
  expect(
    screen.queryByLabelText("Markdown package file")
  ).not.toBeInTheDocument();
  expect(screen.getByText("Gemini 3.5 Flash")).toBeVisible();
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  expect(
    screen.queryByLabelText("New document labeling model")
  ).not.toBeInTheDocument();
  const file = new File(
    ["<a href='EK-1.pdf'>Ekler için tıklayınız</a>"],
    "Main regulation.html",
    { type: "text/html" }
  );
  await user.upload(screen.getByLabelText("New document source file"), file);
  await user.type(
    screen.getByLabelText("Original document URL"),
    "https://example.gov/main.html"
  );
  await user.click(
    screen.getByRole("button", { name: "Prepare source and attachments" })
  );
  await screen.findByText("Source package ready · 2 documents");
  const upload = fetchMock.mock.calls.find(
    ([url]) => url === `${sourceBase}/upload`
  )![1]!.body as FormData;
  expect(upload.get("base_url")).toBe("https://example.gov/main.html");
  expect(upload.get("document_set_id")).toBe("7");
  await user.click(
    screen.getByRole("button", { name: "Prepare new documents" })
  );
  const root = await screen.findByRole("region", {
    name: "Main regulation.html",
  });
  const annex = screen.getByRole("region", { name: "EK-1.pdf" });
  expect(within(root).getByText("Main document")).toBeVisible();
  expect(
    within(annex).getByText("Attachment to Main regulation.html")
  ).toBeVisible();
  expect(screen.getByText("2 new files · 2 added chunks")).toBeVisible();
  expect(
    await within(root).findByDisplayValue("Main document rule")
  ).toBeVisible();
  const added = within(root).getByRole("region", { name: "Added chunk 1" });
  expect(within(added).getByText("Added")).toBeVisible();
  expect(within(added).getByText(/^New chunk 1/)).toBeVisible();
  expect(within(added).getByLabelText("After text")).toBeEnabled();
  expect(within(added).getByText("(new — no existing chunk)")).toBeVisible();
  expect(
    within(added).getByRole("table", { name: "After chunk fields" })
  ).toBeVisible();
  expect(
    within(added).getByLabelText("After metadata article_no")
  ).toBeEnabled();
  expect(
    within(added).getByLabelText("After validity_start_date")
  ).toBeEnabled();
  expect(
    within(added).queryByRole("button", { name: "Edit chunk" })
  ).not.toBeInTheDocument();
  expect(within(added).getByText("File: Main regulation.html")).toBeVisible();
  expect(within(added).getByText("Chunk labels")).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Preview chunks and labels" })
  ).not.toBeInTheDocument();
  expect(
    within(root).getByRole("button", { name: "Review previous legislation" })
  ).toBeVisible();
  expect(
    within(root).queryByRole("button", { name: "Search previous legislation" })
  ).not.toBeInTheDocument();
  expect(
    await within(annex).findByDisplayValue("Attachment duty table")
  ).toBeVisible();
  expect(within(annex).getByText("duty")).toBeVisible();
  await user.click(
    screen.getByRole("button", { name: "Approve all new documents" })
  );
  await screen.findByText("New documents: approving");
  expect(screen.queryByText(/Import completed in/)).not.toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledWith(
    `${base}/batches/intake-1/approve`,
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        expected_review_sha256: readyBatch.review_sha256,
      }),
    })
  );
  const creation = fetchMock.mock.calls.find(
    ([url, init]) => url === `${base}/batches` && init?.method === "POST"
  )!;
  expect(JSON.parse(String(creation[1]!.body))).toEqual({
    source_package_id: "source-1",
    idempotency_key: expect.any(String),
  });
  expect(
    fetchMock.mock.calls.some(([url]) =>
      String(url).includes("/amendments/analyze")
    )
  ).toBe(false);
});

test.each([
  ["legislation.zip", "application/zip"],
  ["ana.md", "text/markdown"],
])(
  "uploads %s as prepared Markdown and uses the shared review flow",
  async (name, type) => {
    const fetchMock = mockApi({
      source: {
        ...sourcePackage,
        status: "processing",
        asset_count: 0,
        assets: [],
      },
      polledSources: [sourcePackage],
    });
    const user = setupUser();
    render(<NewDocumentIntake />);
    await user.click(
      await screen.findByRole("button", { name: "Markdown package" })
    );
    expect(screen.getByLabelText("Markdown package file")).toHaveAttribute(
      "accept",
      ".zip,.md"
    );
    expect(
      screen.queryByLabelText("Original document URL")
    ).not.toBeInTheDocument();
    expect(screen.getByText(/Conversion is skipped/)).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Download Markdown package template" })
    ).toHaveAttribute("href", `${base}/markdown-template`);
    const file = new File(["# Ana belge\n[EK-1](ekler/ek-1.md)"], name, {
      type,
    });
    await user.upload(screen.getByLabelText("Markdown package file"), file);
    await user.click(
      screen.getByRole("button", { name: "Prepare source and attachments" })
    );
    await screen.findByText("Source package ready · 2 documents");
    const creation = fetchMock.mock.calls.find(
      ([url]) => url === `${base}/source-packages/markdown`
    )!;
    expect(creation[1]?.method).toBe("POST");
    const upload = creation[1]!.body as FormData;
    expect(upload.get("file")).toBe(file);
    expect(upload.get("idempotency_key")).toEqual(expect.any(String));
    expect([...upload.keys()].sort()).toEqual(["file", "idempotency_key"]);
    expect(fetchMock).toHaveBeenCalledWith(
      `${sourceBase}/source-1?document_set_id=7`,
      undefined
    );
    expect(
      fetchMock.mock.calls.some(([url]) => url === `${sourceBase}/upload`)
    ).toBe(false);
    await user.click(
      screen.getByRole("button", { name: "Prepare new documents" })
    );
    expect(
      await screen.findByRole("region", { name: "EK-1.pdf" })
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Approve all new documents" })
    ).toBeEnabled();
  }
);

test("shows Markdown package validation errors without starting document preparation", async () => {
  const fetchMock = mockApi({
    prepareResponse: Promise.resolve({
      ok: false,
      status: 422,
      json: async () => ({ detail: "The ZIP is missing manifest.json." }),
    } as Response),
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", { name: "Markdown package" })
  );
  await user.upload(
    screen.getByLabelText("Markdown package file"),
    new File(["invalid package"], "invalid.zip", { type: "application/zip" })
  );
  await user.click(
    screen.getByRole("button", { name: "Prepare source and attachments" })
  );
  expect(
    await screen.findByText("The ZIP is missing manifest.json.")
  ).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Prepare new documents" })
  ).toBeDisabled();
  expect(
    fetchMock.mock.calls.some(
      ([url, init]) => url === `${base}/batches` && init?.method === "POST"
    )
  ).toBe(false);
  await user.click(screen.getByRole("button", { name: "File" }));
  expect(
    screen.getByRole("button", { name: "Prepare source and attachments" })
  ).toBeDisabled();
  expect(
    screen.queryByText("The ZIP is missing manifest.json.")
  ).not.toBeInTheDocument();
});

test("opens trusted attachment links in the same import and navigates back without changing external citations", async () => {
  const batch: DocumentIntakeBatch = {
    ...readyBatch,
    documents: readyBatch.documents.map((document) => ({
      ...document,
      document_links:
        document.id === "root"
          ? [
              {
                href: "ekler/ek\\-1.md#tablo",
                label: "EK-1",
                target_document_id: "annex",
              },
              {
                href: "https://example.gov/reference",
                label: "Official citation",
                target_document_id: "outside-this-import",
              },
            ]
          : [
              {
                href: "../ana.md",
                label: "Ana belge",
                target_document_id: "root",
              },
            ],
    })),
  };
  mockApi({
    history: [batch],
    chunksByDocument: {
      root: "# Ana belge\n\n[EK-1](ekler/%65k-1.md#tablo)\n\n[Official citation](https://example.gov/reference)",
      annex: "# Ek belge\n\nAttachment rule.\n\n[Ana belge](../ana.md)",
    },
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", { name: "Main regulation.html (ready)" })
  );
  const root = screen.getByRole("region", { name: "Main regulation.html" });
  const annex = screen.getByRole("region", { name: "EK-1.pdf" });
  expect(within(root).getByRole("link", { name: "Open EK-1" })).toHaveAttribute(
    "href",
    "#intake-document-annex"
  );
  expect(
    within(annex).getByRole("link", { name: "Back to Main regulation.html" })
  ).toHaveAttribute("href", "#intake-document-root");
  const attachmentLink = await within(root).findByRole("link", {
    name: "EK-1",
  });
  expect(attachmentLink).toHaveAttribute("href", "#intake-document-annex");
  const external = within(root).getByRole("link", {
    name: "Official citation",
  });
  expect(external).toHaveAttribute("href", "https://example.gov/reference");
  expect(external).toHaveAttribute("target", "_blank");
  expect(external).toHaveAttribute("rel", "noopener noreferrer");
  const pathname = window.location.pathname;
  await user.click(attachmentLink);
  expect(await within(annex).findByText("Attachment rule.")).toBeVisible();
  expect(annex).toHaveFocus();
  expect(window.location.pathname).toBe(pathname);
  await user.click(within(root).getByRole("button", { name: "Hide chunks" }));
  await user.click(within(annex).getByRole("link", { name: "Ana belge" }));
  expect(await within(root).findByRole("link", { name: "EK-1" })).toBeVisible();
  expect(root).toHaveFocus();
  expect(window.location.pathname).toBe(pathname);
  await user.click(within(root).getByRole("button", { name: "Hide chunks" }));
  await user.click(
    within(annex).getByRole("link", { name: "Back to Main regulation.html" })
  );
  expect(await within(root).findByRole("link", { name: "EK-1" })).toBeVisible();
  expect(root).toHaveFocus();
});

test("keeps saved import status synchronized with completion polling", async () => {
  jest.useFakeTimers();
  try {
    const approving: DocumentIntakeBatch = {
      ...readyBatch,
      status: "approving",
      stage: "indexing",
    };
    const completed: DocumentIntakeBatch = {
      ...readyBatch,
      status: "completed",
      stage: "completed",
      documents: readyBatch.documents.map((document) => ({
        ...document,
        status: "completed",
      })),
    };
    const unrelated: DocumentIntakeBatch = {
      ...readyBatch,
      id: "intake-other",
      documents: [
        { ...readyBatch.documents[0]!, display_name: "Other regulation.pdf" },
      ],
    };
    const fetchMock = mockApi({
      history: [approving, unrelated],
      polledBatches: [approving, completed],
    });
    const user = setupUser({ advanceTimers: jest.advanceTimersByTime });
    render(<NewDocumentIntake />);
    await user.click(
      await screen.findByRole("button", {
        name: "Main regulation.html (approving)",
      })
    );
    await screen.findByText("New documents: approving");
    await act(async () => {
      await jest.advanceTimersByTimeAsync(2100);
    });

    expect(await screen.findByText("New documents: completed")).toBeVisible();
    expect(
      await screen.findByRole("button", {
        name: "Main regulation.html (completed)",
      })
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Other regulation.pdf (ready)" })
    ).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Start another import" })
    );
    expect(
      screen.getByRole("button", { name: "Main regulation.html (completed)" })
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Main regulation.html (approving)" })
    ).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter(([url]) => url === `${base}/batches`)
    ).toHaveLength(1);
  } finally {
    jest.useRealTimers();
  }
});

test("shows a missing attachment and keeps document preparation disabled", async () => {
  const fetchMock = mockApi({
    source: {
      ...sourcePackage,
      status: "partial",
      issues: [
        {
          code: "linked_source_unreadable",
          locator: "https://example.gov/EK-2.pdf",
          retryable: true,
        },
      ],
    },
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await screen.findByText(/will be added to PC Külliyatı/);
  await user.click(screen.getByRole("button", { name: "URL" }));
  await user.type(
    screen.getByLabelText("New document source URL"),
    "https://example.gov/main.html"
  );
  await user.click(
    screen.getByRole("button", { name: "Prepare source and attachments" })
  );
  expect(
    await screen.findByText(/linked source unreadable.*EK-2.pdf/)
  ).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Prepare new documents" })
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Retry source preparation" })
  ).toBeVisible();
  expect(
    fetchMock.mock.calls.some(
      ([url, init]) => url === `${base}/batches` && init?.method === "POST"
    )
  ).toBe(false);
});

test("identifies the unresolved attachment and explains how to resolve relative links", async () => {
  mockApi({
    source: {
      ...sourcePackage,
      status: "partial",
      issues: [
        { code: "missing_base_url", locator: "html:0", retryable: true },
      ],
    },
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await screen.findByText(/will be added to PC Külliyatı/);
  await user.click(screen.getByRole("button", { name: "URL" }));
  await user.type(
    screen.getByLabelText("New document source URL"),
    "https://example.gov/main.html"
  );
  await user.click(
    screen.getByRole("button", { name: "Prepare source and attachments" })
  );
  expect(
    await screen.findByText(/Ekler için tıklayınız.*ekler\/ek-1.html/)
  ).toBeVisible();
  expect(
    screen.getByText(
      "Add the original document URL to resolve this attachment."
    )
  ).toBeVisible();
});

test("keeps failed labeling visible and prevents approval", async () => {
  mockApi({
    history: [
      {
        ...readyBatch,
        status: "failed",
        stage: "labeling",
        error: "Labeling has unresolved chunks.",
        documents: [
          {
            ...readyBatch.documents[0]!,
            status: "failed",
            error: "Labeling provider rejected the request.",
          },
        ],
      },
    ],
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", { name: "Main regulation.html (failed)" })
  );
  expect(
    await screen.findByText("Labeling provider rejected the request.")
  ).toBeVisible();
  expect(screen.getByText("Labeling has unresolved chunks.")).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Retry new documents" })
  ).toBeVisible();
});

test("requires a usable labeling connection before document preparation", async () => {
  const fetchMock = mockApi({
    configurationErrors: ["Gemini 3.5 Flash credentials are unavailable."],
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await screen.findByText("Gemini 3.5 Flash credentials are unavailable.");
  await user.click(screen.getByRole("button", { name: "URL" }));
  await user.type(
    screen.getByLabelText("New document source URL"),
    "https://example.gov/main.html"
  );
  await user.click(
    screen.getByRole("button", { name: "Prepare source and attachments" })
  );
  await screen.findByText("Source package ready · 2 documents");
  expect(
    screen.getByRole("button", { name: "Prepare new documents" })
  ).toBeDisabled();
  expect(
    fetchMock.mock.calls.some(
      ([url, init]) => url === `${base}/batches` && init?.method === "POST"
    )
  ).toBe(false);
});

test("ignores an old source preparation after the user changes source", async () => {
  const preparation = deferred<Response>();
  mockApi({ prepareResponse: preparation.promise });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await screen.findByText(/will be added to PC Külliyatı/);
  await user.click(screen.getByRole("button", { name: "URL" }));
  await user.type(
    screen.getByLabelText("New document source URL"),
    "https://example.gov/old.html"
  );
  await user.click(
    screen.getByRole("button", { name: "Prepare source and attachments" })
  );
  await user.clear(screen.getByLabelText("New document source URL"));
  await user.type(
    screen.getByLabelText("New document source URL"),
    "https://example.gov/new.html"
  );
  await act(async () => preparation.resolve(json(sourcePackage)));
  expect(
    screen.queryByText("Source package ready · 2 documents")
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Prepare new documents" })
  ).toBeDisabled();
});

test("shows the missing PC target explicitly without starting preparation", async () => {
  const fetchMock = mockApi({ enabled: false });
  render(<NewDocumentIntake />);
  expect(
    await screen.findByText("PC Külliyatı is not configured.")
  ).toBeVisible();
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
});

test("edits a new chunk before approval and requests labels only after saving the reviewed draft", async () => {
  const fetchMock = mockApi({
    history: [readyBatch],
    chunksByDocument: { root: "MADDE 1 — Original rule." },
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", { name: "Main regulation.html (ready)" })
  );
  const root = screen.getByRole("region", { name: "Main regulation.html" });
  const text = within(root).getByLabelText("After text");
  await user.clear(text);
  await user.type(text, "MADDE 1 — Reviewed rule.");
  await user.type(
    within(root).getByLabelText("After heading_path"),
    "BÖLÜM 1\nMadde 1"
  );
  await user.type(
    within(root).getByLabelText("After validity_start_date"),
    "2026-02-30"
  );
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  expect(
    within(root).getByRole("button", { name: "Save chunk changes" })
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Start another import" })
  ).toBeDisabled();
  expect(within(root).getByText("Use a valid YYYY-MM-DD date.")).toBeVisible();
  await user.clear(within(root).getByLabelText("After validity_start_date"));
  await user.type(
    within(root).getByLabelText("After validity_start_date"),
    "2026-10-26"
  );
  await user.click(
    within(root).getByRole("button", { name: "Save chunk changes" })
  );
  expect(await screen.findByText("New documents: preparing")).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  expect(fetchMock).toHaveBeenCalledWith(
    `${base}/batches/intake-1/documents/root/chunks/chunk-root`,
    expect.objectContaining({
      method: "PATCH",
      body: JSON.stringify({
        expected_review_sha256: readyBatch.review_sha256,
        text: "MADDE 1 — Reviewed rule.",
        heading_path: ["BÖLÜM 1", "Madde 1"],
        chunk_type: "madde",
        validity_start_date: "2026-10-26",
        validity_end_date: null,
      }),
    })
  );
  expect(
    fetchMock.mock.calls.some(([url]) => String(url).endsWith("/approve"))
  ).toBe(false);
  expect(
    fetchMock.mock.calls.some(([url]) =>
      String(url).includes("/amendments/analyze")
    )
  ).toBe(false);
});

test("keeps a failed or stale chunk edit available to correct or discard", async () => {
  mockApi({
    history: [readyBatch],
    chunksByDocument: { root: "Original rule." },
    editResponse: {
      ok: false,
      status: 409,
      json: async () => ({
        detail: "The review changed. Refresh before saving.",
      }),
    } as Response,
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", { name: "Main regulation.html (ready)" })
  );
  const root = screen.getByRole("region", { name: "Main regulation.html" });
  await user.type(within(root).getByLabelText("After text"), " Reviewed.");
  await user.click(
    within(root).getByRole("button", { name: "Save chunk changes" })
  );
  expect(
    await within(root).findByText("The review changed. Refresh before saving.")
  ).toBeVisible();
  expect(within(root).getByLabelText("After text")).toHaveValue(
    "Original rule. Reviewed."
  );
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  await user.click(
    within(root).getByRole("button", { name: "Discard chunk changes" })
  );
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeEnabled();
  expect(within(root).getByLabelText("After text")).toHaveValue(
    "Original rule."
  );
});

test("reviews predecessor suggestions at document level and requires explicit repeal evidence before saving", async () => {
  const fetchMock = mockApi({
    history: [readyBatch],
    predecessors: [
      {
        user_file_id: "old-law",
        title: "4000 sayılı Gümrük Kanunu",
        reason: "Similar legislation title",
        score: 0.9,
        status: "current",
      },
    ],
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", { name: "Main regulation.html (ready)" })
  );
  const root = screen.getByRole("region", { name: "Main regulation.html" });
  await user.click(
    within(root).getByRole("button", { name: "Review previous legislation" })
  );
  await user.click(
    await within(root).findByRole("button", {
      name: "Select 4000 sayılı Gümrük Kanunu",
    })
  );
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  expect(
    within(root).getByRole("button", {
      name: "Save previous legislation relationship",
    })
  ).toBeEnabled();
  await user.click(
    within(root).getByRole("checkbox", {
      name: "This document explicitly repeals the previous legislation",
    })
  );
  expect(
    within(root).getByRole("button", {
      name: "Save previous legislation relationship",
    })
  ).toBeDisabled();
  await user.type(
    within(root).getByLabelText("Repeal effective date"),
    "2026-10-26"
  );
  await user.type(
    within(root).getByLabelText("Repeal provision"),
    "4000 sayılı Kanun yürürlükten kaldırılmıştır."
  );
  await user.click(
    within(root).getByRole("button", {
      name: "Save previous legislation relationship",
    })
  );
  expect(await within(root).findByText("Pending publication")).toBeVisible();
  expect(
    within(root).getByText(
      "The previous document changes only after approval and successful publication."
    )
  ).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeEnabled();
  expect(fetchMock).toHaveBeenCalledWith(
    `${base}/batches/intake-1/documents/root/predecessor-selection`,
    expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({
        expected_review_sha256: readyBatch.review_sha256,
        predecessor_user_file_id: "old-law",
        relation: "repeals",
        effective_date: "2026-10-26",
        evidence: "4000 sayılı Kanun yürürlükten kaldırılmıştır.",
      }),
    })
  );
  expect(
    fetchMock.mock.calls.some(([url]) =>
      String(url).includes("/amendments/analyze")
    )
  ).toBe(false);
  expect(
    fetchMock.mock.calls.some(([url]) => String(url).endsWith("/approve"))
  ).toBe(false);
  await user.click(
    screen.getByRole("button", { name: "Approve all new documents" })
  );
  expect(fetchMock).toHaveBeenCalledWith(
    `${base}/batches/intake-1/approve`,
    expect.objectContaining({
      body: JSON.stringify({ expected_review_sha256: "a".repeat(64) }),
    })
  );
});

test("offers a relationship-only retry after publication is blocked by a removed predecessor", async () => {
  const blocked: DocumentIntakeBatch = {
    ...readyBatch,
    status: "blocked",
    stage: "lineage",
    review_sha256: null,
    error: "The previous legislation is no longer in PC Külliyatı.",
  };
  mockApi({ history: [blocked] });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", {
      name: "Main regulation.html (blocked)",
    })
  );
  expect(
    screen.getByRole("button", { name: "Retry document relationship" })
  ).toBeVisible();
  expect(
    screen.getByText(
      "Restore the previous file to PC Külliyatı, then retry the document relationship. Published chunks and labels will be kept."
    )
  ).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Retry new documents" })
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
});

test("shows added chunks immediately in bounded pages and preserves unsaved edit guards", async () => {
  const batch: DocumentIntakeBatch = {
    ...readyBatch,
    documents: readyBatch.documents.map((document) =>
      document.id === "root" ? { ...document, chunk_count: 11 } : document
    ),
  };
  const fetchMock = mockApi({ history: [batch] });
  const original = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation(async (input, init) => {
    const url = String(input);
    if (url.includes("/documents/root/chunks?")) {
      const offset = Number(
        new URL(url, "http://localhost").searchParams.get("offset")
      );
      return json({
        items: Array.from({ length: offset === 0 ? 10 : 1 }, (_, index) => ({
          id: `root-chunk-${offset + index}`,
          position: offset + index,
          text: `Added regulation rule ${offset + index + 1}.`,
          heading_path: [`MADDE ${offset + index + 1}`],
          chunk_type: "madde",
          validity_start_date: null,
          validity_end_date: null,
          chunk_metadata: { labels: ["customs"] },
        })),
        total: 11,
        offset,
        limit: 10,
      });
    }
    return original(input, init);
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", { name: "Main regulation.html (ready)" })
  );
  const root = screen.getByRole("region", { name: "Main regulation.html" });
  expect(
    await within(root).findByDisplayValue("Added regulation rule 1.")
  ).toBeVisible();
  expect(
    within(root).getAllByRole("region", { name: /^Added chunk/ })
  ).toHaveLength(10);
  expect(screen.getByText("2 new files · 12 added chunks")).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Prepare source and attachments" })
  ).not.toBeInTheDocument();
  const first = within(root).getByRole("region", { name: "Added chunk 1" });
  await user.type(within(first).getByLabelText("After text"), " Reviewed.");
  expect(
    within(root).getByRole("button", { name: "Next chunks" })
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  await user.click(
    within(first).getByRole("button", { name: "Discard chunk changes" })
  );
  await user.click(within(root).getByRole("button", { name: "Next chunks" }));
  expect(
    await within(root).findByDisplayValue("Added regulation rule 11.")
  ).toBeVisible();
  expect(
    within(root).getAllByRole("region", { name: /^Added chunk/ })
  ).toHaveLength(1);
  expect(within(root).getByText("11–11 of 11")).toBeVisible();
  expect(
    within(root).getByRole("button", { name: "Next chunks" })
  ).toBeDisabled();
  expect(fetchMock).toHaveBeenCalledWith(
    `${base}/batches/intake-1/documents/root/chunks?offset=10&limit=10`,
    undefined
  );
});

test("allows manual repair with a terminal partial edit receipt while approval stays disabled", async () => {
  const partial: DocumentIntakeBatch = {
    ...readyBatch,
    status: "partial",
    stage: "labeling",
    review_sha256: null,
    edit_review_sha256: "c".repeat(64),
    error: "One chunk needs labeling review.",
  };
  const fetchMock = mockApi({
    history: [partial],
    chunksByDocument: { root: "Original rule." },
  });
  const user = setupUser();
  render(<NewDocumentIntake />);
  await user.click(
    await screen.findByRole("button", {
      name: "Main regulation.html (partial)",
    })
  );
  const root = screen.getByRole("region", { name: "Main regulation.html" });
  const text = await within(root).findByLabelText("After text");
  expect(text).not.toHaveAttribute("readonly");
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  await user.type(text, " Reviewed.");
  await user.type(
    within(root).getByLabelText("After metadata article_no"),
    "7"
  );
  await user.click(
    within(root).getByRole("button", { name: "Save chunk changes" })
  );
  expect(await screen.findByText("New documents: preparing")).toBeVisible();
  const request = fetchMock.mock.calls.find(
    ([url, init]) =>
      String(url).endsWith("/chunks/chunk-root") && init?.method === "PATCH"
  )!;
  expect(JSON.parse(String(request[1]!.body))).toEqual(
    expect.objectContaining({
      expected_review_sha256: "c".repeat(64),
      text: "Original rule. Reviewed.",
      chunk_metadata: { article_no: "7" },
    })
  );
  expect(within(root).getByLabelText("After text")).toHaveAttribute("readonly");
  expect(
    screen.getByRole("button", { name: "Approve all new documents" })
  ).toBeDisabled();
  expect(
    fetchMock.mock.calls.some(([url]) => String(url).endsWith("/approve"))
  ).toBe(false);
});

test.each([
  ["partial", null],
  ["preparing", "c".repeat(64)],
] as const)(
  "keeps %s read-only without a usable terminal edit review",
  async (status, receipt) => {
    const batch: DocumentIntakeBatch = {
      ...readyBatch,
      status,
      stage: "labeling",
      edit_review_sha256: receipt,
    };
    mockApi({
      history: [batch],
      chunksByDocument: { root: "Unresolved rule." },
    });
    const user = setupUser();
    render(<NewDocumentIntake />);
    await user.click(
      await screen.findByRole("button", {
        name: `Main regulation.html (${status})`,
      })
    );
    const root = screen.getByRole("region", { name: "Main regulation.html" });
    expect(await within(root).findByLabelText("After text")).toHaveAttribute(
      "readonly"
    );
    expect(
      within(root).getByLabelText("After metadata article_no")
    ).toHaveAttribute("readonly");
    expect(
      within(root).queryByRole("button", { name: "Save chunk changes" })
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Approve all new documents" })
    ).toBeDisabled();
  }
);
