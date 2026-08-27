import { act, render, screen, setupUser } from "@tests/setup/test-utils";

import {
  analyzeAmendment,
  extractAmendmentPdf,
  extractAmendmentUrl,
  getAmendmentAnalysis,
  listAmendmentBatches,
  retryAmendmentBatch,
} from "@/lib/regulatory/amendments";
import AmendmentsPage from "@/views/admin/AmendmentsPage";

let mockDocumentSets: Array<{ id: number; name: string }> = [];

jest.mock("next/navigation", () => ({
  useRouter: () => ({ back: jest.fn(), push: jest.fn(), replace: jest.fn() }),
}));

jest.mock("@/lib/hooks/useDocumentSets", () => ({
  useDocumentSets: () => ({ documentSets: mockDocumentSets }),
}));

jest.mock("@/lib/regulatory/amendments", () => ({
  analyzeAmendment: jest.fn(),
  approveProposal: jest.fn(),
  extractAmendmentPdf: jest.fn(),
  extractAmendmentUrl: jest.fn(),
  getAmendmentAnalysis: jest.fn(),
  listAmendmentBatches: jest.fn(),
  listAmendmentProposals: jest.fn(),
  rejectProposal: jest.fn(),
  retryAmendmentBatch: jest.fn(),
}));

const queuedBatch = {
  id: 42,
  document_set_id: 7,
  raw_text: "MADDE 1",
  reference_date: null,
  status: "queued" as const,
  stage: "queued" as const,
  instruction_count: 0,
  processed_instruction_count: 0,
  error_message: null,
  created_by: null,
  created_at: "2026-08-27T12:00:00Z",
  updated_at: "2026-08-27T12:00:00Z",
  started_at: null,
  heartbeat_at: null,
  completed_at: null,
};

const mockedAnalyzeAmendment = analyzeAmendment as jest.MockedFunction<
  typeof analyzeAmendment
>;
const mockedGetAmendmentAnalysis = getAmendmentAnalysis as jest.MockedFunction<
  typeof getAmendmentAnalysis
>;

const mockedExtractAmendmentUrl = extractAmendmentUrl as jest.MockedFunction<
  typeof extractAmendmentUrl
>;
const mockedExtractAmendmentPdf = extractAmendmentPdf as jest.MockedFunction<
  typeof extractAmendmentPdf
>;
const mockedListAmendmentBatches = listAmendmentBatches as jest.MockedFunction<
  typeof listAmendmentBatches
>;
const mockedRetryAmendmentBatch = retryAmendmentBatch as jest.MockedFunction<
  typeof retryAmendmentBatch
>;

beforeAll(() => {
  window.HTMLElement.prototype.scrollIntoView = jest.fn();
});

beforeEach(() => {
  mockDocumentSets = [{ id: 7, name: "Transit rules" }];
  mockedListAmendmentBatches.mockReset();
  mockedListAmendmentBatches.mockResolvedValue([]);
  mockedExtractAmendmentUrl.mockReset();
  mockedExtractAmendmentUrl.mockResolvedValue({
    text: "MADDE 1- Yeni metin.",
    source_type: "html",
    display_name: "20260826-2.htm",
  });
  mockedExtractAmendmentPdf.mockReset();
  mockedExtractAmendmentPdf.mockResolvedValue({
    text: "MADDE 2- PDF metni.",
    source_type: "pdf",
    display_name: "amendment.pdf",
  });
  mockedAnalyzeAmendment.mockReset();
  mockedAnalyzeAmendment.mockResolvedValue(queuedBatch);
  mockedGetAmendmentAnalysis.mockReset();
  mockedGetAmendmentAnalysis.mockResolvedValue({
    batch: {
      ...queuedBatch,
      status: "analyzing",
      stage: "processing",
      instruction_count: 5,
      processed_instruction_count: 2,
    },
    proposals: [],
    unmatched_instructions: [],
  });
  mockedRetryAmendmentBatch.mockReset();
  mockedRetryAmendmentBatch.mockResolvedValue(queuedBatch);
});

test("places extracted URL text in the editable amendment text area", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");

  await user.click(screen.getByRole("button", { name: "URL" }));
  await user.type(
    screen.getByRole("textbox", { name: "Amendment source URL" }),
    "https://example.gov/20260826-2.htm"
  );
  await user.click(screen.getByRole("button", { name: "Extract" }));

  expect(await screen.findByDisplayValue("MADDE 1- Yeni metin.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
});

test("places extracted PDF text in the editable amendment text area", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");

  await user.click(screen.getByRole("button", { name: "PDF" }));
  const file = new File(["pdf contents"], "amendment.pdf", {
    type: "application/pdf",
  });
  await user.upload(screen.getByLabelText("Amendment source PDF"), file);
  await user.click(screen.getByRole("button", { name: "Extract" }));

  expect(await screen.findByDisplayValue("MADDE 2- PDF metni.")).toBeVisible();
  expect(mockedExtractAmendmentPdf).toHaveBeenCalledWith(file);
  expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
});

test("queues pasted text and polls durable progress", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.type(screen.getByPlaceholderText(/Paste the official/), "MADDE 1");
  await user.click(screen.getByRole("button", { name: "Analyze" }));

  expect(mockedAnalyzeAmendment).toHaveBeenCalledWith(7, "MADDE 1");
  expect(mockedGetAmendmentAnalysis).toHaveBeenCalledWith(42);
  expect(await screen.findByText("Analyzing 2 / 5")).toBeVisible();
});

test("queues PDF-extracted text through the same durable analysis", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.click(screen.getByRole("button", { name: "PDF" }));
  await user.upload(
    screen.getByLabelText("Amendment source PDF"),
    new File(["pdf"], "amendment.pdf", { type: "application/pdf" })
  );
  await user.click(screen.getByRole("button", { name: "Extract" }));
  await user.click(screen.getByRole("button", { name: "Analyze" }));

  expect(mockedAnalyzeAmendment).toHaveBeenCalledWith(7, "MADDE 2- PDF metni.");
});

test("renders all grouped instructions on one proposal card", async () => {
  const analyzedBatch = {
    ...queuedBatch,
    status: "analyzed" as const,
    stage: "finalizing" as const,
    instruction_count: 2,
    processed_instruction_count: 2,
  };
  mockedListAmendmentBatches.mockResolvedValue([analyzedBatch]);
  mockedGetAmendmentAnalysis.mockResolvedValue({
    batch: analyzedBatch,
    proposals: [
      {
        id: 19,
        batch_id: 42,
        instruction_index: 0,
        instruction_text: "Replace Article 1.",
        instruction_indices: [0, 1],
        instruction_texts: [
          "Replace Article 1.",
          "Add the Article 1 exception.",
        ],
        old_chunk_id: "chunk-1",
        old_chunk_snapshot: { text: "Current Article 1." },
        new_chunk_draft: { text: "Updated Article 1." },
        match_confidence: 0.9,
        match_rationale: "Same target",
        date_rationale: null,
        status: "pending" as const,
        applied_new_chunk_id: null,
        decided_by: null,
        decided_at: null,
        created_at: "2026-08-27T12:00:00Z",
        updated_at: "2026-08-27T12:00:00Z",
        duplicate_target: false,
      },
    ],
    unmatched_instructions: [],
  });

  const user = setupUser();
  render(<AmendmentsPage />);
  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.click(
    await screen.findByRole("button", { name: "Batch #42 (analyzed)" })
  );

  expect(await screen.findByText("Replace Article 1.")).toBeVisible();
  expect(screen.getByText("Add the Article 1 exception.")).toBeVisible();
  expect(
    screen.getAllByRole("article", { name: "Amendment proposal" })
  ).toHaveLength(1);
});

test("retries a failed batch from its checkpoint", async () => {
  const failedBatch = {
    ...queuedBatch,
    status: "failed" as const,
    stage: "processing" as const,
    instruction_count: 5,
    processed_instruction_count: 2,
    error_message: "provider timeout",
  };
  mockedListAmendmentBatches.mockResolvedValue([failedBatch]);
  mockedGetAmendmentAnalysis
    .mockResolvedValueOnce({
      batch: failedBatch,
      proposals: [],
      unmatched_instructions: [],
    })
    .mockResolvedValue({
      batch: {
        ...queuedBatch,
        instruction_count: 5,
        processed_instruction_count: 2,
      },
      proposals: [],
      unmatched_instructions: [],
    });

  const user = setupUser();
  render(<AmendmentsPage />);
  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.click(await screen.findByRole("button", { name: /Batch #42/ }));
  await user.click(await screen.findByRole("button", { name: "Retry" }));

  expect(mockedRetryAmendmentBatch).toHaveBeenCalledWith(42);
  expect(mockedGetAmendmentAnalysis).toHaveBeenCalledTimes(2);
});
