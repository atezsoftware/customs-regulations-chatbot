import {
  act,
  fireEvent,
  render,
  screen,
  setupUser,
} from "@tests/setup/test-utils";

import {
  analyzeAmendment,
  approveProposal,
  extractAmendmentDocx,
  extractAmendmentPdf,
  extractAmendmentUrl,
  getAmendmentAnalysis,
  listAmendmentBatches,
  listAmendmentProposals,
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
  extractAmendmentDocx: jest.fn(),
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
const mockedApproveProposal = approveProposal as jest.MockedFunction<
  typeof approveProposal
>;
const mockedGetAmendmentAnalysis = getAmendmentAnalysis as jest.MockedFunction<
  typeof getAmendmentAnalysis
>;

const mockedExtractAmendmentUrl = extractAmendmentUrl as jest.MockedFunction<
  typeof extractAmendmentUrl
>;
const mockedExtractAmendmentDocx = extractAmendmentDocx as jest.MockedFunction<
  typeof extractAmendmentDocx
>;
const mockedExtractAmendmentPdf = extractAmendmentPdf as jest.MockedFunction<
  typeof extractAmendmentPdf
>;
const mockedListAmendmentBatches = listAmendmentBatches as jest.MockedFunction<
  typeof listAmendmentBatches
>;
const mockedListAmendmentProposals =
  listAmendmentProposals as jest.MockedFunction<typeof listAmendmentProposals>;
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
  mockedListAmendmentProposals.mockReset();
  mockedListAmendmentProposals.mockResolvedValue([]);
  mockedExtractAmendmentUrl.mockReset();
  mockedExtractAmendmentUrl.mockResolvedValue({
    text: "MADDE 1- Yeni metin.",
    source_type: "html",
    display_name: "20260826-2.htm",
  });
  mockedExtractAmendmentDocx.mockReset();
  mockedExtractAmendmentDocx.mockResolvedValue({
    text: "MADDE 3- Word metni.",
    source_type: "docx",
    display_name: "değişiklik.docx",
  });
  mockedExtractAmendmentPdf.mockReset();
  mockedExtractAmendmentPdf.mockResolvedValue({
    text: "MADDE 2- PDF metni.",
    source_type: "pdf",
    display_name: "amendment.pdf",
  });
  mockedAnalyzeAmendment.mockReset();
  mockedAnalyzeAmendment.mockResolvedValue(queuedBatch);
  mockedApproveProposal.mockReset();
  mockedApproveProposal.mockResolvedValue({} as never);
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

test("queues DOCX-extracted text through the same durable analysis", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.click(screen.getByRole("button", { name: "Word (.docx)" }));
  await user.upload(
    screen.getByLabelText("Amendment source Word document"),
    new File(["docx"], "değişiklik.docx", {
      type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    })
  );
  await user.click(screen.getByRole("button", { name: "Extract" }));

  expect(await screen.findByDisplayValue("MADDE 3- Word metni.")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Analyze" }));
  expect(mockedAnalyzeAmendment).toHaveBeenCalledWith(
    7,
    "MADDE 3- Word metni."
  );
});

test("clears the selected file when switching between DOCX and PDF", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");
  await user.click(screen.getByRole("button", { name: "Word (.docx)" }));
  await user.upload(
    screen.getByLabelText("Amendment source Word document"),
    new File(["docx"], "değişiklik.docx", {
      type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    })
  );
  expect(screen.getByText("değişiklik.docx")).toBeVisible();

  await user.click(screen.getByRole("button", { name: "PDF" }));

  expect(screen.queryByText("değişiklik.docx")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Extract" })).toBeDisabled();
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

test("shows before and after field tables and approves edits without exposing JSON", async () => {
  const analyzedBatch = {
    ...queuedBatch,
    status: "analyzed" as const,
    stage: "finalizing" as const,
    instruction_count: 1,
    processed_instruction_count: 1,
  };
  const proposedDraft = {
    user_file_id: "00000000-0000-0000-0000-000000000123",
    position: 15,
    text: "MADDE 15 - (2) a)",
    chunk_type: "article",
    heading_path: ["TIR İşlemleri", "MADDE 15"],
    metadata: { article_no: "15" },
    effective_start_date: null,
    effective_end_date: null,
  };
  const pendingProposal = {
    id: 19,
    batch_id: 42,
    instruction_index: 0,
    instruction_text: "(a) bendi yürürlükten kaldırılmıştır.",
    instruction_indices: [0],
    instruction_texts: ["(a) bendi yürürlükten kaldırılmıştır."],
    old_chunk_id: "old-chunk",
    old_chunk_snapshot: {
      id: "old-chunk",
      user_file_id: proposedDraft.user_file_id,
      position: 15,
      text: "MADDE 15 - (2) a)",
      chunk_type: "article",
      heading_path: proposedDraft.heading_path,
      metadata: proposedDraft.metadata,
      validity_start_date: "2010-12-31",
      validity_end_date: null,
      status: "active",
      source: "indexed",
      supersedes_chunk_id: null,
      superseded_by_chunk_id: null,
    },
    new_chunk_draft: proposedDraft,
    match_confidence: 0.99,
    match_rationale: "Exact provision",
    date_rationale: "Publication date was not provided.",
    status: "pending" as const,
    applied_new_chunk_id: null,
    decided_by: null,
    decided_at: null,
    created_at: "2026-08-27T12:00:00Z",
    updated_at: "2026-08-27T12:00:00Z",
    duplicate_target: false,
  };
  mockedListAmendmentBatches.mockResolvedValue([analyzedBatch]);
  mockedGetAmendmentAnalysis.mockResolvedValue({
    batch: analyzedBatch,
    proposals: [pendingProposal],
    unmatched_instructions: [],
  });
  mockedApproveProposal.mockResolvedValue({
    ...pendingProposal,
    status: "approving",
    new_chunk_draft: proposedDraft,
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

  expect(await screen.findByText("Before")).toBeVisible();
  expect(screen.getByText("After")).toBeVisible();
  expect(screen.getByText("2010-12-31")).toBeVisible();
  expect(screen.getAllByText("validity_end_date")).toHaveLength(2);
  expect(screen.queryByText("superseded_by_chunk_id")).not.toBeInTheDocument();
  expect(
    screen.queryByRole("textbox", { name: "Proposed chunk JSON" })
  ).not.toBeInTheDocument();

  const reviewedDraft = {
    ...proposedDraft,
    text: "MADDE 15 - (2) a) (Mülga)",
    effective_start_date: "2026-08-28",
    metadata: { article_no: "15/a" },
  };
  fireEvent.change(screen.getByRole("textbox", { name: "After text" }), {
    target: { value: reviewedDraft.text },
  });
  fireEvent.change(
    screen.getByRole("textbox", { name: "After validity_start_date" }),
    {
      target: { value: reviewedDraft.effective_start_date },
    }
  );
  fireEvent.change(
    screen.getByRole("textbox", { name: "After metadata article_no" }),
    {
      target: { value: "15/a" },
    }
  );
  expect(
    screen.getByRole("textbox", { name: "After user_file_id" })
  ).toHaveAttribute("readonly");
  expect(
    screen.getByRole("textbox", { name: "After position" })
  ).toHaveAttribute("readonly");
  expect(screen.getAllByText("Generated on approval")).not.toHaveLength(0);
  expect(screen.getByText("amendment")).toBeVisible();

  await user.click(screen.getByRole("button", { name: "Approve" }));

  expect(mockedApproveProposal).toHaveBeenCalledWith(19, reviewedDraft);
  expect(
    await screen.findByText(/Approval is running.*indexed in the background/)
  ).toBeVisible();
});

test("shows a green success message after approval and indexing complete", async () => {
  const analyzedBatch = {
    ...queuedBatch,
    status: "analyzed" as const,
    stage: "finalizing" as const,
    instruction_count: 1,
    processed_instruction_count: 1,
  };
  mockedListAmendmentBatches.mockResolvedValue([analyzedBatch]);
  mockedGetAmendmentAnalysis.mockResolvedValue({
    batch: analyzedBatch,
    proposals: [
      {
        id: 21,
        batch_id: 42,
        instruction_index: 0,
        instruction_text: "MADDE 15 değiştirilmiştir.",
        instruction_indices: [0],
        instruction_texts: ["MADDE 15 değiştirilmiştir."],
        old_chunk_id: "old-chunk",
        old_chunk_snapshot: { id: "old-chunk", text: "Eski metin" },
        new_chunk_draft: {
          user_file_id: "00000000-0000-0000-0000-000000000123",
          position: 15,
          text: "Yeni metin",
          chunk_type: "article",
          heading_path: [],
          metadata: {},
          effective_start_date: "2026-08-31",
          effective_end_date: null,
        },
        match_confidence: 0.99,
        match_rationale: "Exact provision",
        date_rationale: null,
        status: "approved" as const,
        applied_new_chunk_id: "new-chunk",
        decided_by: "admin-id",
        decided_at: "2026-08-31T12:00:00Z",
        created_at: "2026-08-31T11:00:00Z",
        updated_at: "2026-08-31T12:00:00Z",
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

  const success = await screen.findByRole("status");
  expect(success).toHaveTextContent(
    "Success — this proposal was approved and indexed."
  );
  expect(success).toHaveClass("bg-status-success-01");
});

test("shows a terminal indexing error instead of an endless running state", async () => {
  const analyzedBatch = {
    ...queuedBatch,
    status: "analyzed" as const,
    stage: "finalizing" as const,
    instruction_count: 1,
    processed_instruction_count: 1,
  };
  mockedListAmendmentBatches.mockResolvedValue([analyzedBatch]);
  mockedGetAmendmentAnalysis.mockResolvedValue({
    batch: analyzedBatch,
    proposals: [
      {
        id: 22,
        batch_id: 42,
        instruction_index: 0,
        instruction_text: "MADDE 15 değiştirilmiştir.",
        instruction_indices: [0],
        instruction_texts: ["MADDE 15 değiştirilmiştir."],
        old_chunk_id: "old-chunk",
        old_chunk_snapshot: { id: "old-chunk", text: "Eski metin" },
        new_chunk_draft: {
          user_file_id: "00000000-0000-0000-0000-000000000123",
          position: 15,
          text: "Yeni metin",
        },
        match_confidence: 0.99,
        match_rationale: "Exact provision",
        date_rationale: null,
        status: "approval_failed" as const,
        applied_new_chunk_id: "new-chunk",
        approval_indexing_job_id: "job-id",
        approval_error: "Indexing failed. The approval was not published.",
        decided_by: "admin-id",
        decided_at: null,
        created_at: "2026-08-31T11:00:00Z",
        updated_at: "2026-08-31T12:00:00Z",
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

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Indexing failed. The approval was not published."
  );
  expect(
    screen.queryByText(/Approval is running.*indexed in the background/)
  ).not.toBeInTheDocument();
});

test("keeps an invalid date edit from being approved", async () => {
  const analyzedBatch = {
    ...queuedBatch,
    status: "analyzed" as const,
    stage: "finalizing" as const,
  };
  mockedListAmendmentBatches.mockResolvedValue([analyzedBatch]);
  mockedGetAmendmentAnalysis.mockResolvedValue({
    batch: analyzedBatch,
    proposals: [
      {
        id: 20,
        batch_id: 42,
        instruction_index: 0,
        instruction_text: "Geçici değişiklik.",
        instruction_indices: [0],
        instruction_texts: ["Geçici değişiklik."],
        old_chunk_id: "old-chunk",
        old_chunk_snapshot: { id: "old-chunk", text: "Eski metin" },
        new_chunk_draft: {
          user_file_id: "00000000-0000-0000-0000-000000000123",
          position: 15,
          text: "Yeni metin",
          chunk_type: "article",
          heading_path: [],
          metadata: {},
          effective_start_date: null,
          effective_end_date: null,
        },
        match_confidence: 0.9,
        match_rationale: "Exact provision",
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

  fireEvent.change(
    await screen.findByRole("textbox", { name: "After validity_start_date" }),
    { target: { value: "not-a-date" } }
  );

  expect(screen.getByText("Use a YYYY-MM-DD date.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
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
