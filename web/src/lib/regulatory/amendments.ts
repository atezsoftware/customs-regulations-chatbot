export interface AmendmentBatch {
  id: number;
  document_set_id: number;
  raw_text: string;
  reference_date: string | null;
  status: "queued" | "analyzing" | "analyzed" | "failed";
  stage: "queued" | "segmenting" | "processing" | "finalizing";
  instruction_count: number;
  processed_instruction_count: number;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  heartbeat_at: string | null;
  completed_at: string | null;
}

export interface AmendmentProposal {
  id: number;
  batch_id: number;
  instruction_index: number;
  instruction_text: string;
  instruction_indices: number[];
  instruction_texts: string[];
  old_chunk_id: string | null;
  old_chunk_snapshot: Record<string, unknown>;
  new_chunk_draft: Record<string, unknown>;
  match_confidence: number | null;
  match_rationale: string | null;
  date_rationale: string | null;
  status: "pending" | "approving" | "approval_failed" | "approved" | "rejected";
  applied_new_chunk_id: string | null;
  approval_indexing_job_id?: string | null;
  approval_error?: string | null;
  decided_by: string | null;
  decided_at: string | null;
  created_at: string;
  updated_at: string;
  duplicate_target: boolean;
}

export interface AnalyzeAmendmentResponse {
  batch: AmendmentBatch;
  proposals: AmendmentProposal[];
  unmatched_instructions: string[];
}

export interface AmendmentSourceExtraction {
  text: string;
  source_type: "html" | "pdf" | "docx";
  display_name: string;
}

export class RegulatoryRequestError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message);
    this.name = "RegulatoryRequestError";
  }
}

const handleRequestError = (action: string, response: Response): never => {
  throw new RegulatoryRequestError(
    `${action} failed (Status: ${response.status})`,
    response.status
  );
};

export async function analyzeAmendment(
  documentSetId: number,
  rawText: string
): Promise<AmendmentBatch> {
  const response = await fetch("/api/regulatory/amendments/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_set_id: documentSetId,
      raw_text: rawText,
    }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Amendment analysis failed (Status: ${response.status})`
    );
  }
  return response.json();
}

export async function getAmendmentAnalysis(
  batchId: number
): Promise<AnalyzeAmendmentResponse> {
  const response = await fetch(
    `/api/regulatory/amendments/batches/${batchId}/analysis`
  );
  if (!response.ok) {
    handleRequestError("Get amendment analysis", response);
  }
  return response.json();
}

export async function retryAmendmentBatch(
  batchId: number
): Promise<AmendmentBatch> {
  const response = await fetch(
    `/api/regulatory/amendments/batches/${batchId}/retry`,
    { method: "POST" }
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Retry analysis failed (Status: ${response.status})`
    );
  }
  return response.json();
}

export async function extractAmendmentUrl(
  url: string
): Promise<AmendmentSourceExtraction> {
  const response = await fetch("/api/regulatory/amendments/sources/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `URL extraction failed (Status: ${response.status})`
    );
  }
  return response.json();
}

export async function extractAmendmentPdf(
  file: File
): Promise<AmendmentSourceExtraction> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/regulatory/amendments/sources/pdf", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `PDF extraction failed (Status: ${response.status})`
    );
  }
  return response.json();
}

export async function extractAmendmentDocx(
  file: File
): Promise<AmendmentSourceExtraction> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/regulatory/amendments/sources/docx", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Word extraction failed (Status: ${response.status})`
    );
  }
  return response.json();
}

export async function listAmendmentBatches(
  documentSetId: number
): Promise<AmendmentBatch[]> {
  const response = await fetch(
    `/api/regulatory/amendments/batches?document_set_id=${documentSetId}`
  );
  if (!response.ok) {
    handleRequestError("List amendment batches", response);
  }
  return response.json();
}

export async function listAmendmentProposals(
  batchId: number
): Promise<AmendmentProposal[]> {
  const response = await fetch(
    `/api/regulatory/amendments/batches/${batchId}/proposals`
  );
  if (!response.ok) {
    handleRequestError("List amendment proposals", response);
  }
  return response.json();
}

export async function approveProposal(
  proposalId: number,
  newChunkDraft: Record<string, unknown>
): Promise<AmendmentProposal> {
  const response = await fetch(
    `/api/regulatory/amendments/proposals/${proposalId}/approve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_chunk_draft: newChunkDraft }),
    }
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Approve failed (Status: ${response.status})`
    );
  }
  return response.json();
}

export async function rejectProposal(
  proposalId: number
): Promise<AmendmentProposal> {
  const response = await fetch(
    `/api/regulatory/amendments/proposals/${proposalId}/reject`,
    { method: "POST" }
  );
  if (!response.ok) {
    handleRequestError("Reject proposal", response);
  }
  return response.json();
}
