export interface AmendmentBatch {
  source_package_id?: string | null;
  source_text_sha256?: string | null;
  source_parent_batch_id?: number | null;
  superseded_by_batch_id?: number | null;
  annex_group_count?: number;
  annex_review_revision_count?: number;
  annex_pending_count?: number;
  id: number;
  document_set_id: number;
  raw_text: string;
  reference_date: string | null;
  status: "queued" | "analyzing" | "analyzed" | "failed" | "paused";
  stage:
    | "queued"
    | "segmenting"
    | "processing"
    | "finalizing"
    | "waiting_resources";
  instruction_count: number;
  processed_instruction_count: number;
  matched_instruction_count?: number;
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
  chunk_changes?: AmendmentProposalChunkChange[];
  match_confidence: number | null;
  match_rationale: string | null;
  date_rationale: string | null;
  status: "pending" | "approving" | "approval_failed" | "approved" | "rejected";
  applied_new_chunk_id: string | null;
  applied_new_chunk_ids?: string[];
  approval_indexing_job_id?: string | null;
  approval_error?: string | null;
  approval_execution?: {
    schema_version: "approval-execution-v1";
    stage: "baseline" | "review" | "context" | "publication";
    state: "running" | "failed";
    code: string | null;
    retryable: boolean;
    attempt: number;
    message: string;
    manifest_sha256: string | null;
  } | null;
  decided_by: string | null;
  decided_at: string | null;
  created_at: string;
  updated_at: string;
  duplicate_target: boolean;
}

export interface AmendmentProposalChunkChange {
  old_chunk_id: string | null;
  old_chunk_snapshot: Record<string, unknown>;
  new_chunk_draft: Record<string, unknown>;
  instruction_indices: number[];
  instruction_texts: string[];
  match_confidence: number | null;
  match_rationale: string | null;
  date_rationale: string | null;
}

export interface AmendmentAnalysisLogEntry {
  at: string;
  step: string;
  [field: string]: unknown;
}

export interface AnalyzeAmendmentResponse {
  annex_groups?: AnnexReview[];
  batch: AmendmentBatch;
  proposals: AmendmentProposal[];
  unmatched_instructions: string[];
  analysis_log?: AmendmentAnalysisLogEntry[];
}

export interface AmendmentSourceExtraction {
  text: string;
  source_type: "html" | "pdf" | "docx";
  display_name: string;
}

export interface AnnexCapabilities {
  enabled: boolean;
  grouped_review: boolean;
  immutable_review_revisions: boolean;
  asynchronous_source_preparation: boolean;
  publication_requires_verified_index: boolean;
}

export type AmendmentSourcePackageStatus =
  | "processing"
  | "ready"
  | "partial"
  | "blocked"
  | "failed";

export interface AmendmentSourceIssue {
  code: string;
  locator?: string | null;
  retryable?: boolean;
  failure_detail?: string | null;
}

export interface AmendmentSourceAsset {
  id: string;
  sha256: string;
  mime_type: string;
  display_name: string;
  byte_count: number;
  original_url: string | null;
  final_url: string | null;
  text_sha256: string | null;
}

export interface AmendmentSourcePackage {
  id: string;
  document_set_id: number;
  status: AmendmentSourcePackageStatus;
  asset_count: number;
  total_bytes: number;
  issues: AmendmentSourceIssue[];
  manifest_sha256: string | null;
  assets: AmendmentSourceAsset[];
  created_at: string;
  updated_at: string;
}

export interface AnnexLocator {
  page: number | null;
  sheet: string | null;
  cell: string | null;
  row: number | null;
  column: number | null;
  path: string | null;
  original_box: [number, number, number, number] | null;
  normalized_box: [number, number, number, number] | null;
  original_width: number | null;
  original_height: number | null;
  coordinate_system:
    | "top_left_pixels"
    | "top_left_points"
    | "pdf_user_space"
    | null;
}

export interface AnnexElementReference {
  position: number;
  text: string;
  locator: AnnexLocator;
}

export interface AnnexDifference {
  operation:
    | "replace"
    | "insert"
    | "remove"
    | "move"
    | "split"
    | "merge"
    | "visual";
  old: AnnexElementReference[];
  new: AnnexElementReference[];
  explanation: string;
  uncertain: boolean;
}

export interface AnnexExtractionElement {
  extraction_method: "native" | "vision" | "canonical" | "unknown";
  canonical_chunk_id: string | null;
  bound_to_regulatory_chunk_id: string | null;
  canonical_role: "authoritative" | "supporting";
  kind: "text" | "table_row" | "table_cell" | "footnote" | "image_region";
  text: string;
  semantic_key: string | null;
  locator: AnnexLocator;
  formula: string | null;
  value: string | number | boolean | null;
  evidence_kind: "original" | "rendered_preview" | "canonical";
  status: "readable" | "uncertain" | "unreadable";
  issues: string[];
  aggregate: boolean;
  image_file_id: string | null;
  source_asset_id: string | null;
}

export interface AnnexEvidenceView {
  sha256: string;
  label: string;
  parents: Array<{
    file_id: string;
    sha256: string;
    mime_type: string;
    extraction_sha256: string;
    element_count: number;
    page_count: number | null;
    canonical_chunk_ids: string[];
  }>;
  pages: Array<{
    parent_index: number;
    original_page: number;
    view_page: number;
    normalized_box: [number, number, number, number];
  }>;
  selected_positions: number[];
  element_mappings: Array<{
    parent_index: number;
    original_position: number;
    original_locator: AnnexLocator;
    view_position: number;
  }>;
  boundary_positions: number[];
  selection_method: string;
}

export interface AnnexExtraction {
  evidence_view: AnnexEvidenceView | null;
  page_count: number | null;
  schema_version: number;
  source_sha256: string;
  mime_type: string;
  elements: AnnexExtractionElement[];
  issues: string[];
}

export interface AnnexCanonicalSnapshot {
  id: string;
  user_file_id: string;
  chunk_type: string | null;
  status: string;
  projection_ordinal: number;
  supersedes_chunk_id: string | null;
  superseded_by_chunk_id: string | null;
  position: number;
  text: string;
  heading_path: string[];
  metadata: Record<string, unknown>;
  source: string;
  validity_start_date: string | null;
  validity_end_date: string | null;
}

export interface AnnexReviewEvidence {
  id: string;
  side: "old" | "new";
  kind:
    | "original"
    | "comparison_page"
    | "comparison_tile"
    | "comparison_region";
  file_id: string;
  sha256: string;
  mime_type: string;
  byte_count: number;
  parent_file_id: string;
  parent_sha256: string;
  source_asset_id: string | null;
  locator: AnnexLocator;
}

export interface AnnexElementCorrection {
  position: number;
  before_text: string;
  corrected_text: string;
  reason: string;
}

export interface AnnexContextImpact {
  direct_canonical_changes: string[];
  contextual_candidates: string[];
  embedding_changes: string[];
  context_only: string[];
  metadata_only: string[];
  retire_history: string[];
  unchanged: string[];
  reasons: Record<string, string[]>;
  ready: boolean;
}

export interface AnnexPublicationReview {
  artifact_file_id: string;
  artifact_sha256: string;
  artifact_byte_count: number;
  scope: {
    tenant_id: string;
    environment: string;
    database_identity: string;
  };
  indexes: Array<{
    index_name: string;
    index_uuid: string;
    search_settings_id: number;
    model_provider: string;
    model_name: string;
    vector_dimension: number;
    embedding_config_sha256: string;
    multitenant: boolean;
  }>;
  counts: {
    canonical_changes: number;
    context_consumers: number;
    embeddings: number;
    exact_vector_reuses: number;
    preserved_vectors?: number;
    metadata_updates?: number;
    unresolved_dependencies?: number;
    historical_projections: number;
    retired_projections: number;
    total_projections: number;
  };
  effective_dates: string[];
  projection_ids: string[];
}

export interface AnnexReviewPayload {
  impact_strategy?: "legacy_full_file" | "source_dependencies_v1";
  selection_parent_id?: string | null;
  instruction_indices: number[];
  instruction_texts: string[];
  annex_label: string;
  effective_date: string | null;
  date_resolution?: {
    effective_start_date: string | null;
    effective_end_date: string | null;
    rationale: string;
  } | null;
  source_package_id: string | null;
  source_text_sha256: string | null;
  source_manifest_sha256: string | null;
  original_source_text_sha256: string | null;
  // Canonical OLD uses current indexed text plus any explicitly bound images.
  old_evidence_kind: "visual" | "canonical_text";
  submitted_source_text: string | null;
  raw_new_extraction: AnnexExtraction | null;
  old_extraction: AnnexExtraction | null;
  new_extraction: AnnexExtraction | null;
  corrections: AnnexElementCorrection[];
  correction_reconciliation: {
    supported: boolean;
    rationale: string;
    input_sha256: string | null;
  } | null;
  baseline_scope?: AnnexCanonicalSnapshot[];
  comparison: {
    changes: AnnexDifference[];
    issues: string[];
    ready: boolean;
  } | null;
  patch_plan: {
    patches: Array<{
      old_chunk_id: string | null;
      old_text: string | null;
      new_text: string | null;
      old_positions: number[];
      new_positions: number[];
      operation: "replace" | "insert" | "remove" | "move" | "visual";
    }>;
    direct_canonical_changes: string[];
    metadata_only: string[];
    retire_history: string[];
    unchanged: string[];
    issues: string[];
    ready: boolean;
  } | null;
  items: Array<{
    operation:
      | "replace"
      | "insert"
      | "remove"
      | "split"
      | "merge"
      | "move"
      | "visual";
    old_chunk_ids: string[];
    new_chunks: AnnexCanonicalSnapshot[];
    old_positions: number[];
    new_positions: number[];
  }>;
  impact: AnnexContextImpact | null;
  evidence: AnnexReviewEvidence[];
  source_only_canonical_ids: string[];
  after_window_authority: {
    kind:
      | "restore_predecessor"
      | "scheduled_successor"
      | "cessation"
      | "unresolved";
    effective_date: string;
    source_text_sha256: string;
    source_start: number;
    source_end: number;
    source_quote: string;
    predecessor_ids: string[];
    successor_ids: string[];
  } | null;
  publication: AnnexPublicationReview | null;
  issues: string[];
}

export type AnnexReviewStatus =
  | "pending"
  | "blocked"
  | "approving"
  | "preparing"
  | "publishing"
  | "approved"
  | "rejected"
  | "failed";

export interface AnnexReview {
  dependency_summary?: {
    unresolved: number;
    affected: number;
    reasons: string[];
  };
  id: string;
  logical_group_id: string;
  review_revision: number;
  batch_id: number;
  status: AnnexReviewStatus;
  review_sha256: string;
  publication_generation: number;
  preparation?: {
    status: "queued" | "running" | "completed" | "failed";
    stage: string;
    completed_chunks: number;
    total_chunks: number;
    error_message: string | null;
    result_review_id: string | null;
  } | null;
  review_payload: AnnexReviewPayload;
  error_message: string | null;
  created_at: string;
}

export interface AnnexChunkReviewItem {
  id: string;
  position: number;
  operation: string;
  old_chunks: AnnexCanonicalSnapshot[];
  new_chunks: AnnexCanonicalSnapshot[];
  old_image_evidence_ids: string[];
  new_image_evidence_ids: string[];
  selection: AnnexReview | null;
}

export interface AnnexChunkReviewPage {
  selection_count?: number;
  items: AnnexChunkReviewItem[];
  total: number;
  offset: number;
  limit: number;
}

export interface AnnexSourceText {
  package_id: string;
  manifest_sha256: string;
  original_text: string;
  original_text_sha256: string;
}

export class RegulatoryRequestError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "RegulatoryRequestError";
  }
}

const handleRequestError = (action: string, response: Response): never => {
  throw new RegulatoryRequestError(
    `${action} failed (Status: ${response.status})`,
    response.status,
  );
};

async function requestJson<T>(
  action: string,
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    if (body?.detail) {
      throw new RegulatoryRequestError(String(body.detail), response.status);
    }
    handleRequestError(action, response);
  }
  return response.json() as Promise<T>;
}

export async function getAnnexCapabilities(): Promise<AnnexCapabilities | null> {
  const response = await fetch("/api/regulatory/amendments/capabilities");
  if (response.status === 404) return null;
  if (!response.ok) handleRequestError("Get annex capabilities", response);
  return response.json();
}

export async function createAmendmentSourcePackage(
  documentSetId: number,
  idempotencyKey: string,
  source: { text: string } | { url: string },
): Promise<AmendmentSourcePackage> {
  return requestJson(
    "Create amendment source package",
    "/api/regulatory/amendments/source-packages",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_set_id: documentSetId,
        idempotency_key: idempotencyKey,
        ...source,
      }),
    },
  );
}

export async function uploadAmendmentSourcePackage(
  documentSetId: number,
  idempotencyKey: string,
  file: File,
): Promise<AmendmentSourcePackage> {
  const body = new FormData();
  body.append("document_set_id", String(documentSetId));
  body.append("idempotency_key", idempotencyKey);
  body.append("file", file);
  return requestJson(
    "Upload amendment source package",
    "/api/regulatory/amendments/source-packages/upload",
    {
      method: "POST",
      body,
    },
  );
}

export async function getAmendmentSourcePackage(
  documentSetId: number,
  packageId: string,
): Promise<AmendmentSourcePackage> {
  return requestJson(
    "Get amendment source package",
    `/api/regulatory/amendments/source-packages/${packageId}?document_set_id=${documentSetId}`,
  );
}

export async function retryAmendmentSourcePackage(
  documentSetId: number,
  packageId: string,
): Promise<AmendmentSourcePackage> {
  return requestJson(
    "Retry amendment source package",
    `/api/regulatory/amendments/source-packages/${packageId}/retry?document_set_id=${documentSetId}`,
    { method: "POST" },
  );
}

export async function getAmendmentSourceText(
  documentSetId: number,
  packageId: string,
): Promise<AnnexSourceText> {
  return requestJson(
    "Get original amendment source text",
    `/api/regulatory/amendments/source-packages/${packageId}/text?document_set_id=${documentSetId}`,
  );
}

export function getAnnexReview(review: AnnexReview): Promise<AnnexReview> {
  return requestJson(
    "Get annex review",
    `/api/regulatory/amendments/batches/${review.batch_id}/annex-groups/${review.id}`,
  );
}

export function getAnnexChunkPage(
  review: AnnexReview,
  offset: number,
): Promise<AnnexChunkReviewPage> {
  return requestJson(
    "Get chunk changes",
    `/api/regulatory/amendments/batches/${review.batch_id}/annex-groups/${review.id}/chunks?offset=${offset}&limit=10`,
  );
}

export function prepareAnnexSelection(
  review: AnnexReview,
  itemIds: string[],
): Promise<AnnexReview> {
  return requestJson(
    "Prepare selected chunks",
    `/api/regulatory/amendments/batches/${review.batch_id}/annex-groups/${review.id}/selections`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_review_sha256: review.review_sha256,
        item_ids: itemIds,
      }),
    },
  );
}

export async function listAnnexReviews(
  batchId: number,
): Promise<AnnexReview[]> {
  return requestJson(
    "List annex reviews",
    `/api/regulatory/amendments/batches/${batchId}/annex-groups`,
  );
}

export async function listAnnexReviewRevisions(
  batchId: number,
  reviewId: string,
): Promise<AnnexReview[]> {
  return requestJson(
    "List annex review revisions",
    `/api/regulatory/amendments/batches/${batchId}/annex-groups/${reviewId}/revisions`,
  );
}

async function decideAnnexReview(
  action: "approve" | "reject" | "retry",
  batchId: number,
  reviewId: string,
  expectedReviewSha256: string,
): Promise<AnnexReview> {
  return requestJson(
    `${action[0]?.toUpperCase()}${action.slice(1)} annex review`,
    `/api/regulatory/amendments/batches/${batchId}/annex-groups/${reviewId}/${action}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_review_sha256: expectedReviewSha256 }),
    },
  );
}

export function approveAnnexReview(
  batchId: number,
  reviewId: string,
  expectedReviewSha256: string,
): Promise<AnnexReview> {
  return decideAnnexReview("approve", batchId, reviewId, expectedReviewSha256);
}

export function rejectAnnexReview(
  batchId: number,
  reviewId: string,
  expectedReviewSha256: string,
): Promise<AnnexReview> {
  return decideAnnexReview("reject", batchId, reviewId, expectedReviewSha256);
}

export function retryAnnexReview(
  batchId: number,
  reviewId: string,
  expectedReviewSha256: string,
): Promise<AnnexReview> {
  return decideAnnexReview("retry", batchId, reviewId, expectedReviewSha256);
}

export async function editAnnexReview(
  batchId: number,
  reviewId: string,
  expectedReviewSha256: string,
  corrections: AnnexElementCorrection[],
): Promise<AnnexReview> {
  return requestJson(
    "Revalidate annex review",
    `/api/regulatory/amendments/batches/${batchId}/annex-groups/${reviewId}/edit`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_review_sha256: expectedReviewSha256,
        corrections,
      }),
    },
  );
}

export async function createAmendmentSourceTextRevision(
  batchId: number,
  rawText: string,
  expectedSourceTextSha256: string,
  sourcePackageId?: string | null,
): Promise<AmendmentBatch> {
  return requestJson(
    "Create amendment source text revision",
    `/api/regulatory/amendments/batches/${batchId}/source-revisions`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_text: rawText,
        expected_source_text_sha256: expectedSourceTextSha256,
        source_package_id: sourcePackageId ?? null,
      }),
    },
  );
}

export function getAnnexEvidenceUrl(
  batchId: number,
  reviewId: string,
  evidenceId: string,
): string {
  return `/api/regulatory/amendments/batches/${batchId}/annex-groups/${reviewId}/evidence/${evidenceId}`;
}

export async function analyzeAmendment(
  documentSetId: number,
  rawText: string,
  sourcePackageId?: string,
): Promise<AmendmentBatch> {
  const response = await fetch("/api/regulatory/amendments/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_set_id: documentSetId,
      raw_text: rawText,
      ...(sourcePackageId ? { source_package_id: sourcePackageId } : {}),
    }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Amendment analysis failed (Status: ${response.status})`,
    );
  }
  return response.json();
}

export async function getAmendmentAnalysis(
  batchId: number,
): Promise<AnalyzeAmendmentResponse> {
  const response = await fetch(
    `/api/regulatory/amendments/batches/${batchId}/analysis`,
  );
  if (!response.ok) {
    handleRequestError("Get amendment analysis", response);
  }
  return response.json();
}

export async function retryAmendmentBatch(
  batchId: number,
): Promise<AmendmentBatch> {
  const response = await fetch(
    `/api/regulatory/amendments/batches/${batchId}/retry`,
    { method: "POST" },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Retry analysis failed (Status: ${response.status})`,
    );
  }
  return response.json();
}

export async function extractAmendmentUrl(
  url: string,
): Promise<AmendmentSourceExtraction> {
  const response = await fetch("/api/regulatory/amendments/sources/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `URL extraction failed (Status: ${response.status})`,
    );
  }
  return response.json();
}

export async function extractAmendmentPdf(
  file: File,
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
      body?.detail || `PDF extraction failed (Status: ${response.status})`,
    );
  }
  return response.json();
}

export async function extractAmendmentDocx(
  file: File,
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
      body?.detail || `Word extraction failed (Status: ${response.status})`,
    );
  }
  return response.json();
}

export async function listAmendmentBatches(
  documentSetId: number,
): Promise<AmendmentBatch[]> {
  const response = await fetch(
    `/api/regulatory/amendments/batches?document_set_id=${documentSetId}`,
  );
  if (!response.ok) {
    handleRequestError("List amendment batches", response);
  }
  return response.json();
}

export async function listAmendmentProposals(
  batchId: number,
): Promise<AmendmentProposal[]> {
  const response = await fetch(
    `/api/regulatory/amendments/batches/${batchId}/proposals`,
  );
  if (!response.ok) {
    handleRequestError("List amendment proposals", response);
  }
  return response.json();
}

export async function approveProposal(
  proposalId: number,
  newChunkDraft: Record<string, unknown>,
  chunkChanges?: AmendmentProposalChunkChange[],
): Promise<AmendmentProposal> {
  const response = await fetch(
    `/api/regulatory/amendments/proposals/${proposalId}/approve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        new_chunk_draft: newChunkDraft,
        chunk_changes: chunkChanges,
      }),
    },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Approve failed (Status: ${response.status})`,
    );
  }
  return response.json();
}

export async function retryProposalIndexing(
  proposalId: number,
): Promise<AmendmentProposal> {
  const response = await fetch(
    `/api/regulatory/amendments/proposals/${proposalId}/retry`,
    { method: "POST" },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Retry indexing failed (Status: ${response.status})`,
    );
  }
  return response.json();
}

export async function rejectProposal(
  proposalId: number,
): Promise<AmendmentProposal> {
  const response = await fetch(
    `/api/regulatory/amendments/proposals/${proposalId}/reject`,
    { method: "POST" },
  );
  if (!response.ok) {
    handleRequestError("Reject proposal", response);
  }
  return response.json();
}
