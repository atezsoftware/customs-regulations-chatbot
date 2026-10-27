import type { RegulatoryChunkPage } from "@/lib/regulatory/interfaces";
import type {
  AmendmentSourceIssue,
  AmendmentSourcePackage,
} from "@/lib/regulatory/amendments";
import type { LabelingSetup } from "@/lib/documentSetLabeling/interfaces";

export interface DocumentIntakeCapabilities {
  enabled: boolean;
  target_document_set_id: number | null;
  target_document_set_name: string;
  error: string | null;
}

export interface IntakeDocument {
  id: string;
  asset_id: string;
  display_name: string;
  role: "root" | "annex";
  parent_document_id: string | null;
  source_url: string | null;
  status: string;
  chunk_count: number;
  labeling_failed_count?: number;
  labels: Record<string, unknown>;
  error: string | null;
  user_file_id: string | null;
  document_links?: {
    href: string;
    label: string;
    target_document_id: string;
  }[];
  lineage_review?: IntakeLineageReview | null;
}

export interface IntakePredecessorCandidate {
  user_file_id: string;
  title: string;
  reason: string;
  score: number;
  status: string;
}

export interface IntakeLineageSelection {
  predecessor_user_file_id: string | null;
  relation: "successor" | "repeals";
  effective_date: string | null;
  evidence: string | null;
}

export interface IntakeLineageReview extends IntakeLineageSelection {
  title: string;
  status: "draft" | "frozen" | "active";
}

export interface IntakeChunkUpdate {
  text: string;
  heading_path: string[];
  chunk_type: string | null;
  validity_start_date: string | null;
  validity_end_date: string | null;
  chunk_metadata?: Record<string, string | number | null>;
}

export interface IntakeChunkEdit {
  document_id: string;
  chunk_id: string;
  updates: IntakeChunkUpdate;
}

export interface IntakeRepreparePreview {
  eligible: boolean;
  expected_sha256: string | null;
  reason: string | null;
  request_id: string | null;
  status: string | null;
  previous_labeling_run_id: string | null;
  documents: Record<string, unknown>;
}

export interface LegislationLineageEdge {
  user_file_id: string;
  title: string;
  relation: "successor" | "repeals";
  effective_date: string | null;
  evidence: string | null;
}

export interface LegislationLineage {
  user_file_id: string;
  title: string;
  status: "current" | "repeal_scheduled" | "repealed";
  effective_date: string | null;
  predecessors: LegislationLineageEdge[];
  successors: LegislationLineageEdge[];
}

export interface DocumentIntakeBatch {
  id: string;
  source_package_id: string;
  status:
    | "preparing"
    | "ready"
    | "approving"
    | "completed"
    | "partial"
    | "blocked"
    | "failed";
  stage: string;
  target_document_set_id: number;
  target_document_set_name: string;
  review_sha256: string | null;
  edit_review_sha256?: string | null;
  labeling_failed_count?: number;
  error?: string | null;
  issues: AmendmentSourceIssue[];
  documents: IntakeDocument[];
}

const baseUrl = "/api/regulatory/document-intake";

export interface IntakeSourceLink {
  source_field: string;
  label: string;
  original_url: string | null;
  requested_url: string | null;
  final_url: string | null;
  target_asset_hash: string | null;
}

export function getDocumentIntakeSourceLinks(
  packageId: string
): Promise<IntakeSourceLink[]> {
  return requestJson(`/source-packages/${encodeURIComponent(packageId)}/links`);
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Document intake failed (Status: ${response.status})`
    );
  }
  return response.json() as Promise<T>;
}

export function getDocumentIntakeCapabilities(): Promise<DocumentIntakeCapabilities> {
  return requestJson("/capabilities");
}

export function getDocumentIntakeLabelingSetup(): Promise<LabelingSetup> {
  return requestJson("/labeling-setup");
}

export function uploadDocumentIntakeMarkdownPackage(
  file: File,
  idempotencyKey: string
): Promise<AmendmentSourcePackage> {
  const body = new FormData();
  body.append("file", file);
  body.append("idempotency_key", idempotencyKey);
  return requestJson("/source-packages/markdown", {
    method: "POST",
    body,
  });
}

export function uploadDocumentIntakeZipPackage(
  file: File,
  idempotencyKey: string
): Promise<AmendmentSourcePackage> {
  const body = new FormData();
  body.append("file", file);
  body.append("idempotency_key", idempotencyKey);
  return requestJson("/source-packages/zip", { method: "POST", body });
}

export function listDocumentIntakeBatches(): Promise<DocumentIntakeBatch[]> {
  return requestJson("/batches");
}

export function getDocumentIntakeBatch(
  id: string
): Promise<DocumentIntakeBatch> {
  return requestJson(`/batches/${encodeURIComponent(id)}`);
}

export function createDocumentIntakeBatch(
  sourcePackageId: string,
  idempotencyKey: string
): Promise<DocumentIntakeBatch> {
  return requestJson("/batches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_package_id: sourcePackageId,
      idempotency_key: idempotencyKey,
    }),
  });
}

export function retryDocumentIntakeBatch(
  id: string
): Promise<DocumentIntakeBatch> {
  return requestJson(`/batches/${encodeURIComponent(id)}/retry`, {
    method: "POST",
  });
}

export function getDocumentIntakeRepreparePreview(
  batchId: string
): Promise<IntakeRepreparePreview> {
  return requestJson(`/batches/${encodeURIComponent(batchId)}/reprepare`);
}

export function reprepareDocumentIntakeBatch(
  batchId: string,
  expectedSha256: string,
  idempotencyKey: string
): Promise<DocumentIntakeBatch> {
  return requestJson(`/batches/${encodeURIComponent(batchId)}/reprepare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      expected_sha256: expectedSha256,
      idempotency_key: idempotencyKey,
    }),
  });
}

export function approveDocumentIntakeBatch(
  id: string,
  expectedReviewSha256: string
): Promise<DocumentIntakeBatch> {
  return requestJson(`/batches/${encodeURIComponent(id)}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_review_sha256: expectedReviewSha256 }),
  });
}

export function getDocumentIntakeChunks(
  batchId: string,
  documentId: string,
  offset = 0
): Promise<RegulatoryChunkPage> {
  return requestJson(
    `/batches/${encodeURIComponent(batchId)}/documents/${encodeURIComponent(documentId)}/chunks?offset=${offset}&limit=10`
  );
}

export function updateDocumentIntakeChunk(
  batchId: string,
  documentId: string,
  chunkId: string,
  expectedReviewSha256: string,
  draft: IntakeChunkUpdate
): Promise<DocumentIntakeBatch> {
  return requestJson(
    `/batches/${encodeURIComponent(batchId)}/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_review_sha256: expectedReviewSha256,
        ...draft,
      }),
    }
  );
}

export function updateDocumentIntakeChunks(
  batchId: string,
  expectedReviewSha256: string,
  edits: IntakeChunkEdit[]
): Promise<DocumentIntakeBatch> {
  return requestJson(`/batches/${encodeURIComponent(batchId)}/chunks`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      expected_review_sha256: expectedReviewSha256,
      edits,
    }),
  });
}

export function getDocumentIntakePredecessorCandidates(
  batchId: string,
  documentId: string,
  query = ""
): Promise<IntakePredecessorCandidate[]> {
  const suffix = query ? `?query=${encodeURIComponent(query)}` : "";
  return requestJson(
    `/batches/${encodeURIComponent(batchId)}/documents/${encodeURIComponent(documentId)}/predecessor-candidates${suffix}`
  );
}

export function saveDocumentIntakePredecessorSelection(
  batchId: string,
  documentId: string,
  expectedReviewSha256: string,
  selection: IntakeLineageSelection
): Promise<DocumentIntakeBatch> {
  return requestJson(
    `/batches/${encodeURIComponent(batchId)}/documents/${encodeURIComponent(documentId)}/predecessor-selection`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_review_sha256: expectedReviewSha256,
        ...selection,
      }),
    }
  );
}

export function getLegislationLineage(
  fileId: string
): Promise<LegislationLineage> {
  return requestJson(`/files/${encodeURIComponent(fileId)}/lineage`);
}
