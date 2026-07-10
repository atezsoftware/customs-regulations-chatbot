import {
  resolveCoreApiRestUrl,
  resolveDatabaseUrl,
  virtualCorpusKey,
} from '../../directories/services';

export interface ProposalRecord {
  id: string;
  batchId: string;
  instructionIndex: number;
  instructionText: string;
  oldChunkId: string | null;
  oldChunkSnapshot: Record<string, unknown>;
  newChunkDraft: Record<string, unknown>;
  matchConfidence: number | null;
  matchRationale: string | null;
  dateRationale: string | null;
  status: 'pending' | 'approved' | 'rejected';
  appliedNewChunkId: string | null;
  decidedBy: string | null;
  decidedAt: string | null;
  createdAt: string;
  updatedAt: string;
  duplicateTarget: boolean;
}

export interface AmendmentBatch {
  id: string;
  corpusId: string;
  rawText: string;
  referenceDate: string | null;
  status: 'analyzing' | 'analyzed' | 'failed';
  errorMessage: string | null;
  createdBy: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface AnalyzeAmendmentResult {
  batchId: string;
  referenceDate: string | null;
  proposals: ProposalRecord[];
  unmatchedInstructions: string[];
}

export interface ApproveAmendmentResult {
  applied: ProposalRecord[];
  failed: Array<{proposalId: string; reason: string}>;
}

interface RawProposalRecord {
  id: string;
  batch_id: string;
  instruction_index: number;
  instruction_text: string;
  old_chunk_id: string | null;
  old_chunk_snapshot: Record<string, unknown>;
  new_chunk_draft: Record<string, unknown>;
  match_confidence: number | null;
  match_rationale: string | null;
  date_rationale: string | null;
  status: 'pending' | 'approved' | 'rejected';
  applied_new_chunk_id: string | null;
  decided_by: string | null;
  decided_at: string | null;
  created_at: string;
  updated_at: string;
  duplicate_target: boolean;
}

interface RawAmendmentBatch {
  id: string;
  corpus_id: string;
  raw_text: string;
  reference_date: string | null;
  status: 'analyzing' | 'analyzed' | 'failed';
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  updated_at: string;
}

function toProposalRecord(raw: RawProposalRecord): ProposalRecord {
  return {
    id: raw.id,
    batchId: raw.batch_id,
    instructionIndex: raw.instruction_index,
    instructionText: raw.instruction_text,
    oldChunkId: raw.old_chunk_id,
    oldChunkSnapshot: raw.old_chunk_snapshot,
    newChunkDraft: raw.new_chunk_draft,
    matchConfidence: raw.match_confidence,
    matchRationale: raw.match_rationale,
    dateRationale: raw.date_rationale,
    status: raw.status,
    appliedNewChunkId: raw.applied_new_chunk_id,
    decidedBy: raw.decided_by,
    decidedAt: raw.decided_at,
    createdAt: raw.created_at,
    updatedAt: raw.updated_at,
    duplicateTarget: raw.duplicate_target,
  };
}

function toAmendmentBatch(raw: RawAmendmentBatch): AmendmentBatch {
  return {
    id: raw.id,
    corpusId: raw.corpus_id,
    rawText: raw.raw_text,
    referenceDate: raw.reference_date,
    status: raw.status,
    errorMessage: raw.error_message,
    createdBy: raw.created_by,
    createdAt: raw.created_at,
    updatedAt: raw.updated_at,
  };
}

function coreHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = {'Content-Type': 'application/json', ...extra};
  if (process.env.CORE_INTERNAL_TOKEN) {
    headers['X-Internal-Token'] = process.env.CORE_INTERNAL_TOKEN;
  }
  return headers;
}

/**
 * Thrown by `coreFetch` for any non-2xx (or `{error: ...}`-shaped) response
 * from core-api, carrying the original HTTP status so callers (the
 * controller) can translate it into the right `HttpErrors.*` instead of
 * everything collapsing into an opaque 500 — e.g. core-api's 404 "No index
 * found for this folder" should reach the admin as a 404, not a generic
 * Internal Server Error with the real reason only visible in server logs.
 */
export class CoreApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'CoreApiError';
  }
}

async function coreFetch(path: string, init: RequestInit = {}): Promise<Record<string, unknown>> {
  const endpoint = `${resolveCoreApiRestUrl()}${path}`;
  let res: Response;
  try {
    res = await fetch(endpoint, {...init, headers: coreHeaders(init.headers as Record<string, string>)});
  } catch (error) {
    throw new CoreApiError(
      `Core is not reachable at ${endpoint}. Start core-api with ` +
        '`scripts/run.sh --env dev --apps core-api` or run the full stack. ' +
        `Original error: ${error instanceof Error ? error.message : String(error)}`,
      503,
    );
  }

  const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (!res.ok) {
    const message = typeof body.error === 'string' ? body.error : `HTTP ${res.status}`;
    throw new CoreApiError(`Core amendments request failed: ${message}`, res.status);
  }
  if (typeof body.error === 'string') {
    throw new CoreApiError(`Core amendments request returned error: ${body.error}`, 502);
  }
  return body;
}

export async function analyzeAmendment(
  directoryId: number,
  rawText: string,
): Promise<AnalyzeAmendmentResult> {
  const body = await coreFetch('/api/amendments/analyze', {
    method: 'POST',
    body: JSON.stringify({
      corpus_folder: virtualCorpusKey(directoryId),
      raw_text: rawText,
      database_url: resolveDatabaseUrl(),
    }),
  });
  return {
    batchId: String(body.batch_id),
    referenceDate: (body.reference_date as string | null) ?? null,
    proposals: (body.proposals as RawProposalRecord[]).map(toProposalRecord),
    unmatchedInstructions: (body.unmatched_instructions as string[]) ?? [],
  };
}

export async function listAmendmentProposals(params: {
  directoryId?: number;
  status?: string;
  batchId?: string;
}): Promise<ProposalRecord[]> {
  const query = new URLSearchParams();
  if (params.status) query.set('status', params.status);
  if (params.batchId) query.set('batch_id', params.batchId);
  if (params.directoryId !== undefined) {
    query.set('corpus_folder', virtualCorpusKey(params.directoryId));
  }
  const databaseUrl = resolveDatabaseUrl();
  if (databaseUrl) query.set('database_url', databaseUrl);

  const body = await coreFetch(`/api/amendments/proposals?${query.toString()}`);
  return ((body.proposals as RawProposalRecord[]) ?? []).map(toProposalRecord);
}

export async function getAmendmentBatch(
  batchId: string,
): Promise<{batch: AmendmentBatch; proposals: ProposalRecord[]}> {
  const query = new URLSearchParams();
  const databaseUrl = resolveDatabaseUrl();
  if (databaseUrl) query.set('database_url', databaseUrl);

  const body = await coreFetch(`/api/amendments/batches/${batchId}?${query.toString()}`);
  return {
    batch: toAmendmentBatch(body.batch as RawAmendmentBatch),
    proposals: ((body.proposals as RawProposalRecord[]) ?? []).map(toProposalRecord),
  };
}

export async function approveAmendmentProposals(
  proposalIds: string[],
  decidedBy?: string,
): Promise<ApproveAmendmentResult> {
  const body = await coreFetch('/api/amendments/proposals/approve', {
    method: 'POST',
    body: JSON.stringify({
      proposal_ids: proposalIds,
      database_url: resolveDatabaseUrl(),
      decided_by: decidedBy,
    }),
  });
  return {
    applied: ((body.applied as RawProposalRecord[]) ?? []).map(toProposalRecord),
    failed:
      (body.failed as Array<{proposal_id: string; reason: string}>)?.map(f => ({
        proposalId: f.proposal_id,
        reason: f.reason,
      })) ?? [],
  };
}

export async function rejectAmendmentProposal(
  proposalId: string,
  decidedBy?: string,
): Promise<ProposalRecord> {
  const body = await coreFetch(`/api/amendments/proposals/${proposalId}/reject`, {
    method: 'POST',
    body: JSON.stringify({database_url: resolveDatabaseUrl(), decided_by: decidedBy}),
  });
  return toProposalRecord(body.proposal as RawProposalRecord);
}

export async function deleteAmendmentProposal(proposalId: string): Promise<void> {
  const query = new URLSearchParams();
  const databaseUrl = resolveDatabaseUrl();
  if (databaseUrl) query.set('database_url', databaseUrl);
  await coreFetch(`/api/amendments/proposals/${proposalId}?${query.toString()}`, {
    method: 'DELETE',
  });
}
