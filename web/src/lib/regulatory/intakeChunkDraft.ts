import type { IntakeChunkUpdate } from "@/lib/regulatory/documentIntake";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import {
  intakeMetadataEdits,
  intakeMetadataInputs,
} from "@/lib/regulatory/intakeMetadata";

export interface IntakeChunkReviewDraft {
  values: IntakeChunkUpdate;
  baseValues: IntakeChunkUpdate;
  baseMetadata: Record<string, unknown>;
  metadataInputs: Record<string, string>;
  headingPath: string;
  reviewSha256: string | null;
}

export interface IntakePendingChunkEdit {
  documentId: string;
  chunkId: string;
  draft: IntakeChunkReviewDraft;
}

export function createIntakeChunkDraft(
  chunk: RegulatoryChunk,
  reviewSha256: string | null
): IntakeChunkReviewDraft {
  const values = {
    text: chunk.text,
    heading_path: chunk.heading_path,
    chunk_type: chunk.chunk_type ?? null,
    validity_start_date: chunk.validity_start_date ?? null,
    validity_end_date: chunk.validity_end_date ?? null,
  };
  return {
    values,
    baseValues: values,
    baseMetadata: chunk.chunk_metadata,
    metadataInputs: intakeMetadataInputs(chunk.chunk_metadata),
    headingPath: chunk.heading_path.join("\n"),
    reviewSha256,
  };
}

export function intakeChunkDraftDirty(draft: IntakeChunkReviewDraft): boolean {
  return (
    JSON.stringify(draft.values) !== JSON.stringify(draft.baseValues) ||
    draft.headingPath !== draft.baseValues.heading_path.join("\n") ||
    intakeMetadataEdits(draft.baseMetadata, draft.metadataInputs).dirty
  );
}

export function validIntakeDate(value: string | null): boolean {
  if (!value) return true;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return (
    Number.isFinite(parsed.getTime()) &&
    parsed.toISOString().slice(0, 10) === value
  );
}

export function intakeChunkDraftValidation(
  draft: IntakeChunkReviewDraft
): string | null {
  const values = draft.values;
  if (!values.text.trim()) return "Chunk text cannot be empty.";
  if (
    !validIntakeDate(values.validity_start_date) ||
    !validIntakeDate(values.validity_end_date)
  )
    return "Use a valid YYYY-MM-DD date.";
  if (
    values.validity_start_date &&
    values.validity_end_date &&
    values.validity_end_date <= values.validity_start_date
  )
    return "The end date must be after the start date.";
  return intakeMetadataEdits(draft.baseMetadata, draft.metadataInputs).error;
}

export function intakeChunkDraftUpdate(
  draft: IntakeChunkReviewDraft
): IntakeChunkUpdate {
  const metadata = intakeMetadataEdits(
    draft.baseMetadata,
    draft.metadataInputs
  );
  return {
    ...draft.values,
    heading_path: draft.headingPath
      .split("\n")
      .map((heading) => heading.trim())
      .filter(Boolean),
    ...(metadata.dirty ? { chunk_metadata: metadata.patch } : {}),
  };
}

export function hasIntakeLabelingProblem(chunk: RegulatoryChunk): boolean {
  return (
    Boolean(chunk.labeling_error) ||
    ["failed", "abstained", "abstain", "stale", "cancelled"].includes(
      chunk.labeling_status ?? ""
    )
  );
}
