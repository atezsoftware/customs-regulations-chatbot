"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  Button,
  InputSelect,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import { SettingsLayouts, toast } from "@opal/layouts";
import SvgHistory from "@opal/icons/history";
import SvgChevronLeft from "@opal/icons/chevron-left";
import SvgChevronRight from "@opal/icons/chevron-right";
import { useDocumentSets } from "@/lib/hooks/useDocumentSets";
import {
  type AmendmentSourcePackage,
  type AnnexCapabilities,
  type AnnexReview,
  type AmendmentAnalysisLogEntry,
  type AmendmentBatch,
  type AmendmentProposal,
  type AmendmentProposalChunkChange,
  RegulatoryRequestError,
  analyzeAmendment,
  approveProposal,
  createAmendmentSourcePackage,
  extractAmendmentDocx,
  extractAmendmentPdf,
  extractAmendmentUrl,
  getAmendmentAnalysis,
  getAmendmentSourcePackage,
  getAmendmentSourceText,
  getAnnexCapabilities,
  listAnnexReviews,
  listAmendmentBatches,
  listAmendmentProposals,
  rejectProposal,
  retryAmendmentBatch,
  retryAmendmentSourcePackage,
  retryProposalIndexing,
  uploadAmendmentSourcePackage,
} from "@/lib/regulatory/amendments";
import AnnexChunkReview from "@/views/admin/AnnexChunkReview";
import { ChunkContent } from "@/sections/cards/ChunkChangeCard";

type AmendmentSourceMode =
  | "text"
  | "url"
  | "pdf"
  | "docx"
  | "image"
  | "html"
  | "xlsx";

const FILE_SOURCE_MODES = new Set<AmendmentSourceMode>([
  "pdf",
  "docx",
  "image",
  "html",
  "xlsx",
]);

const sourceModeDetails: Record<
  Exclude<AmendmentSourceMode, "text" | "url">,
  { button: string; choose: string; label: string; accept: string }
> = {
  pdf: {
    button: "PDF",
    choose: "PDF",
    label: "Amendment source PDF",
    accept: "application/pdf,.pdf",
  },
  docx: {
    button: "Word (.docx)",
    choose: "Word document",
    label: "Amendment source Word document",
    accept:
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx",
  },
  image: {
    button: "Image",
    choose: "image",
    label: "Amendment source image",
    accept:
      "image/png,image/jpeg,image/webp,image/tiff,.png,.jpg,.jpeg,.webp,.tif,.tiff",
  },
  html: {
    button: "HTML",
    choose: "HTML file",
    label: "Amendment source HTML file",
    accept: "text/html,application/xhtml+xml,.html,.htm",
  },
  xlsx: {
    button: "Excel (.xlsx)",
    choose: "Excel workbook",
    label: "Amendment source Excel workbook",
    accept:
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,.xlsx",
  },
};

function sourceIdentity(
  mode: AmendmentSourceMode,
  url: string,
  file: File | null,
  text = "",
) {
  if (mode === "text") {
    const normalizedText = text.trim();
    if (!normalizedText) return null;
    let hash = 2166136261;
    for (let index = 0; index < normalizedText.length; index += 1) {
      hash = Math.imul(hash ^ normalizedText.charCodeAt(index), 16777619);
    }
    return `text:${normalizedText.length}:${(hash >>> 0).toString(16)}`;
  }
  if (mode === "url") return url.trim() ? `url:${url.trim()}` : null;
  if (FILE_SOURCE_MODES.has(mode) && file) {
    return `${mode}:${file.name}:${file.size}:${file.lastModified}`;
  }
  return null;
}

function sourceRequestIdentity() {
  return `annex-source-${globalThis.crypto.randomUUID()}`;
}

function sourceIssueMessage(code: string) {
  if (code === "source_preparation_timeout") {
    return "Source preparation timed out while reading the document or its attachments. Retry source preparation.";
  }
  if (code === "image_source_requires_new_preparation") {
    return "This image source was saved without extracted text. Click Prepare source to create a new package.";
  }
  return code.replaceAll("_", " ");
}

function analysisProgressLabel(batch: AmendmentBatch) {
  if (batch.stage === "waiting_resources")
    return "Waiting for available memory; saved progress is preserved.";
  if (batch.stage === "segmenting") return "Segmenting amendment…";
  if (batch.stage === "finalizing") return "Finalizing analysis…";
  if (batch.instruction_count > 0) {
    if (batch.matched_instruction_count !== undefined) {
      return `Matched ${batch.matched_instruction_count} / ${batch.instruction_count} · Finalized ${batch.processed_instruction_count} / ${batch.instruction_count}`;
    }
    return `Analyzing ${batch.processed_instruction_count} / ${batch.instruction_count}`;
  }
  return "Analysis queued…";
}

const emptyCurrentChunkSnapshot: Record<string, unknown> = {
  id: null,
  user_file_id: null,
  position: null,
  text: null,
  chunk_type: null,
  heading_path: [],
  metadata: {},
  validity_start_date: null,
  validity_end_date: null,
  status: null,
  source: null,
  supersedes_chunk_id: null,
  superseded_by_chunk_id: null,
  created_at: null,
  updated_at: null,
};

const chunkFieldOrder = [
  "id",
  "user_file_id",
  "position",
  "text",
  "chunk_type",
  "heading_path",
  "metadata",
  "validity_start_date",
  "validity_end_date",
  "status",
  "source",
  "supersedes_chunk_id",
  "superseded_by_chunk_id",
  "created_at",
  "updated_at",
] as const;

const alwaysVisibleFields = new Set([
  "validity_start_date",
  "validity_end_date",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && !Array.isArray(value) && typeof value === "object";
}

function cloneEditableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cloneEditableValue);
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [
        key,
        cloneEditableValue(child),
      ]),
    );
  }
  return value;
}

function cloneDraft(draft: Record<string, unknown>): Record<string, unknown> {
  return cloneEditableValue(draft) as Record<string, unknown>;
}

function ReadOnlyFieldValue({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === "") {
    return <span className="text-text-03">—</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-text-03">Empty</span>;
    }
    return (
      <div className="flex flex-col gap-1">
        {value.map((item, index) => (
          <div key={index} className="break-words text-sm text-text-05">
            {String(item)}
          </div>
        ))}
      </div>
    );
  }
  if (isRecord(value)) {
    const visibleEntries = Object.entries(value).filter(
      ([, child]) => child !== null && child !== undefined,
    );
    if (visibleEntries.length === 0) {
      return <span className="text-text-03">Empty</span>;
    }
    return (
      <div className="flex flex-col gap-1.5">
        {visibleEntries.map(([key, child]) => (
          <div
            key={key}
            className="grid grid-cols-[minmax(6rem,0.35fr)_minmax(0,1fr)] gap-2"
          >
            <span className="break-words font-mono text-xs text-text-03">
              {key}
            </span>
            <ReadOnlyFieldValue value={child} />
          </div>
        ))}
      </div>
    );
  }
  return (
    <span className="whitespace-pre-wrap break-words text-sm text-text-05">
      {String(value)}
    </span>
  );
}

function FieldTable({
  title,
  fields,
  renderValue,
}: {
  title: string;
  fields: Record<string, unknown> | null;
  renderValue?: (key: string, value: unknown) => React.ReactNode;
}) {
  const visibleKeys = fields
    ? chunkFieldOrder.filter(
        (key) =>
          alwaysVisibleFields.has(key) ||
          (fields[key] !== null && fields[key] !== undefined),
      )
    : [];

  return (
    <div className="min-w-0 flex-1 overflow-hidden rounded-12 border border-border-02 bg-background-neutral-00">
      <div className="border-b border-border-02 bg-background-tint-01 px-3 py-2.5">
        <Text font="main-ui-action" color="text-04">
          {title}
        </Text>
      </div>
      {fields === null ? (
        <div className="p-3">
          <Text font="main-ui-body" color="text-03">
            (new — no existing chunk)
          </Text>
        </div>
      ) : (
        <div role="table" aria-label={`${title} chunk fields`}>
          {visibleKeys.map((key) => (
            <div
              key={key}
              role="row"
              className="grid grid-cols-1 gap-2 border-b border-border-01 px-3 py-2.5 last:border-b-0 sm:grid-cols-[minmax(9rem,0.35fr)_minmax(0,1fr)]"
            >
              <div role="rowheader">
                <span className="break-words font-mono text-xs text-text-03">
                  {key}
                </span>
              </div>
              <div role="cell" className="min-w-0">
                {renderValue ? (
                  renderValue(key, fields[key])
                ) : key === "text" && typeof fields[key] === "string" ? (
                  <ChunkContent text={fields[key]} />
                ) : (
                  <ReadOnlyFieldValue value={fields[key]} />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function isIsoDate(value: unknown): boolean {
  if (value === null || value === "") return true;
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return false;
  }
  const year = Number(value.slice(0, 4));
  const month = Number(value.slice(5, 7));
  const day = Number(value.slice(8, 10));
  const parsed = new Date(Date.UTC(year, month - 1, day));
  return (
    parsed.getUTCFullYear() === year &&
    parsed.getUTCMonth() === month - 1 &&
    parsed.getUTCDate() === day
  );
}

function draftValidationError(draft: Record<string, unknown>): string | null {
  if (typeof draft.text !== "string" || !draft.text.trim()) {
    return "Replacement text cannot be empty.";
  }
  if (
    !isIsoDate(draft.effective_start_date) ||
    !isIsoDate(draft.effective_end_date)
  ) {
    return "Use a YYYY-MM-DD date.";
  }
  return null;
}

interface MetadataLeaf {
  path: string[];
  value: unknown;
}

function metadataLeaves(
  value: Record<string, unknown>,
  prefix: string[] = [],
): MetadataLeaf[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = [...prefix, key];
    if (child === null || child === undefined) return [];
    if (isRecord(child) && Object.keys(child).length > 0) {
      return metadataLeaves(child, path);
    }
    return [{ path, value: child }];
  });
}

function updateNestedValue(
  root: Record<string, unknown>,
  path: string[],
  value: unknown,
): Record<string, unknown> {
  const [head, ...tail] = path;
  if (head === undefined) return root;
  if (tail.length === 0) return { ...root, [head]: value };
  const child = isRecord(root[head]) ? root[head] : {};
  return { ...root, [head]: updateNestedValue(child, tail, value) };
}

function metadataInputValue(value: unknown): string {
  if (Array.isArray(value)) return value.map(String).join("\n");
  if (value === null || value === undefined) return "";
  return String(value);
}

function metadataValueFromInput(original: unknown, value: string): unknown {
  if (Array.isArray(original)) {
    const lines = value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (original.every((item) => typeof item === "number")) {
      const numbers = lines.map(Number);
      return numbers.every(Number.isFinite) ? numbers : lines;
    }
    return lines;
  }
  if (typeof original === "number") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : value;
  }
  if (typeof original === "boolean") return value.toLowerCase() === "true";
  return value;
}

function ProposalCard({
  proposal,
  onUpdated,
  reviewEnabled,
}: {
  proposal: AmendmentProposal;
  onUpdated: (proposal: AmendmentProposal) => void;
  reviewEnabled: boolean;
}) {
  const [deciding, setDeciding] = useState(false);
  const fallbackChange = useMemo<AmendmentProposalChunkChange>(
    () => ({
      old_chunk_id: proposal.old_chunk_id,
      old_chunk_snapshot: proposal.old_chunk_snapshot,
      new_chunk_draft: proposal.new_chunk_draft,
      instruction_indices: proposal.instruction_indices,
      instruction_texts: proposal.instruction_texts,
      match_confidence: proposal.match_confidence,
      match_rationale: proposal.match_rationale,
      date_rationale: proposal.date_rationale,
    }),
    [proposal],
  );
  const proposalChanges = useMemo<AmendmentProposalChunkChange[]>(() => {
    const storedChanges = proposal.chunk_changes ?? [];
    return storedChanges.length > 1 ? storedChanges : [fallbackChange];
  }, [fallbackChange, proposal.chunk_changes]);
  const [activeChangeIndex, setActiveChangeIndex] = useState(0);
  const [drafts, setDrafts] = useState<Record<string, unknown>[]>(() =>
    proposalChanges.map((change) => cloneDraft(change.new_chunk_draft)),
  );
  const draft = drafts[activeChangeIndex] ?? {};
  const activeChange = proposalChanges[activeChangeIndex] ?? fallbackChange;
  const validationError = useMemo(() => {
    const invalid = drafts
      .map((item, index) => ({ index, error: draftValidationError(item) }))
      .find((item) => item.error !== null);
    return invalid ? `Chunk ${invalid.index + 1}: ${invalid.error}` : null;
  }, [drafts]);

  useEffect(() => {
    if (proposal.status !== "pending") {
      setDrafts(
        proposalChanges.map((change) => cloneDraft(change.new_chunk_draft)),
      );
      setActiveChangeIndex(0);
    }
  }, [proposal.status, proposal.updated_at, proposalChanges]);

  const handleApprove = useCallback(async () => {
    if (validationError !== null) return;
    setDeciding(true);
    try {
      const reviewedChanges = proposalChanges.map((change, index) => ({
        ...change,
        new_chunk_draft: drafts[index] ?? change.new_chunk_draft,
      }));
      const primaryDraft = drafts[0];
      if (primaryDraft === undefined) {
        throw new Error("Proposal has no editable chunk draft.");
      }
      const queuedProposal =
        proposalChanges.length > 1
          ? await approveProposal(proposal.id, primaryDraft, reviewedChanges)
          : await approveProposal(proposal.id, primaryDraft);
      onUpdated(queuedProposal);
      toast.info("Approval queued. Indexing will continue in the background.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Approve failed.");
    } finally {
      setDeciding(false);
    }
  }, [proposal.id, drafts, proposalChanges, validationError, onUpdated]);

  const handleReject = useCallback(async () => {
    setDeciding(true);
    try {
      const rejectedProposal = await rejectProposal(proposal.id);
      toast.success("Proposal rejected.");
      onUpdated(rejectedProposal);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Reject failed.");
    } finally {
      setDeciding(false);
    }
  }, [proposal.id, onUpdated]);

  const handleRetryIndexing = useCallback(async () => {
    setDeciding(true);
    try {
      const queuedProposal = await retryProposalIndexing(proposal.id);
      onUpdated(queuedProposal);
      toast.info("Indexing retry queued.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Retry indexing failed.");
    } finally {
      setDeciding(false);
    }
  }, [proposal.id, onUpdated]);

  const isNewChunk = Object.keys(activeChange.old_chunk_snapshot).length === 0;
  const isConsolidated = proposal.instruction_texts.length > 1;
  const effectiveStart = draft.effective_start_date;
  const currentChunk = isNewChunk
    ? null
    : { ...emptyCurrentChunkSnapshot, ...activeChange.old_chunk_snapshot };
  const descendantSnapshots =
    activeChange.old_chunk_snapshot.descendant_snapshots;
  if (currentChunk && Array.isArray(descendantSnapshots)) {
    currentChunk.text = [
      currentChunk.text,
      ...descendantSnapshots.map((item: Record<string, unknown>) =>
        String(item.text ?? ""),
      ),
    ].join("\n\n");
  }
  const afterChunk: Record<string, unknown> = {
    id: "Generated on approval",
    user_file_id: draft.user_file_id,
    position: draft.position,
    text: draft.text,
    chunk_type: draft.chunk_type,
    heading_path: draft.heading_path,
    metadata: draft.metadata,
    validity_start_date: draft.effective_start_date,
    validity_end_date: draft.effective_end_date,
    status: "active",
    source: "amendment",
    supersedes_chunk_id: activeChange.old_chunk_id,
    superseded_by_chunk_id: null,
    created_at: "Generated on approval",
    updated_at: "Generated on approval",
  };
  const readOnly = proposal.status !== "pending";

  const updateDraftField = (key: string, value: unknown) => {
    setDrafts((current) =>
      current.map((item, index) =>
        index === activeChangeIndex ? { ...item, [key]: value } : item,
      ),
    );
  };

  const renderAfterValue = (key: string, value: unknown) => {
    if (key === "user_file_id" || key === "position") {
      return (
        <InputTypeIn
          aria-label={`After ${key}`}
          value={value === null || value === undefined ? "" : String(value)}
          variant="readOnly"
        />
      );
    }
    if (key === "text") {
      return (
        <InputTextArea
          aria-label="After text"
          value={typeof draft.text === "string" ? draft.text : ""}
          onChange={(event) => updateDraftField("text", event.target.value)}
          variant={readOnly ? "readOnly" : "primary"}
          rows={5}
          maxRows={14}
          autoResize
        />
      );
    }
    if (key === "chunk_type") {
      return (
        <InputTypeIn
          aria-label="After chunk_type"
          value={typeof draft.chunk_type === "string" ? draft.chunk_type : ""}
          onChange={(event) =>
            updateDraftField("chunk_type", event.target.value || null)
          }
          variant={readOnly ? "readOnly" : "primary"}
        />
      );
    }
    if (key === "heading_path") {
      const headingPath = Array.isArray(draft.heading_path)
        ? draft.heading_path.map(String)
        : [];
      return (
        <div className="flex flex-col gap-1.5">
          <InputTextArea
            aria-label="After heading_path"
            value={headingPath.join("\n")}
            onChange={(event) =>
              updateDraftField(
                "heading_path",
                event.target.value
                  .split("\n")
                  .map((heading) => heading.trim())
                  .filter(Boolean),
              )
            }
            variant={readOnly ? "readOnly" : "primary"}
            rows={Math.max(2, Math.min(headingPath.length, 6))}
            maxRows={8}
            autoResize
          />
          <Text font="secondary-body" color="text-03">
            One heading per line.
          </Text>
        </div>
      );
    }
    if (key === "metadata") {
      const metadata = isRecord(draft.metadata) ? draft.metadata : {};
      const leaves = metadataLeaves(metadata);
      if (leaves.length === 0) {
        return <span className="text-sm text-text-03">Empty</span>;
      }
      return (
        <div className="flex flex-col gap-2">
          {leaves.map((leaf) => {
            const metadataKey = leaf.path.join(".");
            const multiline = Array.isArray(leaf.value);
            const sharedProps = {
              "aria-label": `After metadata ${metadataKey}`,
              value: metadataInputValue(leaf.value),
              onChange: (
                event: React.ChangeEvent<
                  HTMLInputElement | HTMLTextAreaElement
                >,
              ) => {
                const nextMetadata = updateNestedValue(
                  metadata,
                  leaf.path,
                  metadataValueFromInput(leaf.value, event.target.value),
                );
                updateDraftField("metadata", nextMetadata);
              },
              variant: readOnly ? ("readOnly" as const) : ("primary" as const),
            };
            return (
              <div key={metadataKey} className="flex flex-col gap-1">
                <span className="break-words font-mono text-xs text-text-03">
                  {metadataKey}
                </span>
                {multiline ? (
                  <InputTextArea
                    {...sharedProps}
                    rows={2}
                    maxRows={6}
                    autoResize
                  />
                ) : (
                  <InputTypeIn {...sharedProps} />
                )}
              </div>
            );
          })}
        </div>
      );
    }
    if (key === "validity_start_date" || key === "validity_end_date") {
      const draftKey =
        key === "validity_start_date"
          ? "effective_start_date"
          : "effective_end_date";
      const dateValue = draft[draftKey];
      return (
        <InputTypeIn
          aria-label={`After ${key}`}
          value={typeof dateValue === "string" ? dateValue : ""}
          placeholder="YYYY-MM-DD"
          onChange={(event) =>
            updateDraftField(draftKey, event.target.value || null)
          }
          variant={
            readOnly ? "readOnly" : isIsoDate(dateValue) ? "primary" : "error"
          }
        />
      );
    }
    return <ReadOnlyFieldValue value={value} />;
  };

  return (
    <div
      className="rounded-lg border border-border-02 p-4 flex flex-col gap-3"
      role="article"
      aria-label="Amendment proposal"
    >
      <div className="flex items-start justify-between gap-2">
        {isConsolidated ? (
          <div className="flex flex-col gap-2">
            <Tag
              title={`${proposal.instruction_texts.length} consolidated changes`}
            />
            <ol className="list-decimal list-inside flex flex-col gap-1">
              {proposal.instruction_texts.map((instruction, index) => (
                <Text
                  key={`${proposal.id}-${index}`}
                  font="main-ui-body"
                  color="text-05"
                  as="li"
                >
                  {instruction}
                </Text>
              ))}
            </ol>
          </div>
        ) : (
          <Text font="main-ui-body" color="text-05" as="p">
            {proposal.instruction_text}
          </Text>
        )}
        <div className="flex items-center gap-2 shrink-0">
          {isNewChunk && <Tag title="New article" />}
          {proposal.duplicate_target && <Tag title="Duplicate target" />}
          <Tag title={proposal.status} />
        </div>
      </div>

      {proposalChanges.length > 1 && (
        <div
          className="flex items-center gap-2"
          role="navigation"
          aria-label="Changed chunks"
        >
          <Button
            icon={SvgChevronLeft}
            prominence="tertiary"
            size="sm"
            aria-label="Previous changed chunk"
            onClick={() =>
              setActiveChangeIndex((current) => Math.max(0, current - 1))
            }
            disabled={activeChangeIndex === 0}
          />
          <Text font="secondary-action" color="text-03">
            {`Chunk ${activeChangeIndex + 1} / ${proposalChanges.length}`}
          </Text>
          <Button
            icon={SvgChevronRight}
            prominence="tertiary"
            size="sm"
            aria-label="Next changed chunk"
            onClick={() =>
              setActiveChangeIndex((current) =>
                Math.min(proposalChanges.length - 1, current + 1),
              )
            }
            disabled={activeChangeIndex === proposalChanges.length - 1}
          />
          <Tag title="Approved together" />
        </div>
      )}

      {proposal.match_rationale && (
        <Text font="secondary-body" color="text-03">
          {`Match: ${proposal.match_rationale}${
            proposal.match_confidence != null
              ? ` (confidence ${(proposal.match_confidence * 100).toFixed(0)}%)`
              : ""
          }`}
        </Text>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <FieldTable title="Before" fields={currentChunk} />
        <FieldTable
          title="After"
          fields={afterChunk}
          renderValue={renderAfterValue}
        />
      </div>

      <Text font="secondary-body" color="text-03">
        user_file_id and position identify the target and cannot be changed.
      </Text>
      {validationError && (
        <Text font="secondary-body" color="text-05">
          {validationError}
        </Text>
      )}

      {!isNewChunk && (
        <Text font="secondary-body" color="text-03">
          {`On approval, the current chunk closes at ${
            typeof effectiveStart === "string" && effectiveStart
              ? effectiveStart
              : "the approval date (effective_start_date is null)"
          }. A null effective_end_date means the proposed replacement — including a (Mülga) marker — remains effective indefinitely.`}
        </Text>
      )}

      {proposal.date_rationale && (
        <Text font="secondary-body" color="text-03">
          {`Original date analysis: ${proposal.date_rationale}`}
        </Text>
      )}

      {proposal.status === "approving" && (
        <div
          role="status"
          className="rounded-08 border border-border-02 bg-status-info-01 p-3"
        >
          <Text font="main-ui-action" color="text-05">
            {proposal.approval_execution?.message ||
              "Approval is running in the background."}
          </Text>
        </div>
      )}

      {proposal.status === "approved" && (
        <div
          role="status"
          className="rounded-08 border border-status-success-02 bg-status-success-01 p-3"
        >
          <Text font="main-ui-action" color="status-success-05">
            Success — this proposal was approved and indexed.
          </Text>
        </div>
      )}

      {proposal.status === "pending" && proposal.approval_error && (
        <div
          role="alert"
          className="rounded-08 border border-status-error-02 bg-status-error-01 p-3"
        >
          <Text font="main-ui-action" color="status-error-05">
            {proposal.approval_error}
          </Text>
        </div>
      )}

      {proposal.status === "approval_failed" && (
        <div className="flex flex-col gap-2">
          <div
            role="alert"
            className="rounded-08 border border-status-error-02 bg-status-error-01 p-3"
          >
            <Text font="main-ui-action" color="status-error-05">
              {proposal.approval_error ||
                "Indexing failed. The approval was not published."}
            </Text>
          </div>
          <div className="flex justify-end">
            <Button
              onClick={() => void handleRetryIndexing()}
              disabled={deciding || !reviewEnabled}
            >
              Retry indexing
            </Button>
          </div>
        </div>
      )}

      {proposal.status === "pending" && (
        <div className="flex gap-2 justify-end">
          <Button
            variant="danger"
            prominence="secondary"
            onClick={() => void handleReject()}
            disabled={deciding || !reviewEnabled}
          >
            Reject
          </Button>
          <Button
            onClick={() => void handleApprove()}
            disabled={deciding || !reviewEnabled || validationError !== null}
          >
            {proposalChanges.length > 1
              ? `Approve all ${proposalChanges.length} chunks`
              : "Approve"}
          </Button>
        </div>
      )}
    </div>
  );
}

export default function AmendmentsPage() {
  const { documentSets } = useDocumentSets();
  const [annexCapabilities, setAnnexCapabilities] =
    useState<AnnexCapabilities | null>(null);
  const [selectedDocumentSetId, setSelectedDocumentSetId] = useState<
    string | null
  >(null);
  const [rawText, setRawText] = useState("");
  const [sourceMode, setSourceMode] = useState<AmendmentSourceMode>("text");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [extractedSourceIdentity, setExtractedSourceIdentity] = useState<
    string | null
  >(null);
  const [extracting, setExtracting] = useState(false);
  const [sourcePackage, setSourcePackage] =
    useState<AmendmentSourcePackage | null>(null);
  const [sourcePackageIdentity, setSourcePackageIdentity] = useState<
    string | null
  >(null);
  const [retryingSourcePackage, setRetryingSourcePackage] = useState(false);
  const [sourcePackagePollError, setSourcePackagePollError] = useState<
    string | null
  >(null);
  const sourceRequestTokenRef = useRef<string | null>(null);
  const sourceDraftRef = useRef<{ identity: string; text: string } | null>(
    null,
  );
  const sourceFileInputRef = useRef<HTMLInputElement>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [pollRevision, setPollRevision] = useState(0);

  const [batches, setBatches] = useState<AmendmentBatch[]>([]);
  const [selectedBatchId, setSelectedBatchId] = useState<number | null>(null);
  const [proposals, setProposals] = useState<AmendmentProposal[]>([]);
  const [annexReviews, setAnnexReviews] = useState<AnnexReview[]>([]);
  const [unmatched, setUnmatched] = useState<string[]>([]);
  const [analysisLog, setAnalysisLog] = useState<AmendmentAnalysisLogEntry[]>(
    [],
  );

  const annexEnabled =
    annexCapabilities?.enabled === true &&
    annexCapabilities.grouped_review &&
    annexCapabilities.immutable_review_revisions &&
    annexCapabilities.asynchronous_source_preparation &&
    annexCapabilities.publication_requires_verified_index;

  useEffect(() => {
    let cancelled = false;
    void getAnnexCapabilities()
      .then((capabilities) => {
        if (!cancelled) setAnnexCapabilities(capabilities);
      })
      .catch(() => {
        if (!cancelled) setAnnexCapabilities(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshBatches = useCallback(async (documentSetId: number) => {
    const result = await listAmendmentBatches(documentSetId);
    setBatches(result);
    return result;
  }, []);

  const updateProposal = useCallback((updatedProposal: AmendmentProposal) => {
    setProposals((current) =>
      current.map((proposal) =>
        proposal.id === updatedProposal.id ? updatedProposal : proposal,
      ),
    );
  }, []);

  useEffect(() => {
    if (!selectedDocumentSetId) return;
    void refreshBatches(Number(selectedDocumentSetId));
  }, [selectedDocumentSetId, refreshBatches]);

  useEffect(() => {
    if (selectedBatchId === null) {
      setProposals([]);
      setAnnexReviews([]);
      setUnmatched([]);
      setAnalysisLog([]);
      return;
    }

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let pollErrorReported = false;
    let pollDelayMs = 2000;

    const poll = async () => {
      try {
        const result = await getAmendmentAnalysis(selectedBatchId);
        if (cancelled) return;
        pollErrorReported = false;
        pollDelayMs = 2000;

        setBatches((current) => {
          const exists = current.some((batch) => batch.id === result.batch.id);
          return exists
            ? current.map((batch) =>
                batch.id === result.batch.id ? result.batch : batch,
              )
            : [result.batch, ...current];
        });
        setProposals(result.proposals);
        setAnnexReviews(result.annex_groups ?? []);
        setUnmatched(result.unmatched_instructions);
        setAnalysisLog(result.analysis_log ?? []);

        if (
          result.batch.status === "queued" ||
          result.batch.status === "analyzing"
        ) {
          timeoutId = setTimeout(() => void poll(), pollDelayMs);
        }
      } catch (e) {
        if (!cancelled) {
          if (!pollErrorReported) {
            toast.error(
              e instanceof Error ? e.message : "Could not refresh analysis.",
            );
            pollErrorReported = true;
          }
          if (
            e instanceof RegulatoryRequestError &&
            [401, 403, 404].includes(e.status)
          ) {
            return;
          }
          pollDelayMs = Math.min(pollDelayMs * 2, 30_000);
          timeoutId = setTimeout(() => void poll(), pollDelayMs);
        }
      }
    };

    void poll();
    return () => {
      cancelled = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };
  }, [selectedBatchId, pollRevision]);

  const activeAnnexReviewKey = useMemo(
    () =>
      annexReviews
        .filter(
          (review) =>
            ["approving", "preparing", "publishing"].includes(review.status) ||
            ["queued", "running"].includes(review.preparation?.status ?? ""),
        )
        .map((review) => `${review.id}:${review.status}`)
        .sort()
        .join(","),
    [annexReviews],
  );

  useEffect(() => {
    if (!annexEnabled || selectedBatchId === null || !activeAnnexReviewKey) {
      return;
    }
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let delayMs = 1500;
    const pollReviews = async () => {
      try {
        const reviews = await listAnnexReviews(selectedBatchId);
        if (cancelled) return;
        setAnnexReviews(reviews);
        delayMs = 1500;
        if (
          reviews.some(
            (review) =>
              ["approving", "preparing", "publishing"].includes(
                review.status,
              ) ||
              ["queued", "running"].includes(review.preparation?.status ?? ""),
          )
        ) {
          timeoutId = setTimeout(() => void pollReviews(), delayMs);
        }
      } catch {
        if (cancelled) return;
        delayMs = Math.min(delayMs * 2, 30_000);
        timeoutId = setTimeout(() => void pollReviews(), delayMs);
      }
    };
    timeoutId = setTimeout(() => void pollReviews(), delayMs);
    return () => {
      cancelled = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };
  }, [activeAnnexReviewKey, annexEnabled, selectedBatchId]);

  const updateAnnexReview = useCallback((updatedReview: AnnexReview) => {
    setAnnexReviews((current) => {
      const others = current.filter(
        (review) => review.logical_group_id !== updatedReview.logical_group_id,
      );
      return [...others, updatedReview].sort(
        (left, right) =>
          (left.review_payload.instruction_indices[0] ?? 0) -
          (right.review_payload.instruction_indices[0] ?? 0),
      );
    });
  }, []);

  const approvingProposalIdsKey = useMemo(
    () =>
      proposals
        .filter((proposal) => proposal.status === "approving")
        .map((proposal) => proposal.id)
        .sort((left, right) => left - right)
        .join(","),
    [proposals],
  );

  useEffect(() => {
    if (selectedBatchId === null || !approvingProposalIdsKey) return;

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let pollDelayMs = 1500;
    const trackedIds = new Set(
      approvingProposalIdsKey.split(",").map((value) => Number(value)),
    );

    const pollApprovals = async () => {
      try {
        const refreshed = await listAmendmentProposals(selectedBatchId);
        if (cancelled) return;

        const completedCount = refreshed.filter(
          (proposal) =>
            trackedIds.has(proposal.id) && proposal.status === "approved",
        ).length;
        const failedCount = refreshed.filter(
          (proposal) =>
            trackedIds.has(proposal.id) &&
            ["pending", "approval_failed"].includes(proposal.status),
        ).length;
        setProposals(refreshed);

        if (completedCount > 0) {
          toast.success(
            completedCount === 1
              ? "Proposal approved and indexed."
              : `${completedCount} proposals approved and indexed.`,
          );
        }
        if (failedCount > 0) {
          toast.error(
            failedCount === 1
              ? refreshed.find(
                  (proposal) =>
                    trackedIds.has(proposal.id) &&
                    ["pending", "approval_failed"].includes(proposal.status),
                )?.approval_error ||
                  "Approval was interrupted. Open the proposal for details."
              : `${failedCount} approvals were interrupted. Open the proposals for details.`,
          );
        }

        if (
          refreshed.some(
            (proposal) =>
              trackedIds.has(proposal.id) && proposal.status === "approving",
          )
        ) {
          pollDelayMs = 1500;
          timeoutId = setTimeout(() => void pollApprovals(), pollDelayMs);
        }
      } catch {
        if (cancelled) return;
        pollDelayMs = Math.min(pollDelayMs * 2, 30_000);
        timeoutId = setTimeout(() => void pollApprovals(), pollDelayMs);
      }
    };

    timeoutId = setTimeout(() => void pollApprovals(), pollDelayMs);
    return () => {
      cancelled = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };
  }, [selectedBatchId, approvingProposalIdsKey]);

  const currentSourceIdentity = sourceIdentity(
    sourceMode,
    sourceUrl,
    sourceFile,
    rawText,
  );
  // Pasted text is already the amendment text: there is nothing to download,
  // transcribe, or freeze before analysis can read it.
  const hasCurrentSourceExtraction =
    sourceMode === "text" ||
    (annexEnabled
      ? sourcePackage?.status === "ready" &&
        currentSourceIdentity !== null &&
        currentSourceIdentity === sourcePackageIdentity
      : currentSourceIdentity !== null &&
        currentSourceIdentity === extractedSourceIdentity);
  const canAnalyze = Boolean(rawText.trim()) && hasCurrentSourceExtraction;
  const sourcePackageId = sourcePackage?.id ?? null;
  const sourcePackageStatus = sourcePackage?.status ?? null;
  const sourcePreparationBusy =
    retryingSourcePackage ||
    (annexEnabled &&
      sourcePackageStatus === "processing" &&
      currentSourceIdentity !== null &&
      currentSourceIdentity === sourcePackageIdentity);

  useEffect(() => {
    if (
      !annexEnabled ||
      !selectedDocumentSetId ||
      !sourcePackageId ||
      !["processing", "ready"].includes(sourcePackageStatus ?? "") ||
      !sourcePackageIdentity
    ) {
      return;
    }
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let pollDelayMs = 1500;
    let pollErrorReported = false;
    const expectedIdentity = sourcePackageIdentity;
    const expectedRequestToken = sourceRequestTokenRef.current;
    const isCurrentRequest = () =>
      !cancelled && sourceRequestTokenRef.current === expectedRequestToken;
    const pollPackage = async () => {
      try {
        if (sourcePackageStatus === "ready") {
          const sourceText = await getAmendmentSourceText(
            Number(selectedDocumentSetId),
            sourcePackageId,
          );
          if (!isCurrentRequest()) return;
          setSourcePackagePollError(null);
          if (sourceText.original_text.trim()) {
            const draft = sourceDraftRef.current;
            setRawText(
              [
                sourceText.original_text,
                draft?.identity === expectedIdentity ? draft.text : "",
              ]
                .filter((text) => text.trim())
                .join("\n\n"),
            );
          }
          sourceDraftRef.current = null;
          setExtractedSourceIdentity(expectedIdentity);
          toast.success("Frozen source package is ready for analysis.");
          return;
        }
        const refreshed = await getAmendmentSourcePackage(
          Number(selectedDocumentSetId),
          sourcePackageId,
        );
        if (!isCurrentRequest()) return;
        pollErrorReported = false;
        pollDelayMs = 1500;
        setSourcePackagePollError(null);
        setSourcePackage(refreshed);
        if (refreshed.status === "processing") {
          timeoutId = setTimeout(() => void pollPackage(), pollDelayMs);
        }
      } catch (error) {
        if (!isCurrentRequest()) return;
        if (!pollErrorReported) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Could not refresh source preparation.",
          );
          pollErrorReported = true;
        }
        if (
          error instanceof RegulatoryRequestError &&
          [401, 403, 404].includes(error.status)
        ) {
          setSourcePackagePollError(error.message);
          return;
        }
        pollDelayMs = Math.min(pollDelayMs * 2, 30_000);
        timeoutId = setTimeout(() => void pollPackage(), pollDelayMs);
      }
    };
    void pollPackage();
    return () => {
      cancelled = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };
  }, [
    annexEnabled,
    selectedDocumentSetId,
    sourcePackageId,
    sourcePackageIdentity,
    sourcePackageStatus,
  ]);

  const handleSourceModeChange = useCallback((mode: AmendmentSourceMode) => {
    sourceRequestTokenRef.current = null;
    sourceDraftRef.current = null;
    setSourceMode(mode);
    setSourceFile(null);
    if (sourceFileInputRef.current) {
      sourceFileInputRef.current.value = "";
    }
    setRawText("");
    setExtractedSourceIdentity(null);
    setSourcePackage(null);
    setSourcePackageIdentity(null);
    setExtracting(false);
    setAnalyzing(false);
    setRetryingSourcePackage(false);
    setSourcePackagePollError(null);
  }, []);

  const handleExtract = useCallback(async () => {
    if (sourcePreparationBusy) return;
    const identity = sourceIdentity(sourceMode, sourceUrl, sourceFile, rawText);
    if (!identity) {
      toast.error(
        sourceMode === "url"
          ? "Enter an amendment source URL."
          : "Choose a supported annex source file.",
      );
      return;
    }

    const requestToken = sourceRequestIdentity();
    sourceRequestTokenRef.current = requestToken;
    setSourcePackagePollError(null);
    setExtracting(true);
    try {
      if (annexEnabled && selectedDocumentSetId) {
        const prepared =
          sourceMode === "url"
            ? await createAmendmentSourcePackage(
                Number(selectedDocumentSetId),
                requestToken,
                { url: sourceUrl.trim() },
              )
            : await uploadAmendmentSourcePackage(
                Number(selectedDocumentSetId),
                requestToken,
                sourceFile as File,
              );
        if (sourceRequestTokenRef.current !== requestToken) return;
        setSourcePackageIdentity(identity);
        setSourcePackage(prepared);
        toast.info(
          "Source preparation started. Status will update automatically.",
        );
        return;
      }
      const result =
        sourceMode === "url"
          ? await extractAmendmentUrl(sourceUrl.trim())
          : sourceMode === "pdf"
            ? await extractAmendmentPdf(sourceFile as File)
            : await extractAmendmentDocx(sourceFile as File);
      if (sourceRequestTokenRef.current !== requestToken) return;
      setRawText(result.text);
      setExtractedSourceIdentity(identity);
      toast.success(
        `Extracted text from ${result.display_name}. Review it before analysis.`,
      );
    } catch (e) {
      if (sourceRequestTokenRef.current === requestToken) {
        toast.error(
          e instanceof Error ? e.message : "Source extraction failed.",
        );
      }
    } finally {
      if (sourceRequestTokenRef.current === requestToken) {
        setExtracting(false);
      }
    }
  }, [
    annexEnabled,
    rawText,
    selectedDocumentSetId,
    sourceFile,
    sourceMode,
    sourcePreparationBusy,
    sourceUrl,
  ]);

  const handleAnalyze = useCallback(async () => {
    if (!selectedDocumentSetId || !rawText.trim() || sourcePreparationBusy)
      return;
    setAnalyzing(true);
    try {
      if (!canAnalyze) return;
      const result =
        annexEnabled && sourceMode !== "text"
          ? await analyzeAmendment(
              Number(selectedDocumentSetId),
              rawText,
              sourcePackage?.id,
            )
          : await analyzeAmendment(Number(selectedDocumentSetId), rawText);
      toast.success("Analysis queued. Progress will update automatically.");
      setRawText("");
      setBatches((current) => [
        result,
        ...current.filter((batch) => batch.id !== result.id),
      ]);
      setSelectedBatchId(result.id);
      setProposals([]);
      setAnnexReviews([]);
      setUnmatched([]);
      setAnalysisLog([]);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  }, [
    annexEnabled,
    canAnalyze,
    rawText,
    selectedDocumentSetId,
    sourceMode,
    sourcePackage?.id,
    sourcePreparationBusy,
  ]);

  const handleSourcePackageRetry = useCallback(async () => {
    if (!selectedDocumentSetId || !sourcePackage || !sourcePackageIdentity) {
      return;
    }
    const documentSetId = Number(selectedDocumentSetId);
    const packageId = sourcePackage.id;
    const requestToken = sourceRequestIdentity();
    sourceRequestTokenRef.current = requestToken;
    setSourcePackagePollError(null);
    setRetryingSourcePackage(true);
    try {
      const retried = await retryAmendmentSourcePackage(
        documentSetId,
        packageId,
      );
      if (sourceRequestTokenRef.current !== requestToken) return;
      setSourcePackage(retried);
      toast.info("Source preparation retry queued.");
    } catch (error) {
      if (sourceRequestTokenRef.current === requestToken) {
        toast.error(
          error instanceof Error ? error.message : "Source retry failed.",
        );
      }
    } finally {
      if (sourceRequestTokenRef.current === requestToken) {
        setRetryingSourcePackage(false);
      }
    }
  }, [selectedDocumentSetId, sourcePackage, sourcePackageIdentity]);

  const handleRetry = useCallback(async () => {
    if (selectedBatchId === null) return;
    setRetrying(true);
    try {
      const batch = await retryAmendmentBatch(selectedBatchId);
      setBatches((current) =>
        current.map((item) => (item.id === batch.id ? batch : item)),
      );
      setUnmatched([]);
      setAnalysisLog([]);
      setPollRevision((revision) => revision + 1);
      toast.success("Analysis queued again from its last checkpoint.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Retry failed.");
    } finally {
      setRetrying(false);
    }
  }, [selectedBatchId]);

  const selectedBatch = useMemo(
    () => batches.find((b) => b.id === selectedBatchId) ?? null,
    [batches, selectedBatchId],
  );

  return (
    <SettingsLayouts.Root width="lg">
      <SettingsLayouts.Header
        icon={SvgHistory}
        title="Updates"
        description="Paste an official amendment/update text scoped to a document set. It will be segmented into atomic changes, matched against the document set's existing chunks, and drafted for your review — nothing is written until you approve each proposal."
      />
      <SettingsLayouts.Body>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Text font="main-ui-action" color="text-04">
              Document Set
            </Text>
            <InputSelect
              value={selectedDocumentSetId ?? ""}
              onValueChange={(value) => {
                sourceRequestTokenRef.current = null;
                sourceDraftRef.current = null;
                setSelectedDocumentSetId(value);
                setSelectedBatchId(null);
                setSourcePackage(null);
                setSourcePackageIdentity(null);
                setExtracting(false);
                setAnalyzing(false);
                setRetryingSourcePackage(false);
                setSourcePackagePollError(null);
              }}
            >
              <InputSelect.Trigger />
              <InputSelect.Content>
                {documentSets.map((documentSet) => (
                  <InputSelect.Item
                    key={documentSet.id}
                    value={String(documentSet.id)}
                  >
                    {documentSet.name}
                  </InputSelect.Item>
                ))}
              </InputSelect.Content>
            </InputSelect>
          </div>

          {selectedDocumentSetId && (
            <>
              <div className="flex flex-col gap-2">
                <Text font="main-ui-action" color="text-04">
                  Amendment source
                </Text>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    prominence={sourceMode === "text" ? "primary" : "secondary"}
                    onClick={() => handleSourceModeChange("text")}
                  >
                    Text
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    prominence={sourceMode === "url" ? "primary" : "secondary"}
                    onClick={() => handleSourceModeChange("url")}
                  >
                    URL
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    prominence={sourceMode === "pdf" ? "primary" : "secondary"}
                    onClick={() => handleSourceModeChange("pdf")}
                  >
                    PDF
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    prominence={sourceMode === "docx" ? "primary" : "secondary"}
                    onClick={() => handleSourceModeChange("docx")}
                  >
                    Word (.docx)
                  </Button>
                  {annexEnabled &&
                    (["image", "html", "xlsx"] as const).map((mode) => (
                      <Button
                        key={mode}
                        type="button"
                        size="sm"
                        prominence={
                          sourceMode === mode ? "primary" : "secondary"
                        }
                        onClick={() => handleSourceModeChange(mode)}
                      >
                        {sourceModeDetails[mode].button}
                      </Button>
                    ))}
                </div>

                {sourceMode === "url" && (
                  <div className="flex gap-2">
                    <InputTypeIn
                      aria-label="Amendment source URL"
                      value={sourceUrl}
                      onChange={(event) => {
                        sourceRequestTokenRef.current = null;
                        sourceDraftRef.current = null;
                        setSourceUrl(event.target.value);
                        setRawText("");
                        setExtractedSourceIdentity(null);
                        setSourcePackage(null);
                        setSourcePackageIdentity(null);
                        setExtracting(false);
                        setRetryingSourcePackage(false);
                        setSourcePackagePollError(null);
                      }}
                      placeholder="https://www.resmigazete.gov.tr/..."
                    />
                    <Button
                      type="button"
                      onClick={() => void handleExtract()}
                      disabled={
                        extracting || sourcePreparationBusy || !sourceUrl.trim()
                      }
                    >
                      {extracting
                        ? annexEnabled
                          ? "Preparing…"
                          : "Extracting…"
                        : annexEnabled
                          ? "Prepare source"
                          : "Extract"}
                    </Button>
                  </div>
                )}

                {FILE_SOURCE_MODES.has(sourceMode) && (
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      ref={sourceFileInputRef}
                      aria-label={
                        sourceModeDetails[
                          sourceMode as Exclude<
                            AmendmentSourceMode,
                            "text" | "url"
                          >
                        ].label
                      }
                      className="hidden"
                      type="file"
                      accept={
                        sourceModeDetails[
                          sourceMode as Exclude<
                            AmendmentSourceMode,
                            "text" | "url"
                          >
                        ].accept
                      }
                      onChange={(event) => {
                        sourceRequestTokenRef.current = null;
                        sourceDraftRef.current = null;
                        setSourceFile(event.target.files?.[0] ?? null);
                        setRawText("");
                        setExtractedSourceIdentity(null);
                        setSourcePackage(null);
                        setSourcePackageIdentity(null);
                        setExtracting(false);
                        setRetryingSourcePackage(false);
                        setSourcePackagePollError(null);
                      }}
                    />
                    <Button
                      type="button"
                      prominence="secondary"
                      onClick={() => sourceFileInputRef.current?.click()}
                    >
                      {`${sourceFile ? "Choose another" : "Choose"} ${sourceModeDetails[sourceMode as Exclude<AmendmentSourceMode, "text" | "url">].choose}`}
                    </Button>
                    {sourceFile && (
                      <Text font="main-ui-body" color="text-03">
                        {sourceFile.name}
                      </Text>
                    )}
                    <Button
                      type="button"
                      onClick={() => void handleExtract()}
                      disabled={
                        extracting || sourcePreparationBusy || !sourceFile
                      }
                    >
                      {extracting
                        ? annexEnabled
                          ? "Preparing…"
                          : "Extracting…"
                        : annexEnabled
                          ? "Prepare source"
                          : "Extract"}
                    </Button>
                  </div>
                )}

                {annexEnabled && sourcePackage && (
                  <div className="flex flex-col gap-2 rounded-08 border border-border-02 bg-background-tint-01 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <Text font="main-ui-action" color="text-04">
                        {`Source package ${sourcePackage.status}`}
                      </Text>
                      <Tag title={`${sourcePackage.asset_count} assets`} />
                    </div>
                    {sourcePackage.manifest_sha256 && (
                      <Text as="p" font="secondary-body" color="text-03">
                        {`Immutable manifest ${sourcePackage.manifest_sha256}`}
                      </Text>
                    )}
                    {sourcePackage.issues.map((issue) => (
                      <div
                        key={`${issue.code}-${issue.locator ?? ""}`}
                        className="flex flex-col gap-1"
                      >
                        <Text
                          as="p"
                          font="secondary-body"
                          color="status-error-05"
                        >
                          {`${sourceIssueMessage(issue.code)}${issue.locator ? ` · ${issue.locator}` : ""}`}
                        </Text>
                        {issue.failure_detail && (
                          <Text as="p" font="secondary-body" color="text-03">
                            {issue.failure_detail}
                          </Text>
                        )}
                      </div>
                    ))}
                    {sourcePackagePollError && (
                      <Text as="p" font="main-ui-body" color="status-error-05">
                        {`Source package status cannot be refreshed: ${sourcePackagePollError}`}
                      </Text>
                    )}
                    {["partial", "blocked", "failed"].includes(
                      sourcePackage.status,
                    ) &&
                      sourcePackage.issues.some(
                        (issue) => issue.retryable !== false,
                      ) && (
                        <div className="flex justify-end">
                          <Button
                            size="sm"
                            disabled={retryingSourcePackage}
                            onClick={() => void handleSourcePackageRetry()}
                          >
                            {retryingSourcePackage
                              ? "Retrying…"
                              : "Retry source preparation"}
                          </Button>
                        </div>
                      )}
                  </div>
                )}

                <Text font="secondary-body" color="text-03">
                  {sourceMode === "text"
                    ? "Paste the official amendment/update text below."
                    : "Extracted text remains editable. Change the source and extract again before analysis."}
                </Text>
                <Text font="main-ui-action" color="text-04">
                  Amendment text
                </Text>
                <InputTextArea
                  aria-label="Amendment text"
                  value={rawText}
                  onChange={(event) => {
                    if (!annexEnabled || sourceMode === "text") {
                      sourceRequestTokenRef.current = null;
                    }
                    if (
                      annexEnabled &&
                      sourceMode !== "text" &&
                      currentSourceIdentity !== null &&
                      currentSourceIdentity !== extractedSourceIdentity
                    ) {
                      sourceDraftRef.current = {
                        identity: currentSourceIdentity,
                        text: event.target.value,
                      };
                    }
                    setRawText(event.target.value);
                    if (annexEnabled && sourceMode === "text") {
                      setSourcePackage(null);
                      setSourcePackageIdentity(null);
                      setAnalyzing(false);
                      setRetryingSourcePackage(false);
                      setSourcePackagePollError(null);
                    }
                  }}
                  rows={8}
                  autoResize
                  maxRows={20}
                  placeholder="Paste the official amendment/update text here..."
                />
                <div className="flex justify-end">
                  <Button
                    onClick={() => void handleAnalyze()}
                    disabled={
                      analyzing ||
                      sourcePreparationBusy ||
                      !rawText.trim() ||
                      !canAnalyze
                    }
                  >
                    {analyzing ? "Analyzing…" : "Analyze"}
                  </Button>
                </div>
              </div>

              {batches.length > 0 && (
                <div className="flex flex-col gap-2">
                  <Text font="main-ui-action" color="text-04">
                    Past batches
                  </Text>
                  <div className="flex flex-wrap gap-2">
                    {batches.map((batch) => (
                      <Button
                        key={batch.id}
                        prominence={
                          batch.id === selectedBatchId ? "primary" : "secondary"
                        }
                        size="sm"
                        onClick={() => setSelectedBatchId(batch.id)}
                      >
                        {`Batch #${batch.id} (${batch.status})`}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              {selectedBatch && (
                <div className="flex flex-col gap-3">
                  {(selectedBatch.status === "queued" ||
                    selectedBatch.status === "analyzing") && (
                    <Text font="main-ui-body" color="text-05" as="p">
                      {analysisProgressLabel(selectedBatch)}
                    </Text>
                  )}
                  {selectedBatch.status === "paused" && (
                    <div className="flex items-center justify-between gap-3">
                      <Text font="main-ui-body" color="text-05" as="p">
                        Analysis paused to protect memory; saved progress is
                        preserved. Retry after capacity is available.
                      </Text>
                      <Button
                        size="sm"
                        onClick={() => void handleRetry()}
                        disabled={retrying}
                      >
                        {retrying ? "Retrying…" : "Retry"}
                      </Button>
                    </div>
                  )}
                  {selectedBatch.status === "failed" && (
                    <div className="flex items-center justify-between gap-3">
                      <Text font="main-ui-body" color="text-05" as="p">
                        {`Analysis failed: ${selectedBatch.error_message ?? "unknown error"}`}
                      </Text>
                      <Button
                        size="sm"
                        onClick={() => void handleRetry()}
                        disabled={retrying}
                      >
                        {retrying ? "Retrying…" : "Retry"}
                      </Button>
                    </div>
                  )}

                  {selectedBatch.status === "analyzed" &&
                    selectedBatch.instruction_count === 0 && (
                      <Text font="main-ui-body" color="text-03" as="p">
                        No update instructions were detected in this text. Paste
                        an amendment/update, or add context (e.g. "this is the
                        new version of X") if the pasted content is a
                        replacement without formal amendment language.
                      </Text>
                    )}

                  {analysisLog.length > 0 && (
                    <details className="rounded-lg border border-border-02 p-3">
                      <summary className="cursor-pointer">
                        <Text font="main-ui-action" color="text-04">
                          {`Analysis log (${analysisLog.length} steps)`}
                        </Text>
                      </summary>
                      <div className="mt-2 flex max-h-96 flex-col gap-1 overflow-auto">
                        {analysisLog.map((entry, index) => {
                          const { at, step, ...fields } = entry;
                          const detail = Object.entries(fields)
                            .filter(([, value]) => value !== null)
                            .map(
                              ([key, value]) =>
                                `${key}=${JSON.stringify(value)}`,
                            )
                            .join(" ");
                          return (
                            <div
                              key={`${at}-${index}`}
                              className="whitespace-pre-wrap break-words font-mono text-xs text-text-03"
                            >
                              {`${at} ${step} ${detail}`.trim()}
                            </div>
                          );
                        })}
                      </div>
                    </details>
                  )}

                  {unmatched.length > 0 && (
                    <div className="rounded-lg border border-border-02 p-3">
                      <Text font="main-ui-action" color="text-04">
                        Instructions requiring attention
                      </Text>
                      {selectedBatch?.status === "analyzed" && (
                        <Button
                          prominence="secondary"
                          disabled={retrying}
                          onClick={() => void handleRetry()}
                        >
                          Retry unresolved instructions
                        </Button>
                      )}
                      {unmatched.map((instr, i) => (
                        <div key={i} className="whitespace-pre-wrap">
                          <Text font="main-ui-body" color="text-03" as="p">
                            {instr}
                          </Text>
                        </div>
                      ))}
                    </div>
                  )}

                  {annexEnabled && annexReviews.length > 0 && (
                    <div className="flex flex-col gap-3">
                      <Text as="h2" font="heading-h3" color="text-05">
                        Annex chunk changes
                      </Text>
                      <Text as="p" font="secondary-body" color="text-03">
                        Compare changed chunks and prepare the changes you want
                        to approve.
                      </Text>
                      {annexReviews.map((review) => (
                        <AnnexChunkReview
                          key={review.id}
                          review={review}
                          onUpdated={updateAnnexReview}
                        />
                      ))}
                    </div>
                  )}

                  {proposals.map((proposal) => (
                    <ProposalCard
                      key={proposal.id}
                      proposal={proposal}
                      onUpdated={updateProposal}
                      reviewEnabled={selectedBatch.status === "analyzed"}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
