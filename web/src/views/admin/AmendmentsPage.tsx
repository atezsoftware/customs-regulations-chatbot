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
import { useDocumentSets } from "@/lib/hooks/useDocumentSets";
import {
  AmendmentBatch,
  AmendmentProposal,
  RegulatoryRequestError,
  analyzeAmendment,
  approveProposal,
  extractAmendmentDocx,
  extractAmendmentPdf,
  extractAmendmentUrl,
  getAmendmentAnalysis,
  listAmendmentBatches,
  listAmendmentProposals,
  rejectProposal,
  retryAmendmentBatch,
} from "@/lib/regulatory/amendments";

type AmendmentSourceMode = "text" | "url" | "pdf" | "docx";

function sourceIdentity(
  mode: AmendmentSourceMode,
  url: string,
  file: File | null
) {
  if (mode === "url") return url.trim() ? `url:${url.trim()}` : null;
  if ((mode === "pdf" || mode === "docx") && file) {
    return `${mode}:${file.name}:${file.size}:${file.lastModified}`;
  }
  return null;
}

function analysisProgressLabel(batch: AmendmentBatch) {
  if (batch.stage === "segmenting") return "Segmenting amendment…";
  if (batch.stage === "finalizing") return "Finalizing analysis…";
  if (batch.instruction_count > 0) {
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
      ])
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
      ([, child]) => child !== null && child !== undefined
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
          (fields[key] !== null && fields[key] !== undefined)
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
  prefix: string[] = []
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
  value: unknown
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
  const [draft, setDraft] = useState<Record<string, unknown>>(() =>
    cloneDraft(proposal.new_chunk_draft)
  );
  const validationError = useMemo(() => draftValidationError(draft), [draft]);

  useEffect(() => {
    if (proposal.status !== "pending") {
      setDraft(cloneDraft(proposal.new_chunk_draft));
    }
  }, [proposal.status, proposal.updated_at, proposal.new_chunk_draft]);

  const handleApprove = useCallback(async () => {
    if (validationError !== null) return;
    setDeciding(true);
    try {
      const queuedProposal = await approveProposal(proposal.id, draft);
      onUpdated(queuedProposal);
      toast.info("Approval queued. Indexing will continue in the background.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Approve failed.");
    } finally {
      setDeciding(false);
    }
  }, [proposal.id, draft, validationError, onUpdated]);

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

  const isNewChunk = Object.keys(proposal.old_chunk_snapshot).length === 0;
  const isConsolidated = proposal.instruction_texts.length > 1;
  const effectiveStart = draft.effective_start_date;
  const currentChunk = isNewChunk
    ? null
    : { ...emptyCurrentChunkSnapshot, ...proposal.old_chunk_snapshot };
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
    supersedes_chunk_id: proposal.old_chunk_id,
    superseded_by_chunk_id: null,
    created_at: "Generated on approval",
    updated_at: "Generated on approval",
  };
  const readOnly = proposal.status !== "pending";

  const updateDraftField = (key: string, value: unknown) => {
    setDraft((current) => ({ ...current, [key]: value }));
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
                  .filter(Boolean)
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
                event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
              ) => {
                const nextMetadata = updateNestedValue(
                  metadata,
                  leaf.path,
                  metadataValueFromInput(leaf.value, event.target.value)
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
            Approval is running. The updated chunk is being indexed in the
            background.
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
            Approve
          </Button>
        </div>
      )}
    </div>
  );
}

export default function AmendmentsPage() {
  const { documentSets } = useDocumentSets();
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
  const sourceFileInputRef = useRef<HTMLInputElement>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [pollRevision, setPollRevision] = useState(0);

  const [batches, setBatches] = useState<AmendmentBatch[]>([]);
  const [selectedBatchId, setSelectedBatchId] = useState<number | null>(null);
  const [proposals, setProposals] = useState<AmendmentProposal[]>([]);
  const [unmatched, setUnmatched] = useState<string[]>([]);

  const refreshBatches = useCallback(async (documentSetId: number) => {
    const result = await listAmendmentBatches(documentSetId);
    setBatches(result);
    return result;
  }, []);

  const updateProposal = useCallback((updatedProposal: AmendmentProposal) => {
    setProposals((current) =>
      current.map((proposal) =>
        proposal.id === updatedProposal.id ? updatedProposal : proposal
      )
    );
  }, []);

  useEffect(() => {
    if (!selectedDocumentSetId) return;
    void refreshBatches(Number(selectedDocumentSetId));
  }, [selectedDocumentSetId, refreshBatches]);

  useEffect(() => {
    if (selectedBatchId === null) {
      setProposals([]);
      setUnmatched([]);
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
                batch.id === result.batch.id ? result.batch : batch
              )
            : [result.batch, ...current];
        });
        setProposals(result.proposals);
        setUnmatched(result.unmatched_instructions);

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
              e instanceof Error ? e.message : "Could not refresh analysis."
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

  const approvingProposalIdsKey = useMemo(
    () =>
      proposals
        .filter((proposal) => proposal.status === "approving")
        .map((proposal) => proposal.id)
        .sort((left, right) => left - right)
        .join(","),
    [proposals]
  );

  useEffect(() => {
    if (selectedBatchId === null || !approvingProposalIdsKey) return;

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let pollDelayMs = 1500;
    const trackedIds = new Set(
      approvingProposalIdsKey.split(",").map((value) => Number(value))
    );

    const pollApprovals = async () => {
      try {
        const refreshed = await listAmendmentProposals(selectedBatchId);
        if (cancelled) return;

        const completedCount = refreshed.filter(
          (proposal) =>
            trackedIds.has(proposal.id) && proposal.status === "approved"
        ).length;
        const failedCount = refreshed.filter(
          (proposal) =>
            trackedIds.has(proposal.id) && proposal.status === "pending"
        ).length;
        setProposals(refreshed);

        if (completedCount > 0) {
          toast.success(
            completedCount === 1
              ? "Proposal approved and indexed."
              : `${completedCount} proposals approved and indexed.`
          );
        }
        if (failedCount > 0) {
          toast.error(
            failedCount === 1
              ? "Approval failed during indexing. Review the proposal and try again."
              : `${failedCount} approvals failed during indexing. Review them and try again.`
          );
        }

        if (
          refreshed.some(
            (proposal) =>
              trackedIds.has(proposal.id) && proposal.status === "approving"
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
    sourceFile
  );
  const hasCurrentSourceExtraction =
    sourceMode === "text" ||
    (currentSourceIdentity !== null &&
      currentSourceIdentity === extractedSourceIdentity);
  const canAnalyze = Boolean(rawText.trim()) && hasCurrentSourceExtraction;

  const handleSourceModeChange = useCallback((mode: AmendmentSourceMode) => {
    setSourceMode(mode);
    setSourceFile(null);
    if (sourceFileInputRef.current) {
      sourceFileInputRef.current.value = "";
    }
    setRawText("");
    setExtractedSourceIdentity(null);
  }, []);

  const handleExtract = useCallback(async () => {
    const identity = sourceIdentity(sourceMode, sourceUrl, sourceFile);
    if (!identity) {
      toast.error(
        sourceMode === "url"
          ? "Enter an amendment source URL."
          : sourceMode === "pdf"
            ? "Choose a PDF file."
            : "Choose a Word .docx file."
      );
      return;
    }

    setExtracting(true);
    try {
      const result =
        sourceMode === "url"
          ? await extractAmendmentUrl(sourceUrl.trim())
          : sourceMode === "pdf"
            ? await extractAmendmentPdf(sourceFile as File)
            : await extractAmendmentDocx(sourceFile as File);
      setRawText(result.text);
      setExtractedSourceIdentity(identity);
      toast.success(
        `Extracted text from ${result.display_name}. Review it before analysis.`
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Source extraction failed.");
    } finally {
      setExtracting(false);
    }
  }, [sourceFile, sourceMode, sourceUrl]);

  const handleAnalyze = useCallback(async () => {
    if (!selectedDocumentSetId || !canAnalyze) return;
    setAnalyzing(true);
    try {
      const result = await analyzeAmendment(
        Number(selectedDocumentSetId),
        rawText
      );
      toast.success("Analysis queued. Progress will update automatically.");
      setRawText("");
      setBatches((current) => [
        result,
        ...current.filter((batch) => batch.id !== result.id),
      ]);
      setSelectedBatchId(result.id);
      setProposals([]);
      setUnmatched([]);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  }, [selectedDocumentSetId, rawText, canAnalyze]);

  const handleRetry = useCallback(async () => {
    if (selectedBatchId === null) return;
    setRetrying(true);
    try {
      const batch = await retryAmendmentBatch(selectedBatchId);
      setBatches((current) =>
        current.map((item) => (item.id === batch.id ? batch : item))
      );
      setProposals([]);
      setUnmatched([]);
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
    [batches, selectedBatchId]
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
                setSelectedDocumentSetId(value);
                setSelectedBatchId(null);
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
                </div>

                {sourceMode === "url" && (
                  <div className="flex gap-2">
                    <InputTypeIn
                      aria-label="Amendment source URL"
                      value={sourceUrl}
                      onChange={(event) => {
                        setSourceUrl(event.target.value);
                        setRawText("");
                        setExtractedSourceIdentity(null);
                      }}
                      placeholder="https://www.resmigazete.gov.tr/..."
                    />
                    <Button
                      type="button"
                      onClick={() => void handleExtract()}
                      disabled={extracting || !sourceUrl.trim()}
                    >
                      {extracting ? "Extracting…" : "Extract"}
                    </Button>
                  </div>
                )}

                {(sourceMode === "pdf" || sourceMode === "docx") && (
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      ref={sourceFileInputRef}
                      aria-label={
                        sourceMode === "pdf"
                          ? "Amendment source PDF"
                          : "Amendment source Word document"
                      }
                      className="hidden"
                      type="file"
                      accept={
                        sourceMode === "pdf"
                          ? "application/pdf,.pdf"
                          : "application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx"
                      }
                      onChange={(event) => {
                        setSourceFile(event.target.files?.[0] ?? null);
                        setRawText("");
                        setExtractedSourceIdentity(null);
                      }}
                    />
                    <Button
                      type="button"
                      prominence="secondary"
                      onClick={() => sourceFileInputRef.current?.click()}
                    >
                      {sourceMode === "pdf"
                        ? sourceFile
                          ? "Choose another PDF"
                          : "Choose PDF"
                        : sourceFile
                          ? "Choose another Word document"
                          : "Choose Word document"}
                    </Button>
                    {sourceFile && (
                      <Text font="main-ui-body" color="text-03">
                        {sourceFile.name}
                      </Text>
                    )}
                    <Button
                      type="button"
                      onClick={() => void handleExtract()}
                      disabled={extracting || !sourceFile}
                    >
                      {extracting ? "Extracting…" : "Extract"}
                    </Button>
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
                  onChange={(e) => setRawText(e.target.value)}
                  rows={8}
                  autoResize
                  maxRows={20}
                  placeholder="Paste the official amendment/update text here..."
                />
                <div className="flex justify-end">
                  <Button
                    onClick={() => void handleAnalyze()}
                    disabled={analyzing || !canAnalyze}
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

                  {unmatched.length > 0 && (
                    <div className="rounded-lg border border-border-02 p-3">
                      <Text font="main-ui-action" color="text-04">
                        Instructions requiring attention
                      </Text>
                      {unmatched.map((instr, i) => (
                        <div key={i} className="whitespace-pre-wrap">
                          <Text font="main-ui-body" color="text-03" as="p">
                            {instr}
                          </Text>
                        </div>
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
