"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  Button,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import type { IntakeChunkUpdate } from "@/lib/regulatory/documentIntake";
import type { IntakeMetadataEdits } from "@/lib/regulatory/intakeMetadata";
import ChunkFieldsReview, {
  ReadOnlyFieldValue,
} from "@/sections/cards/ChunkFieldsReview";
import IntakeLegalMetadataFields, {
  IntakeChunkSourceDetails,
  IntakeMetadataOrigin,
} from "@/sections/cards/IntakeLegalMetadataFields";

interface IntakeChunkReviewCardProps {
  chunk: RegulatoryChunk;
  editable: boolean;
  disabled: boolean;
  reviewSha256: string | null;
  documentName?: string;
  sourceUrl?: string | null;
  children: ReactNode;
  onSave: (
    draft: IntakeChunkUpdate,
    expectedReviewSha256: string
  ) => Promise<void>;
  onDirtyChange: (chunkId: string, dirty: boolean) => void;
}

function chunkDraft(chunk: RegulatoryChunk): IntakeChunkUpdate {
  return {
    text: chunk.text,
    heading_path: chunk.heading_path,
    chunk_type: chunk.chunk_type ?? null,
    validity_start_date: chunk.validity_start_date ?? null,
    validity_end_date: chunk.validity_end_date ?? null,
  };
}

function validDate(value: string | null): boolean {
  if (!value) return true;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return (
    Number.isFinite(parsed.getTime()) &&
    parsed.toISOString().slice(0, 10) === value
  );
}

export default function IntakeChunkReviewCard({
  chunk,
  editable,
  disabled,
  reviewSha256,
  documentName,
  sourceUrl,
  children,
  onSave,
  onDirtyChange,
}: IntakeChunkReviewCardProps) {
  const [draft, setDraft] = useState(() => chunkDraft(chunk));
  const [baseDraft, setBaseDraft] = useState(() => chunkDraft(chunk));
  const [baseMetadata, setBaseMetadata] = useState(chunk.chunk_metadata);
  const [metadataSession, setMetadataSession] = useState(0);
  const [metadataEdits, setMetadataEdits] = useState<IntakeMetadataEdits>({
    patch: {},
    dirty: false,
    error: null,
  });
  const onMetadataEdits = useCallback(
    (edits: IntakeMetadataEdits) =>
      setMetadataEdits((current) =>
        JSON.stringify(current) === JSON.stringify(edits) ? current : edits
      ),
    []
  );
  const [editReviewSha256, setEditReviewSha256] = useState(reviewSha256);
  const [headingPath, setHeadingPath] = useState(chunk.heading_path.join("\n"));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dirty =
    JSON.stringify(draft) !== JSON.stringify(baseDraft) ||
    headingPath !== baseDraft.heading_path.join("\n") ||
    metadataEdits.dirty;
  useEffect(() => {
    onDirtyChange(chunk.id, dirty);
  }, [chunk.id, dirty, onDirtyChange]);
  useEffect(() => {
    if (dirty) return;
    setDraft(chunkDraft(chunk));
    setBaseDraft(chunkDraft(chunk));
    setBaseMetadata(chunk.chunk_metadata);
    setEditReviewSha256(reviewSha256);
    setHeadingPath(chunk.heading_path.join("\n"));
    setMetadataEdits({ patch: {}, dirty: false, error: null });
    setMetadataSession((current) => current + 1);
  }, [chunk, reviewSha256, dirty]);

  const validation = !draft.text.trim()
    ? "Chunk text cannot be empty."
    : !validDate(draft.validity_start_date) ||
        !validDate(draft.validity_end_date)
      ? "Use a valid YYYY-MM-DD date."
      : draft.validity_start_date &&
          draft.validity_end_date &&
          draft.validity_end_date <= draft.validity_start_date
        ? "The end date must be after the start date."
        : metadataEdits.error;
  const variant = !editable
    ? "readOnly"
    : saving || disabled
      ? "disabled"
      : "primary";
  const labelingProblem =
    Boolean(chunk.labeling_error) ||
    ["failed", "stale", "cancelled"].includes(chunk.labeling_status ?? "");
  const afterChunk: Record<string, unknown> = {
    ...chunk,
    text: draft.text,
    chunk_type: draft.chunk_type,
    heading_path: headingPath
      .split("\n")
      .map((heading) => heading.trim())
      .filter(Boolean),
    metadata: { ...baseMetadata, ...metadataEdits.patch },
    validity_start_date: draft.validity_start_date,
    validity_end_date: draft.validity_end_date,
  };

  function reset() {
    setDraft(chunkDraft(chunk));
    setBaseDraft(chunkDraft(chunk));
    setBaseMetadata(chunk.chunk_metadata);
    setMetadataEdits({ patch: {}, dirty: false, error: null });
    setMetadataSession((current) => current + 1);
    setEditReviewSha256(reviewSha256);
    setHeadingPath(chunk.heading_path.join("\n"));
    setError(null);
    onDirtyChange(chunk.id, false);
  }
  async function save() {
    if (validation || saving || disabled || !editable || !editReviewSha256)
      return;
    setSaving(true);
    setError(null);
    try {
      await onSave(
        {
          ...draft,
          heading_path: headingPath
            .split("\n")
            .map((heading) => heading.trim())
            .filter(Boolean),
          ...(metadataEdits.dirty
            ? { chunk_metadata: metadataEdits.patch }
            : {}),
        },
        editReviewSha256
      );
      reset();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not save the chunk."
      );
    } finally {
      setSaving(false);
    }
  }

  function renderAfterValue(key: string, value: unknown): ReactNode {
    if (key === "user_file_id" || key === "position")
      return (
        <InputTypeIn
          aria-label={`After ${key}`}
          value={value === null || value === undefined ? "" : String(value)}
          variant="readOnly"
        />
      );
    if (key === "text")
      return (
        <InputTextArea
          aria-label="After text"
          value={draft.text}
          onChange={(event) => setDraft({ ...draft, text: event.target.value })}
          variant={variant}
          rows={5}
          maxRows={14}
          autoResize
        />
      );
    if (key === "chunk_type")
      return (
        <InputTypeIn
          aria-label="After chunk_type"
          value={draft.chunk_type ?? ""}
          onChange={(event) =>
            setDraft({ ...draft, chunk_type: event.target.value || null })
          }
          variant={variant}
        />
      );
    if (key === "heading_path")
      return (
        <div className="flex flex-col gap-1.5">
          <InputTextArea
            aria-label="After heading_path"
            value={headingPath}
            onChange={(event) => setHeadingPath(event.target.value)}
            variant={variant}
            rows={Math.max(2, Math.min(draft.heading_path.length, 6))}
            maxRows={8}
            autoResize
          />
          <Text font="secondary-body" color="text-03">
            One heading per line.
          </Text>
        </div>
      );
    if (key === "metadata")
      return (
        <IntakeLegalMetadataFields
          key={`metadata:${metadataSession}`}
          metadata={baseMetadata}
          editing
          readOnly={!editable}
          disabled={saving || disabled}
          validityStart={draft.validity_start_date}
          validityEnd={draft.validity_end_date}
          onEdits={onMetadataEdits}
        />
      );
    if (key === "validity_start_date" || key === "validity_end_date")
      return (
        <div className="flex flex-col gap-1">
          <InputTypeIn
            aria-label={`After ${key}`}
            value={draft[key] ?? ""}
            placeholder="YYYY-MM-DD"
            onChange={(event) =>
              setDraft({ ...draft, [key]: event.target.value || null })
            }
            variant={
              !editable
                ? "readOnly"
                : saving || disabled
                  ? "disabled"
                  : validDate(draft[key])
                    ? "primary"
                    : "error"
            }
          />
          <IntakeMetadataOrigin
            metadata={baseMetadata}
            field={key}
            pendingManual={draft[key] !== baseDraft[key]}
          />
        </div>
      );
    return <ReadOnlyFieldValue value={value} />;
  }

  return (
    <section
      aria-label={`Added chunk ${chunk.position + 1}`}
      className="rounded-lg border border-border-02 p-4 flex flex-col gap-3"
    >
      <div className="flex items-start justify-between gap-2">
        <Text
          font="main-ui-body"
          color="text-05"
          as="p"
        >{`New chunk ${chunk.position + 1}${documentName ? ` · ${documentName}` : ""}`}</Text>
        <div className="flex items-center gap-2 shrink-0">
          <Tag title="Added" />
          <Tag title={editable ? "pending" : (chunk.status ?? "read only")} />
        </div>
      </div>
      {documentName && (
        <Text
          font="secondary-body"
          color="text-03"
        >{`File: ${documentName}`}</Text>
      )}
      {sourceUrl && (
        <Text
          font="secondary-body"
          color="text-03"
        >{`Source: ${sourceUrl}`}</Text>
      )}
      {labelingProblem && (
        <div
          role="alert"
          className="flex flex-col gap-1 rounded-08 border border-status-error-02 bg-status-error-01 p-3"
        >
          <Text
            font="main-ui-action"
            color="status-error-05"
          >{`Labeling: ${chunk.labeling_status ?? "unresolved"}`}</Text>
          <Text as="p" font="secondary-body">
            {chunk.labeling_error ||
              "This chunk has no valid labeling result. Retry labeling before approval."}
          </Text>
        </div>
      )}
      <ChunkFieldsReview
        before={null}
        after={afterChunk}
        renderAfterValue={renderAfterValue}
      />
      <Text font="secondary-body" color="text-03">
        user_file_id and position identify the target and cannot be changed.
      </Text>
      {dirty && (
        <Text as="p" font="secondary-body" color="text-03">
          Saving changes regenerates labels for this document. Other documents
          keep their current labels.
        </Text>
      )}
      {(validation || error) && (
        <Text color="status-error-05">{String(validation ?? error)}</Text>
      )}
      {(editable || dirty) && (
        <div className="flex gap-2 justify-end">
          <Button
            prominence="secondary"
            disabled={!dirty || saving || disabled}
            onClick={reset}
          >
            Discard chunk changes
          </Button>
          <Button
            disabled={
              !dirty || Boolean(validation) || saving || disabled || !editable
            }
            onClick={() => void save()}
          >
            {saving ? "Saving chunk…" : "Save chunk changes"}
          </Button>
        </div>
      )}
      <div className="flex flex-col gap-3">
        <Text font="secondary-action">Rendered preview and labels</Text>
        {children}
      </div>
      <IntakeChunkSourceDetails
        chunkId={chunk.id}
        fileId={chunk.user_file_id}
        metadata={dirty ? baseMetadata : chunk.chunk_metadata}
      />
    </section>
  );
}
