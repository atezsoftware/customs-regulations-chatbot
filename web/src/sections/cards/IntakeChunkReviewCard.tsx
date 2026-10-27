"use client";

import { useEffect, useState, type ReactNode } from "react";
import {
  Button,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import type { IntakeChunkUpdate } from "@/lib/regulatory/documentIntake";
import { intakeMetadataEdits } from "@/lib/regulatory/intakeMetadata";
import {
  type IntakeChunkReviewDraft,
  createIntakeChunkDraft,
  hasIntakeLabelingProblem,
  intakeChunkDraftDirty,
  intakeChunkDraftUpdate,
  intakeChunkDraftValidation,
  validIntakeDate,
} from "@/lib/regulatory/intakeChunkDraft";
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
  onDirtyChange?: (chunkId: string, dirty: boolean) => void;
  persistedDraft?: IntakeChunkReviewDraft;
  onDraftChange?: (
    chunkId: string,
    draft: IntakeChunkReviewDraft | null
  ) => void;
  saveDisabled?: boolean;
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
  persistedDraft,
  onDraftChange,
  saveDisabled = false,
}: IntakeChunkReviewCardProps) {
  const [reviewDraft, setReviewDraft] = useState(
    () => persistedDraft ?? createIntakeChunkDraft(chunk, reviewSha256)
  );
  const {
    values: draft,
    baseValues: baseDraft,
    baseMetadata,
    headingPath,
  } = reviewDraft;
  const metadataEdits = intakeMetadataEdits(
    baseMetadata,
    reviewDraft.metadataInputs
  );
  function setDraft(values: IntakeChunkUpdate) {
    setReviewDraft((current) => ({ ...current, values }));
  }
  function setHeadingPath(value: string) {
    setReviewDraft((current) => ({ ...current, headingPath: value }));
  }
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dirty = intakeChunkDraftDirty(reviewDraft);
  useEffect(() => {
    onDirtyChange?.(chunk.id, dirty);
    onDraftChange?.(chunk.id, dirty ? reviewDraft : null);
  }, [chunk.id, dirty, reviewDraft, onDirtyChange, onDraftChange]);
  useEffect(() => {
    if (dirty) return;
    setReviewDraft(createIntakeChunkDraft(chunk, reviewSha256));
  }, [chunk, reviewSha256, dirty]);

  const validation = intakeChunkDraftValidation(reviewDraft);
  const variant = !editable
    ? "readOnly"
    : saving || disabled
      ? "disabled"
      : "primary";
  const labelingProblem = hasIntakeLabelingProblem(chunk);
  const afterChunk: Record<string, unknown> = {
    ...chunk,
    text: draft.text,
    chunk_type: draft.chunk_type ?? "",
    heading_path: headingPath
      .split("\n")
      .map((heading) => heading.trim())
      .filter(Boolean),
    metadata: { ...baseMetadata, ...metadataEdits.patch },
    validity_start_date: draft.validity_start_date,
    validity_end_date: draft.validity_end_date,
  };

  function reset() {
    setReviewDraft(createIntakeChunkDraft(chunk, reviewSha256));
    setError(null);
    onDirtyChange?.(chunk.id, false);
    onDraftChange?.(chunk.id, null);
  }
  async function save() {
    if (
      validation ||
      saving ||
      disabled ||
      saveDisabled ||
      !editable ||
      !reviewDraft.reviewSha256
    )
      return;
    setSaving(true);
    setError(null);
    try {
      await onSave(
        intakeChunkDraftUpdate(reviewDraft),
        reviewDraft.reviewSha256
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
          metadata={baseMetadata}
          editing
          readOnly={!editable}
          disabled={saving || disabled}
          validityStart={draft.validity_start_date}
          validityEnd={draft.validity_end_date}
          inputs={reviewDraft.metadataInputs}
          onInputsChange={(metadataInputs) =>
            setReviewDraft((current) => ({ ...current, metadataInputs }))
          }
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
                  : validIntakeDate(draft[key])
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
          Save all pending chunk changes together to regenerate labels once per
          changed document. Other documents keep their current labels.
        </Text>
      )}
      {dirty && saveDisabled && (
        <Text as="p" font="secondary-body" color="text-03">
          Multiple chunks have unsaved changes. Use Save all chunk changes
          above.
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
              !dirty ||
              Boolean(validation) ||
              saving ||
              disabled ||
              saveDisabled ||
              !editable
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
