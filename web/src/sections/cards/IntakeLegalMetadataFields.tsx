"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Button,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import {
  type IntakeMetadataEdits,
  intakeLegalMetadataFields,
  intakeMetadataEdits,
  intakeMetadataInput,
  intakeMetadataInputs,
  metadataRecord,
} from "@/lib/regulatory/intakeMetadata";

interface IntakeMetadataOriginProps {
  metadata: Record<string, unknown>;
  field: string;
  pendingManual?: boolean;
}

export function IntakeMetadataOrigin({
  metadata,
  field,
  pendingManual = false,
}: IntakeMetadataOriginProps) {
  const audit = metadataRecord(metadata.intake_legal_metadata);
  const review = metadataRecord(metadataRecord(audit.fields)[field]);
  const source = pendingManual ? "manual" : review.source;
  const basis = pendingManual ? "user_review" : review.basis;
  const label =
    source === "manual"
      ? pendingManual
        ? "Manual · unsaved"
        : "Manual review"
      : source === "llm"
        ? `Gemini · ${typeof audit.model === "string" ? audit.model : "model extraction"}`
        : source === "structural"
          ? "Chunker"
          : "Source not recorded";
  const rationale =
    basis === "relative_publication"
      ? "Calculated relative to publication date."
      : basis === "explicit"
        ? "Explicit wording in the source."
        : basis === "chunker"
          ? "Extracted from document structure."
          : basis === "user_review"
            ? "Entered during user review."
            : null;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap gap-1">
        <Tag title={label} />
        {!pendingManual && source === "llm" && review.stale === true && (
          <Tag title="Source changed · review needed" />
        )}
      </div>
      {rationale && (
        <Text font="secondary-body" color="text-03">
          {rationale}
        </Text>
      )}
      {!pendingManual &&
        typeof review.evidence_quote === "string" &&
        review.evidence_quote && (
          <Text
            as="p"
            font="secondary-body"
            color="text-03"
          >{`Evidence: ${review.evidence_quote}`}</Text>
        )}
    </div>
  );
}

interface IntakeLegalMetadataFieldsProps {
  metadata: Record<string, unknown>;
  editing: boolean;
  readOnly?: boolean;
  disabled: boolean;
  validityStart: string | null;
  validityEnd: string | null;
  onEdits?: (edits: IntakeMetadataEdits) => void;
  inputs?: Record<string, string>;
  onInputsChange?: (inputs: Record<string, string>) => void;
}

export default function IntakeLegalMetadataFields({
  metadata,
  editing,
  readOnly = false,
  disabled,
  validityStart,
  validityEnd,
  onEdits,
  inputs: controlledInputs,
  onInputsChange,
}: IntakeLegalMetadataFieldsProps) {
  const [localInputs, setLocalInputs] = useState(() =>
    intakeMetadataInputs(metadata)
  );
  const inputs = controlledInputs ?? localInputs;
  function setInputs(next: Record<string, string>) {
    setLocalInputs(next);
    onInputsChange?.(next);
  }
  const edits = useMemo(
    () => intakeMetadataEdits(metadata, inputs),
    [metadata, inputs]
  );
  useEffect(() => {
    if (editing) onEdits?.(edits);
  }, [editing, edits, onEdits]);
  const audit = metadataRecord(metadata.intake_legal_metadata);
  const fields = metadataRecord(audit.fields);
  const warnings = Array.from(
    new Set(
      Array.isArray(audit.warnings)
        ? audit.warnings.filter(
            (warning): warning is string => typeof warning === "string"
          )
        : []
    )
  );
  if (
    audit.input_stale === true &&
    !warnings.some((warning) =>
      warning.startsWith("Document context changed after metadata extraction")
    )
  )
    warnings.push(
      "Document context changed after metadata extraction. Review Gemini-derived metadata against the current source; saved manual values are retained."
    );
  const visible = editing
    ? intakeLegalMetadataFields
    : intakeLegalMetadataFields.filter(
        ([key]) => key in metadata || key in fields
      );
  return (
    <div
      role="group"
      aria-label="Legal metadata"
      className="flex flex-col gap-2"
    >
      {visible.length === 0 && (
        <Text font="secondary-body" color="text-03">
          No legal metadata has been extracted for this chunk.
        </Text>
      )}
      <div className="flex flex-col gap-2">
        {visible.map(([key, label, type]) => (
          <div
            key={key}
            role="group"
            aria-label={`Metadata ${key}`}
            className="flex flex-col gap-1"
          >
            <span className="break-words font-mono text-xs text-text-03">
              {key}
            </span>
            {editing ? (
              (typeof metadata[key] === "object" && metadata[key] !== null) ||
              key.endsWith("_title") ? (
                <InputTextArea
                  aria-label={`After metadata ${key}`}
                  value={inputs[key] ?? ""}
                  variant={
                    readOnly ? "readOnly" : disabled ? "disabled" : "primary"
                  }
                  rows={2}
                  maxRows={8}
                  autoResize
                  onChange={(event) =>
                    setInputs({ ...inputs, [key]: event.target.value })
                  }
                />
              ) : (
                <InputTypeIn
                  aria-label={`After metadata ${key}`}
                  value={inputs[key] ?? ""}
                  placeholder={
                    type === "integer"
                      ? "Whole number"
                      : "Unknown / not provided"
                  }
                  variant={
                    readOnly ? "readOnly" : disabled ? "disabled" : "primary"
                  }
                  onChange={(event) =>
                    setInputs({ ...inputs, [key]: event.target.value })
                  }
                />
              )
            ) : (
              <Text as="p" font="secondary-body">
                {intakeMetadataInput(metadata[key]) || "—"}
              </Text>
            )}
            <IntakeMetadataOrigin
              metadata={metadata}
              field={key}
              pendingManual={editing && key in edits.patch}
            />
          </div>
        ))}
        {!editing &&
          (
            [
              ["validity_start_date", "Validity start date", validityStart],
              ["validity_end_date", "Validity end date", validityEnd],
            ] as const
          ).map(([field, label, value]) => (
            <div
              key={field}
              role="group"
              aria-label={`Metadata ${field}`}
              className="flex flex-col gap-1 rounded-08 border border-border-01 p-3"
            >
              <Text font="secondary-action">{label}</Text>
              <Text font="secondary-body">{value || "—"}</Text>
              <IntakeMetadataOrigin metadata={metadata} field={field} />
            </div>
          ))}
      </div>
      {warnings.map((warning, index) => (
        <Text key={index} as="p" font="secondary-body" color="text-04">
          {warning}
        </Text>
      ))}
      {editing && edits.error && (
        <Text color="status-error-05">{edits.error}</Text>
      )}
    </div>
  );
}

interface IntakeChunkSourceDetailsProps {
  chunkId: string;
  fileId: string;
  metadata: Record<string, unknown>;
}

export function IntakeChunkSourceDetails({
  chunkId,
  fileId,
  metadata,
}: IntakeChunkSourceDetailsProps) {
  const [expanded, setExpanded] = useState(false);
  const legalKeys = new Set<string>(
    intakeLegalMetadataFields.map(([key]) => key)
  );
  const technical = Object.fromEntries(
    Object.entries(metadata).filter(([key]) => !legalKeys.has(key))
  );
  return (
    <div className="flex flex-col gap-2 border-t border-border-02 pt-3">
      <div>
        <Button prominence="tertiary" onClick={() => setExpanded(!expanded)}>
          {expanded ? "Hide source details" : "Source details"}
        </Button>
      </div>
      {expanded && (
        <>
          <Text font="secondary-body" color="text-03">
            Technical identifiers and extraction evidence are read-only.
          </Text>
          <InputTypeIn
            aria-label="Read-only chunk ID"
            variant="readOnly"
            value={chunkId}
          />
          <InputTypeIn
            aria-label="Read-only file ID"
            variant="readOnly"
            value={fileId}
          />
          <InputTextArea
            aria-label="Read-only technical metadata"
            variant="readOnly"
            value={JSON.stringify(technical, null, 2)}
            rows={4}
            maxRows={14}
            autoResize
          />
        </>
      )}
    </div>
  );
}
