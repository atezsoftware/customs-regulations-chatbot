"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import {
  Button,
  Checkbox,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import {
  type DocumentIntakeBatch,
  type IntakeDocument,
  type IntakeLineageSelection,
  getDocumentIntakePredecessorCandidates,
} from "@/lib/regulatory/documentIntake";

interface IntakePredecessorReviewProps {
  batch: DocumentIntakeBatch;
  document: IntakeDocument;
  disabled: boolean;
  onSave: (
    selection: IntakeLineageSelection,
    expectedReviewSha256: string
  ) => Promise<void>;
  onDirtyChange: (key: string, dirty: boolean) => void;
}

function selectionFor(document: IntakeDocument): IntakeLineageSelection {
  return {
    predecessor_user_file_id:
      document.lineage_review?.predecessor_user_file_id ?? null,
    relation: document.lineage_review?.relation ?? "successor",
    effective_date: document.lineage_review?.effective_date ?? null,
    evidence: document.lineage_review?.evidence ?? null,
  };
}

export default function IntakePredecessorReview({
  batch,
  document,
  disabled,
  onSave,
  onDirtyChange,
}: IntakePredecessorReviewProps) {
  const [selection, setSelection] = useState(() => selectionFor(document));
  const [expanded, setExpanded] = useState(false);
  const [baseSelection, setBaseSelection] = useState(() =>
    selectionFor(document)
  );
  const [editReviewSha256, setEditReviewSha256] = useState(batch.review_sha256);
  const [selectedTitle, setSelectedTitle] = useState(
    document.lineage_review?.title ?? ""
  );
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const editable = batch.status === "ready" && Boolean(batch.review_sha256);
  const { data: candidates, error: candidateError } = useSWR(
    editable ? ["intake-predecessors", batch.id, document.id, search] : null,
    () => getDocumentIntakePredecessorCandidates(batch.id, document.id, search)
  );
  const dirtyKey = `lineage:${document.id}`;
  const dirty = JSON.stringify(selection) !== JSON.stringify(baseSelection);
  useEffect(() => {
    if (dirty) return;
    const current = selectionFor(document);
    setSelection(current);
    setBaseSelection(current);
    setSelectedTitle(document.lineage_review?.title ?? "");
    setEditReviewSha256(batch.review_sha256);
  }, [document.lineage_review, batch.review_sha256, dirty]);
  useEffect(() => {
    onDirtyChange(dirtyKey, dirty);
  }, [dirtyKey, dirty, onDirtyChange]);
  const validation =
    selection.relation === "repeals" && selection.predecessor_user_file_id
      ? !selection.effective_date ||
        !/^\d{4}-\d{2}-\d{2}$/.test(selection.effective_date)
        ? "A repeal effective date is required (YYYY-MM-DD)."
        : !selection.evidence?.trim()
          ? "Quote the provision that repeals the previous legislation."
          : null
      : null;

  function reset() {
    setSelection(selectionFor(document));
    setBaseSelection(selectionFor(document));
    setEditReviewSha256(batch.review_sha256);
    setSelectedTitle(document.lineage_review?.title ?? "");
    setError(null);
    onDirtyChange(dirtyKey, false);
  }

  async function save() {
    if (disabled || saving || validation || !editReviewSha256) return;
    setSaving(true);
    setError(null);
    try {
      await onSave(selection, editReviewSha256);
      setBaseSelection(selection);
      onDirtyChange(dirtyKey, false);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not save the previous legislation relationship."
      );
    } finally {
      setSaving(false);
    }
  }

  if (!editable && !document.lineage_review) return null;

  return (
    <section
      aria-label={`Previous legislation for ${document.display_name}`}
      className="flex flex-col gap-2 rounded-12 border border-border-02 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Text as="h3" font="secondary-action">
          Previous legislation
        </Text>
        <Button
          prominence="tertiary"
          disabled={dirty || saving || disabled}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded
            ? "Hide previous legislation"
            : "Review previous legislation"}
        </Button>
      </div>
      {!expanded && !document.lineage_review && (
        <Text font="secondary-body" color="text-03">
          {candidateError
            ? "Previous legislation check needs attention."
            : candidates
              ? `${candidates.length} predecessor suggestions · optional document relationship`
              : "Checking previous legislation…"}
        </Text>
      )}
      {document.lineage_review && (
        <div className="flex flex-col gap-1">
          <Text font="secondary-body">{`${document.lineage_review.title} · ${document.lineage_review.relation === "repeals" ? "repealed by this document" : "predecessor of this document"}`}</Text>
          <Tag
            title={
              document.lineage_review.status === "active"
                ? "Relationship active"
                : "Pending publication"
            }
          />
          <Text font="secondary-body" color="text-03">
            {document.lineage_review.status === "active"
              ? "The reciprocal relationship is recorded on both documents."
              : "The previous document changes only after approval and successful publication."}
          </Text>
        </div>
      )}
      {expanded && (
        <Text as="p" font="secondary-body" color="text-03">
          Suggestions use document titles and metadata. A similar title or a
          different law number does not establish repeal. No chunk comparison is
          performed.
        </Text>
      )}
      {expanded && editable && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <InputTypeIn
              aria-label={`Search previous legislation for ${document.display_name}`}
              placeholder="Search by legislation title or number"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              variant={disabled || saving ? "disabled" : "primary"}
            />
            <Button
              prominence="secondary"
              disabled={disabled || saving}
              onClick={() => setSearch(query.trim())}
            >
              Search previous legislation
            </Button>
          </div>
          {!candidates && !candidateError && (
            <Text font="secondary-body">Checking existing legislation…</Text>
          )}
          {candidateError && (
            <Text color="status-error-05">
              {candidateError instanceof Error
                ? candidateError.message
                : "Previous legislation search failed."}
            </Text>
          )}
          {candidates?.length === 0 && (
            <Text font="secondary-body" color="text-03">
              No previous legislation candidate found.
            </Text>
          )}
          {candidates?.map((candidate) => (
            <div
              key={candidate.user_file_id}
              className="flex flex-wrap items-center justify-between gap-2 border-t border-border-02 pt-2"
            >
              <div className="flex flex-col gap-1">
                <Text font="secondary-action">{candidate.title}</Text>
                <Text
                  font="secondary-body"
                  color="text-03"
                >{`${candidate.reason}${candidate.status ? ` · ${candidate.status}` : ""}`}</Text>
              </div>
              <Button
                prominence={
                  selection.predecessor_user_file_id === candidate.user_file_id
                    ? "primary"
                    : "secondary"
                }
                disabled={disabled || saving}
                onClick={() => {
                  setSelection({
                    predecessor_user_file_id: candidate.user_file_id,
                    relation: "successor",
                    effective_date: null,
                    evidence: null,
                  });
                  setSelectedTitle(candidate.title);
                }}
              >{`Select ${candidate.title}`}</Button>
            </div>
          ))}
          {selection.predecessor_user_file_id && (
            <>
              <Text font="secondary-action">{`Selected predecessor: ${selectedTitle || selection.predecessor_user_file_id}`}</Text>
              <div className="flex items-center gap-2">
                <Checkbox
                  aria-label="This document explicitly repeals the previous legislation"
                  checked={selection.relation === "repeals"}
                  disabled={disabled || saving}
                  onCheckedChange={(checked) =>
                    setSelection({
                      ...selection,
                      relation: checked ? "repeals" : "successor",
                    })
                  }
                />
                <Text font="secondary-body">
                  This document explicitly repeals the previous legislation.
                </Text>
              </div>
              {selection.relation === "repeals" && (
                <>
                  <Text font="secondary-action">Repeal effective date</Text>
                  <InputTypeIn
                    aria-label="Repeal effective date"
                    placeholder="YYYY-MM-DD"
                    value={selection.effective_date ?? ""}
                    variant={disabled || saving ? "disabled" : "primary"}
                    onChange={(event) =>
                      setSelection({
                        ...selection,
                        effective_date: event.target.value || null,
                      })
                    }
                  />
                  <Text font="secondary-action">Repeal provision</Text>
                  <InputTextArea
                    aria-label="Repeal provision"
                    value={selection.evidence ?? ""}
                    variant={disabled || saving ? "disabled" : "primary"}
                    rows={3}
                    onChange={(event) =>
                      setSelection({
                        ...selection,
                        evidence: event.target.value || null,
                      })
                    }
                  />
                </>
              )}
              <Button
                prominence="tertiary"
                disabled={disabled || saving}
                onClick={() => {
                  setSelection({
                    predecessor_user_file_id: null,
                    relation: "successor",
                    effective_date: null,
                    evidence: null,
                  });
                  setSelectedTitle("");
                }}
              >
                No previous legislation
              </Button>
            </>
          )}
          {validation && <Text color="status-error-05">{validation}</Text>}
          {error && <Text color="status-error-05">{error}</Text>}
          {dirty && (
            <div className="flex flex-wrap gap-2">
              <Button
                disabled={disabled || saving || Boolean(validation)}
                onClick={() => void save()}
              >
                {saving
                  ? "Saving relationship…"
                  : "Save previous legislation relationship"}
              </Button>
              <Button
                prominence="secondary"
                disabled={disabled || saving}
                onClick={reset}
              >
                Discard relationship changes
              </Button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
