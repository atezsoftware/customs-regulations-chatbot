"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import {
  Button,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import {
  type DocumentIntakeBatch,
  type IntakeDocument,
  type IntakeExistingParent,
  type IntakePublicationSelection,
  getDocumentIntakeExistingParentCandidates,
  getLegislationLineage,
} from "@/lib/regulatory/documentIntake";
import { documentPdfUrl, fetchChunkPageForFile } from "@/lib/regulatory/svc";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import { ChunkContent } from "@/sections/cards/ChunkChangeCard";
import { ReadOnlyFieldValue } from "@/sections/cards/ChunkFieldsReview";

interface RetainedSourceChunkProps {
  chunk: RegulatoryChunk;
}

export function RetainedSourceChunk({ chunk }: RetainedSourceChunkProps) {
  return (
    <section
      aria-label={`Retained source chunk ${chunk.position + 1}`}
      className="flex flex-col gap-2 rounded-12 border border-border-02 p-3"
    >
      <Text font="secondary-action">{`Retained source chunk ${chunk.position + 1}`}</Text>
      <Tag title="Source evidence · not published" />
      <ChunkContent text={chunk.text} />
      <details>
        <summary>
          <Text font="secondary-action">Retained fields and provenance</Text>
        </summary>
        <ReadOnlyFieldValue value={{ ...chunk, text: undefined }} />
      </details>
    </section>
  );
}

interface ExistingParentPreviewProps {
  parent: IntakeExistingParent;
}

function ExistingParentPreview({ parent }: ExistingParentPreviewProps) {
  const [offset, setOffset] = useState(0);
  const { data, error } = useSWR(
    ["intake-existing-parent-chunks", parent.user_file_id, offset],
    () => fetchChunkPageForFile(parent.user_file_id, offset, 10)
  );
  const { data: relations, error: relationsError } = useSWR(
    ["intake-existing-parent-attachments", parent.user_file_id],
    () => getLegislationLineage(parent.user_file_id)
  );
  return (
    <section
      aria-label={`Existing parent: ${parent.display_name}`}
      className="flex flex-col gap-2 rounded-08 border border-border-02 p-3"
    >
      <Text as="h3" font="secondary-action">
        {parent.display_name}
      </Text>
      <Tag title="Existing legislation · read only" />
      <div>
        <Button
          prominence="tertiary"
          href={documentPdfUrl(parent.user_file_id)}
          target="_blank"
        >
          Open existing parent document PDF
        </Button>
      </div>
      {parent.source_url && (
        <Text font="secondary-body" color="text-03">
          {parent.source_url}
        </Text>
      )}
      {error && (
        <Text color="status-error-05">
          {error instanceof Error
            ? error.message
            : "Could not open the existing parent."}
        </Text>
      )}
      {!data && !error && <Text>Loading existing parent chunks…</Text>}
      {data?.items.map((chunk) => (
        <div key={chunk.id} className="border-t border-border-02 pt-2">
          <ChunkContent text={chunk.text} />
        </div>
      ))}
      {data?.total === 0 && (
        <Text>No chunks are available for this existing parent.</Text>
      )}
      {data && data.total > data.limit && (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            prominence="secondary"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - data.limit))}
          >
            Previous parent chunks
          </Button>
          <Text font="secondary-body">{`${offset + 1}–${Math.min(offset + data.items.length, data.total)} of ${data.total}`}</Text>
          <Button
            prominence="secondary"
            disabled={offset + data.limit >= data.total}
            onClick={() => setOffset(offset + data.limit)}
          >
            Next parent chunks
          </Button>
        </div>
      )}
      {relationsError && (
        <Text color="status-error-05">
          Could not load this document's published attachment links.
        </Text>
      )}
      {relations?.attachments?.map((attachment) => (
        <div
          key={attachment.user_file_id}
          className="flex flex-col gap-1 border-t border-border-02 pt-2"
        >
          <Text font="secondary-action">{`Published attachment: ${attachment.display_name}`}</Text>
          {attachment.source_url && (
            <Text font="secondary-body" color="text-03">
              {attachment.source_url}
            </Text>
          )}
          <div>
            <Button
              prominence="tertiary"
              href={documentPdfUrl(attachment.user_file_id)}
              target="_blank"
            >
              {`Open published attachment ${attachment.display_name}`}
            </Button>
          </div>
        </div>
      ))}
    </section>
  );
}

interface IntakePublicationReviewProps {
  batch: DocumentIntakeBatch;
  document: IntakeDocument;
  disabled: boolean;
  onDirtyChange: (key: string, dirty: boolean) => void;
  onSave: (
    selection: IntakePublicationSelection,
    expectedReviewSha256: string
  ) => Promise<void>;
}

function selectionFor(document: IntakeDocument): IntakePublicationSelection {
  return {
    disposition: document.publication_disposition ?? "legislation",
    duplicate_of_document_id: document.duplicate_of_document_id ?? null,
    existing_parent_file_id: document.existing_parent?.user_file_id ?? null,
    reason: document.publication_review?.reason ?? "",
  };
}

export default function IntakePublicationReview({
  batch,
  document,
  disabled,
  onDirtyChange,
  onSave,
}: IntakePublicationReviewProps) {
  const [expanded, setExpanded] = useState(false);
  const [selection, setSelection] = useState(() => selectionFor(document));
  const [baseSelection, setBaseSelection] = useState(() =>
    selectionFor(document)
  );
  const [reviewReceipt, setReviewReceipt] = useState(
    batch.publication_review_sha256
  );
  const [selectedParent, setSelectedParent] = useState(
    document.existing_parent ?? null
  );
  const [openParent, setOpenParent] = useState<IntakeExistingParent | null>(
    null
  );
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dirtyKey = `publication:${document.id}`;
  const dirty = JSON.stringify(selection) !== JSON.stringify(baseSelection);
  const editable =
    Boolean(batch.publication_review_sha256) &&
    !["preparing", "approving", "completed"].includes(batch.status);
  const canSelectExistingParent = document.parent_document_id === null;
  const { data: candidates, error: candidateError } = useSWR(
    expanded &&
      editable &&
      canSelectExistingParent &&
      selection.disposition === "legislation"
      ? [
          "intake-existing-parents",
          batch.id,
          document.id,
          search,
          batch.publication_review_sha256,
        ]
      : null,
    () =>
      getDocumentIntakeExistingParentCandidates(batch.id, document.id, search)
  );
  const possibleRenditions = batch.documents.filter(
    (item) =>
      item.id !== document.id &&
      item.publication_disposition !== "source_evidence"
  );
  useEffect(() => {
    if (dirty) return;
    const current = selectionFor(document);
    setSelection(current);
    setBaseSelection(current);
    setSelectedParent(document.existing_parent ?? null);
    setReviewReceipt(batch.publication_review_sha256);
  }, [document, batch.publication_review_sha256, dirty]);
  useEffect(() => {
    onDirtyChange(dirtyKey, dirty);
  }, [dirtyKey, dirty, onDirtyChange]);

  const validation = !selection.reason.trim()
    ? "Explain why this is the same document or why this existing legislation is its parent."
    : selection.reason.length > 2000
      ? "Keep the review reason within 2000 characters."
      : selection.disposition === "source_evidence" &&
          !selection.duplicate_of_document_id
        ? "Choose the document whose other rendition will be retained as evidence."
        : null;

  function reset() {
    const current = selectionFor(document);
    setSelection(current);
    setBaseSelection(current);
    setSelectedParent(document.existing_parent ?? null);
    setReviewReceipt(batch.publication_review_sha256);
    setError(null);
    onDirtyChange(dirtyKey, false);
  }

  async function save() {
    if (
      !editable ||
      disabled ||
      saving ||
      !dirty ||
      validation ||
      !reviewReceipt
    )
      return;
    setSaving(true);
    setError(null);
    try {
      await onSave(selection, reviewReceipt);
      setBaseSelection(selection);
      onDirtyChange(dirtyKey, false);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not save the document publication review."
      );
    } finally {
      setSaving(false);
    }
  }

  if (!editable && !document.publication_review && !document.existing_parent)
    return null;

  return (
    <section
      aria-label={`Document identity for ${document.display_name}`}
      className="flex flex-col gap-2 rounded-12 border border-border-02 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Text as="h3" font="secondary-action">
          Document identity and attachment parent
        </Text>
        {editable && (
          <Button
            prominence="tertiary"
            disabled={disabled || dirty || saving}
            onClick={() => setExpanded(!expanded)}
          >
            {expanded
              ? "Hide document identity review"
              : "Review document identity"}
          </Button>
        )}
      </div>
      {document.publication_review && (
        <Text font="secondary-body">{`Review reason: ${document.publication_review.reason}`}</Text>
      )}
      {document.existing_parent && (
        <div className="flex flex-col gap-1">
          <Text font="secondary-action">{`Existing parent: ${document.existing_parent.display_name}`}</Text>
          <Text font="secondary-body" color="text-03">
            {batch.status === "completed"
              ? "This new attachment is linked to the existing legislation."
              : "The attachment relationship is published with this new document."}
          </Text>
          <div>
            <Button
              prominence="tertiary"
              disabled={disabled || saving}
              onClick={() => setOpenParent(document.existing_parent!)}
            >
              Open existing parent
            </Button>
          </div>
        </div>
      )}
      {expanded && editable && (
        <>
          <Text as="p" font="secondary-body" color="text-03">
            Keep another rendition as source evidence when it contains the same
            legislation. Its original, chunks and review remain available
            without publishing a second legislation. An existing parent
            identifies the legislation this new attachment belongs to. This
            relationship does not replace or repeal the parent.
          </Text>
          <div>
            <Button
              prominence={
                selection.disposition === "legislation"
                  ? "primary"
                  : "secondary"
              }
              disabled={disabled || saving}
              onClick={() =>
                setSelection({
                  disposition: "legislation",
                  duplicate_of_document_id: null,
                  existing_parent_file_id: null,
                  reason: "",
                })
              }
            >
              Keep as new legislation
            </Button>
          </div>
          <Text font="secondary-action">
            Another rendition of a document in this import
          </Text>
          {possibleRenditions.map((target) => (
            <div
              key={target.id}
              className="flex flex-col gap-1 border-t border-border-02 pt-2"
            >
              <Text font="secondary-action">{target.display_name}</Text>
              {(target.source_display_name || target.mime_type) && (
                <Text font="secondary-body" color="text-03">
                  {[target.mime_type, target.source_display_name]
                    .filter(Boolean)
                    .join(" · ")}
                </Text>
              )}
              {target.source_url && (
                <Text font="secondary-body" color="text-03">
                  {target.source_url}
                </Text>
              )}
              <div>
                <Button
                  aria-label={`Keep as source evidence of ${target.display_name}${target.mime_type ? ` (${target.mime_type})` : ""}`}
                  prominence={
                    selection.duplicate_of_document_id === target.id
                      ? "primary"
                      : "secondary"
                  }
                  disabled={disabled || saving}
                  onClick={() => {
                    setSelection({
                      disposition: "source_evidence",
                      duplicate_of_document_id: target.id,
                      existing_parent_file_id: null,
                      reason: "",
                    });
                    setSelectedParent(null);
                  }}
                >
                  Keep as source evidence
                </Button>
              </div>
            </div>
          ))}
          {selection.disposition === "legislation" &&
            !canSelectExistingParent && (
              <Text font="secondary-body" color="text-03">
                This attachment already has a parent in this import. Its
                existing source relationship is kept.
              </Text>
            )}
          {selection.disposition === "legislation" &&
            canSelectExistingParent && (
              <>
                <Text font="secondary-action">
                  Attach to existing PC Külliyatı legislation
                </Text>
                <div className="flex flex-wrap gap-2">
                  <InputTypeIn
                    aria-label={`Search existing parent for ${document.display_name}`}
                    placeholder="Search by legislation title"
                    maxLength={500}
                    value={query}
                    variant={disabled || saving ? "disabled" : "primary"}
                    onChange={(event) => setQuery(event.target.value)}
                  />
                  <Button
                    prominence="secondary"
                    disabled={disabled || saving}
                    onClick={() => setSearch(query.trim())}
                  >
                    Search existing parent
                  </Button>
                </div>
                {candidateError && (
                  <Text color="status-error-05">
                    {candidateError instanceof Error
                      ? candidateError.message
                      : "Existing parent search failed."}
                  </Text>
                )}
                {!candidates && !candidateError && (
                  <Text>Searching existing legislation…</Text>
                )}
                {candidates?.length === 0 && (
                  <Text font="secondary-body">
                    No eligible completed parent found in PC Külliyatı.
                  </Text>
                )}
                {candidates?.map((candidate) => (
                  <div
                    key={candidate.user_file_id}
                    className="flex flex-col gap-1 border-t border-border-02 pt-2"
                  >
                    <Text font="secondary-action">
                      {candidate.display_name}
                    </Text>
                    <Text
                      font="secondary-body"
                      color="text-03"
                    >{`${candidate.chunk_count} existing chunks${candidate.source_url ? ` · ${candidate.source_url}` : ""}`}</Text>
                    <div>
                      <Button
                        prominence={
                          selection.existing_parent_file_id ===
                          candidate.user_file_id
                            ? "primary"
                            : "secondary"
                        }
                        disabled={disabled || saving}
                        onClick={() => {
                          setSelection({
                            disposition: "legislation",
                            duplicate_of_document_id: null,
                            existing_parent_file_id: candidate.user_file_id,
                            reason: "",
                          });
                          setSelectedParent(candidate);
                        }}
                      >{`Select existing parent ${candidate.display_name}`}</Button>
                    </div>
                  </div>
                ))}
                {selection.existing_parent_file_id && (
                  <>
                    <Text font="secondary-action">{`Selected existing parent: ${selectedParent?.display_name ?? selection.existing_parent_file_id}`}</Text>
                    <div>
                      <Button
                        prominence="tertiary"
                        disabled={disabled || saving}
                        onClick={() => {
                          setSelection({
                            ...selection,
                            existing_parent_file_id: null,
                            reason: "",
                          });
                          setSelectedParent(null);
                        }}
                      >
                        No existing parent
                      </Button>
                    </div>
                  </>
                )}
              </>
            )}
          {dirty && (
            <>
              <InputTextArea
                aria-label={`Document identity review reason for ${document.display_name}`}
                value={selection.reason}
                rows={3}
                variant={disabled || saving ? "disabled" : "primary"}
                onChange={(event) =>
                  setSelection({ ...selection, reason: event.target.value })
                }
              />
              {validation && <Text color="status-error-05">{validation}</Text>}
              <div className="flex flex-wrap gap-2">
                <Button
                  disabled={disabled || saving || Boolean(validation)}
                  onClick={() => void save()}
                >
                  {saving
                    ? "Saving document identity…"
                    : "Save document identity review"}
                </Button>
                <Button
                  prominence="secondary"
                  disabled={disabled || saving}
                  onClick={reset}
                >
                  Discard document identity changes
                </Button>
              </div>
            </>
          )}
          {error && <Text color="status-error-05">{error}</Text>}
        </>
      )}
      {openParent && (
        <ExistingParentPreview
          key={openParent.user_file_id}
          parent={openParent}
        />
      )}
    </section>
  );
}
