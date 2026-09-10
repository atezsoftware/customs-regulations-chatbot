"use client";

import { useEffect, useMemo, useState } from "react";

import {
  Button,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import { toast } from "@opal/layouts";
import {
  type AnnexElementCorrection,
  type AnnexExtraction,
  type AnnexLocator,
  type AnnexReview,
  approveAnnexReview,
  editAnnexReview,
  getAnnexEvidenceUrl,
  rejectAnnexReview,
  retryAnnexReview,
} from "@/lib/regulatory/amendments";

type ReviewAction = (review: AnnexReview) => Promise<AnnexReview>;
type EditAction = (
  corrections: AnnexElementCorrection[]
) => Promise<AnnexReview>;

interface AnnexChangeReviewProps {
  review: AnnexReview;
  onUpdated: (review: AnnexReview) => void;
  onApprove?: ReviewAction;
  onReject?: ReviewAction;
  onRetry?: ReviewAction;
  onEdit?: EditAction;
}

const STATUS_LABELS: Record<AnnexReview["status"], string> = {
  pending: "Pending review",
  blocked: "Blocked",
  approving: "Approval queued",
  preparing: "Preparing publication",
  publishing: "Publishing",
  approved: "Approved",
  rejected: "Rejected",
  failed: "Failed",
};

const STATUS_DESCRIPTIONS: Record<AnnexReview["status"], string> = {
  pending: "The complete frozen annex group is awaiting a decision.",
  blocked:
    "The complete annex group cannot be approved until its blocking evidence checks pass.",
  approving: "Approval intent is durable; publication has not started yet.",
  preparing:
    "The approved frozen revision is preparing verified publication inputs.",
  publishing:
    "The frozen revision is being written and verified across its complete publication scope.",
  approved:
    "The complete publication was verified and the annex revision is active.",
  rejected: "The complete annex group was rejected without publication.",
  failed:
    "Publication or preparation failed; the annex revision is not approved.",
};

const ISSUE_LABELS: Record<string, string> = {
  source_package_missing: "Source package missing",
  temporary_annex_publication_contract_required:
    "Temporary legal window needs source authority",
};

function displayIssue(issue: string) {
  return ISSUE_LABELS[issue] ?? issue.replaceAll("_", " ");
}

function displayAnnexLabel(label: string) {
  const normalized = /^ek:(.+)$/i.exec(label);
  return normalized ? `EK-${normalized[1]}` : label;
}

function correctionsEqual(
  left: AnnexElementCorrection[],
  right: AnnexElementCorrection[]
) {
  if (left.length !== right.length) return false;
  const byPosition = (items: AnnexElementCorrection[]) =>
    [...items].sort((a, b) => a.position - b.position);
  const sortedLeft = byPosition(left);
  const sortedRight = byPosition(right);
  return sortedLeft.every((correction, index) => {
    const other = sortedRight[index];
    return (
      other !== undefined &&
      correction.position === other.position &&
      correction.before_text === other.before_text &&
      correction.corrected_text === other.corrected_text &&
      correction.reason === other.reason
    );
  });
}

function retryResultMessage(review: AnnexReview) {
  if (review.status === "blocked") {
    return "Blocking checks remain; the review was not resumed.";
  }
  if (review.status === "approved") {
    return "The annex group is already approved.";
  }
  if (review.status === "failed") {
    return "The retry finished with a failure; review the current error.";
  }
  if (review.status === "rejected") {
    return "The annex group remains rejected.";
  }
  if (review.status === "pending") {
    return "The annex group is ready for review.";
  }
  return review.publication_generation > 0
    ? `Publication retry is ${STATUS_LABELS[review.status].toLowerCase()} for the same frozen revision.`
    : `Review preparation is ${STATUS_LABELS[review.status].toLowerCase()} without changing its frozen identity.`;
}

function shorten(value: string | null, length = 12) {
  if (!value) return "—";
  return value.length > length ? `${value.slice(0, length)}…` : value;
}

function locatorLabel(locator: AnnexLocator) {
  const parts = [
    locator.page ? `page ${locator.page}` : null,
    locator.sheet ? `sheet ${locator.sheet}` : null,
    locator.cell ? `cell ${locator.cell}` : null,
    locator.row ? `row ${locator.row}` : null,
    locator.column ? `column ${locator.column}` : null,
    locator.path ? `path ${locator.path}` : null,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(" · ") : "whole source";
}

function ReviewSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2 rounded-12 border border-border-02 bg-background-neutral-00 p-3">
      <Text as="h3" font="main-ui-action" color="text-04">
        {title}
      </Text>
      {children}
    </section>
  );
}

function ExtractionPanel({
  title,
  extraction,
}: {
  title: string;
  extraction: AnnexExtraction | null;
}) {
  if (!extraction) {
    return (
      <div className="min-w-0 flex-1 rounded-08 border border-border-01 p-3">
        <Text font="main-ui-body" color="text-03">
          {`${title}: unavailable`}
        </Text>
      </div>
    );
  }
  return (
    <div className="min-w-0 flex-1 rounded-08 border border-border-01 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Text font="main-ui-action" color="text-04">
          {title}
        </Text>
        <Tag title={extraction.mime_type} truncate />
      </div>
      <Text as="p" font="secondary-body" color="text-03">
        {`Frozen extraction ${shorten(extraction.source_sha256)} · ${extraction.elements.length} elements`}
      </Text>
      {extraction.evidence_view?.pages.map((page) => (
        <Text
          key={`${page.parent_index}-${page.view_page}`}
          as="p"
          font="secondary-body"
          color="text-03"
        >
          {`Original page ${page.original_page} → review page ${page.view_page}`}
        </Text>
      ))}
      {extraction.evidence_view?.element_mappings.map((mapping) => (
        <Text
          key={`${mapping.parent_index}-${mapping.original_position}-${mapping.view_position}`}
          as="p"
          font="secondary-body"
          color="text-03"
        >
          {`Original element ${mapping.original_position} → review element ${mapping.view_position} · ${locatorLabel(mapping.original_locator)}`}
        </Text>
      ))}
      <div className="mt-2 flex flex-col gap-2">
        {extraction.elements.map((element, position) => (
          <div
            key={`${position}-${element.semantic_key ?? element.text}`}
            className="rounded-08 bg-background-tint-01 p-2"
          >
            <Text as="p" font="main-ui-body" color="text-05">
              {element.text || "(visual element)"}
            </Text>
            <Text as="p" font="secondary-body" color="text-03">
              {`${element.kind} · ${element.extraction_method} · ${locatorLabel(element.locator)}`}
            </Text>
            {element.formula && (
              <Text as="p" font="secondary-body" color="text-03">
                {`Formula: ${element.formula}`}
              </Text>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function Count({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-08 border border-border-01 bg-background-tint-01 p-2">
      <Text as="p" font="secondary-body" color="text-03">
        {label}
      </Text>
      <Text font="main-ui-action" color="text-05">
        {String(value)}
      </Text>
    </div>
  );
}

function ImpactList({
  title,
  ids,
  reasons,
}: {
  title: string;
  ids: string[];
  reasons: Record<string, string[]>;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Text font="main-ui-action" color="text-04">
        {title}
      </Text>
      {ids.length === 0 ? (
        <Text font="secondary-body" color="text-03">
          None
        </Text>
      ) : (
        ids.map((id) => (
          <div key={id} className="rounded-08 bg-background-tint-01 p-2">
            <Text as="p" font="secondary-body" color="text-05">
              {id}
            </Text>
            {(reasons[id] ?? []).map((reason) => (
              <Text key={reason} as="p" font="secondary-body" color="text-03">
                {reason}
              </Text>
            ))}
          </div>
        ))
      )}
    </div>
  );
}

export default function AnnexChangeReview({
  review,
  onUpdated,
  onApprove,
  onReject,
  onRetry,
  onEdit,
}: AnnexChangeReviewProps) {
  const payload = review.review_payload;
  const [working, setWorking] = useState(false);
  const [corrections, setCorrections] = useState<AnnexElementCorrection[]>(
    payload.corrections
  );
  const editableElements =
    payload.raw_new_extraction?.elements ??
    payload.new_extraction?.elements ??
    [];

  useEffect(() => {
    setCorrections(payload.corrections);
  }, [payload.corrections, review.id, review.review_sha256]);

  const correctionByPosition = useMemo(
    () =>
      new Map(
        corrections.map((correction) => [correction.position, correction])
      ),
    [corrections]
  );
  const canEdit =
    review.publication_generation === 0 &&
    ["pending", "blocked", "rejected", "failed"].includes(review.status) &&
    editableElements.length > 0;
  const correctionsValid = corrections.every(
    (correction) =>
      correction.corrected_text.trim().length > 0 &&
      correction.reason.trim().length > 0 &&
      correction.corrected_text !== correction.before_text
  );
  const correctionsDirty = !correctionsEqual(corrections, payload.corrections);
  const frozenReviewReadyToApprove =
    review.status === "pending" &&
    payload.issues.length === 0 &&
    payload.impact?.ready === true &&
    payload.publication !== null;
  const reconciliationRequired =
    review.status === "failed" &&
    review.error_message?.includes("manual provider reconciliation") === true;

  const runAction = async (
    action: () => Promise<AnnexReview>,
    success: string | ((updated: AnnexReview) => string)
  ) => {
    setWorking(true);
    try {
      const updated = await action();
      onUpdated(updated);
      toast.info(typeof success === "string" ? success : success(updated));
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Annex review action failed."
      );
    } finally {
      setWorking(false);
    }
  };

  const updateCorrection = (
    position: number,
    field: "corrected_text" | "reason",
    value: string
  ) => {
    const element = editableElements[position];
    if (!element) return;
    setCorrections((current) => {
      const existing = current.find((item) => item.position === position);
      const next = {
        position,
        before_text: element.text,
        corrected_text: existing?.corrected_text ?? element.text,
        reason: existing?.reason ?? "",
        [field]: value,
      };
      return [
        ...current.filter((item) => item.position !== position),
        next,
      ].sort((left, right) => left.position - right.position);
    });
  };

  return (
    <article className="flex flex-col gap-3 rounded-12 border border-border-02 bg-background-neutral-01 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <Text as="h2" font="heading-h3" color="text-05">
            {`Annex ${displayAnnexLabel(payload.annex_label)}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Review revision ${review.review_revision}`}
          </Text>
        </div>
        <Tag
          title={STATUS_LABELS[review.status]}
          error={review.status === "blocked" || review.status === "failed"}
        />
      </div>

      <details className="rounded-08 bg-background-tint-01 p-2">
        <summary className="cursor-pointer">
          <Text font="main-ui-action" color="text-04">
            Audit details
          </Text>
        </summary>
        <div className="mt-2 flex flex-col gap-1">
          <Text as="p" font="secondary-body" color="text-03">
            {`Annex key: ${payload.annex_label}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Logical group: ${review.logical_group_id}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Publication generation: ${review.publication_generation}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Frozen review hash: ${review.review_sha256}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Package: ${payload.source_package_id ?? "not attached"}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Package manifest: ${payload.source_manifest_sha256 ?? "unavailable"}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Original source text: ${payload.original_source_text_sha256 ?? "unavailable"}`}
          </Text>
          <Text as="p" font="secondary-body" color="text-03">
            {`Reviewed source text: ${payload.source_text_sha256 ?? "unavailable"}`}
          </Text>
        </div>
      </details>

      <div role="status" className="rounded-08 border border-border-01 p-2">
        <Text as="p" font="main-ui-body" color="text-05">
          {STATUS_DESCRIPTIONS[review.status]}
        </Text>
      </div>

      {review.error_message && (
        <div
          role="alert"
          className="rounded-08 border border-status-error-02 bg-status-error-01 p-3"
        >
          <Text as="p" font="main-ui-body" color="status-error-05">
            {review.error_message}
          </Text>
        </div>
      )}

      <ReviewSection title="Grouped instructions">
        {payload.instruction_texts.map((instruction, index) => (
          <Text
            key={payload.instruction_indices[index]}
            as="p"
            font="main-ui-body"
            color="text-05"
          >
            {`${(payload.instruction_indices[index] ?? index) + 1}. ${instruction}`}
          </Text>
        ))}
      </ReviewSection>

      {payload.issues.length > 0 && (
        <ReviewSection title="Blocking checks">
          {payload.issues.map((issue) => (
            <Text
              key={issue}
              as="p"
              font="main-ui-body"
              color="status-error-05"
            >
              {displayIssue(issue)}
            </Text>
          ))}
        </ReviewSection>
      )}

      <ReviewSection title="Frozen original evidence">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <ExtractionPanel title="OLD" extraction={payload.old_extraction} />
          <ExtractionPanel title="NEW" extraction={payload.new_extraction} />
        </div>
        <div className="flex flex-wrap gap-2">
          {payload.evidence.map((evidence) => (
            <Button
              key={evidence.id}
              href={getAnnexEvidenceUrl(
                review.batch_id,
                review.id,
                evidence.id
              )}
              size="sm"
              prominence="secondary"
            >
              {`${evidence.side.toUpperCase()} ${evidence.kind} · ${locatorLabel(evidence.locator)}`}
            </Button>
          ))}
        </div>
      </ReviewSection>

      {payload.comparison && (
        <ReviewSection title="Detected changes">
          {payload.comparison.changes.map((change, index) => (
            <div
              key={`${change.operation}-${index}`}
              className="rounded-08 border border-border-01 p-2"
            >
              <div className="flex items-center gap-2">
                <Tag title={change.operation} />
                {change.uncertain && <Tag title="Uncertain" error />}
              </div>
              <Text as="p" font="main-ui-body" color="text-05">
                {change.explanation}
              </Text>
              <Text as="p" font="secondary-body" color="text-03">
                {`OLD: ${change.old.map((item) => `${item.text} (${locatorLabel(item.locator)})`).join(" · ") || "none"}`}
              </Text>
              <Text as="p" font="secondary-body" color="text-03">
                {`NEW: ${change.new.map((item) => `${item.text} (${locatorLabel(item.locator)})`).join(" · ") || "none"}`}
              </Text>
            </div>
          ))}
        </ReviewSection>
      )}

      {payload.patch_plan && (
        <ReviewSection title="Canonical result">
          {payload.patch_plan.patches.map((patch, index) => (
            <div
              key={`${patch.old_chunk_id ?? "insert"}-${index}`}
              className="grid grid-cols-1 gap-2 lg:grid-cols-2"
            >
              <div className="rounded-08 bg-background-tint-01 p-2">
                <Text as="p" font="secondary-body" color="text-03">
                  Before
                </Text>
                <Text as="p" font="main-ui-body" color="text-05">
                  {patch.old_text ?? "(new content)"}
                </Text>
              </div>
              <div className="rounded-08 bg-background-tint-01 p-2">
                <Text as="p" font="secondary-body" color="text-03">
                  After
                </Text>
                <Text as="p" font="main-ui-body" color="text-05">
                  {patch.new_text ?? "(removed)"}
                </Text>
              </div>
            </div>
          ))}
          {payload.source_only_canonical_ids.length > 0 && (
            <Text as="p" font="secondary-body" color="text-03">
              {`Source-only updates: ${payload.source_only_canonical_ids.join(", ")}`}
            </Text>
          )}
        </ReviewSection>
      )}

      {payload.impact && (
        <ReviewSection title="Context and embedding impact">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <ImpactList
              title="Direct legal changes"
              ids={payload.impact.direct_canonical_changes}
              reasons={payload.impact.reasons}
            />
            <ImpactList
              title="Context reevaluation candidates"
              ids={payload.impact.contextual_candidates}
              reasons={payload.impact.reasons}
            />
            <ImpactList
              title="Final embedding changes"
              ids={payload.impact.embedding_changes}
              reasons={payload.impact.reasons}
            />
            <ImpactList
              title="Context only"
              ids={payload.impact.context_only}
              reasons={payload.impact.reasons}
            />
            <ImpactList
              title="Metadata only"
              ids={payload.impact.metadata_only}
              reasons={payload.impact.reasons}
            />
            <ImpactList
              title="Unchanged after full proof"
              ids={payload.impact.unchanged}
              reasons={payload.impact.reasons}
            />
          </div>
        </ReviewSection>
      )}

      {payload.publication && (
        <ReviewSection title="Final publication scope">
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <Count
              label="Canonical changes"
              value={payload.publication.counts.canonical_changes}
            />
            <Count
              label="Context consumers"
              value={payload.publication.counts.context_consumers}
            />
            <Count
              label="Embeddings"
              value={payload.publication.counts.embeddings}
            />
            <Count
              label="Exact vector reuse"
              value={payload.publication.counts.exact_vector_reuses}
            />
            <Count
              label="Historical projections"
              value={payload.publication.counts.historical_projections}
            />
            <Count
              label="Retired projections"
              value={payload.publication.counts.retired_projections}
            />
            <Count
              label="Total projections"
              value={payload.publication.counts.total_projections}
            />
            <Count label="Indexes" value={payload.publication.indexes.length} />
          </div>
          <Text as="p" font="main-ui-body" color="text-05">
            {`Effective windows: ${payload.publication.effective_dates.join(" → ") || "undated"}`}
          </Text>
          {payload.date_resolution && (
            <Text as="p" font="secondary-body" color="text-03">
              {`Legal window: ${payload.date_resolution.effective_start_date ?? "open start"} → ${payload.date_resolution.effective_end_date ?? "ongoing"} · ${payload.date_resolution.rationale}`}
            </Text>
          )}
          {payload.after_window_authority && (
            <Text as="p" font="secondary-body" color="text-03">
              {`After-window authority: ${payload.after_window_authority.kind.replaceAll("_", " ")} on ${payload.after_window_authority.effective_date} · ${payload.after_window_authority.source_quote}`}
            </Text>
          )}
        </ReviewSection>
      )}

      {(editableElements.length > 0 ||
        payload.corrections.length > 0 ||
        payload.correction_reconciliation) && (
        <ReviewSection title="Evidence-backed correction">
          {editableElements.map((element, position) => (
            <Text
              key={`raw-${position}`}
              as="p"
              font="secondary-body"
              color="text-03"
            >
              {`Raw extraction ${position}: ${element.text}`}
            </Text>
          ))}
          {payload.corrections.map((correction) => (
            <div
              key={`applied-${correction.position}`}
              className="rounded-08 bg-background-tint-01 p-2"
            >
              <Text as="p" font="main-ui-body" color="text-05">
                {`Applied correction ${correction.position}: ${correction.corrected_text}`}
              </Text>
              <Text as="p" font="secondary-body" color="text-03">
                {`Frozen raw value: ${correction.before_text} · ${correction.reason}`}
              </Text>
            </div>
          ))}
          {payload.correction_reconciliation && (
            <Text as="p" font="secondary-body" color="text-03">
              {`Evidence reconciliation: ${payload.correction_reconciliation.supported ? "supported" : "blocked"} · ${payload.correction_reconciliation.rationale}`}
            </Text>
          )}
          {canEdit && (
            <>
              <Text as="p" font="secondary-body" color="text-03">
                Editing NEW text creates a new immutable review revision and
                revalidates the complete group against frozen evidence.
              </Text>
              {editableElements.map((element, position) => {
                const correction = correctionByPosition.get(position);
                return (
                  <div key={position} className="flex flex-col gap-2">
                    <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                      <InputTextArea
                        aria-label={`Correct NEW element ${position}`}
                        value={correction?.corrected_text ?? element.text}
                        onChange={(event) =>
                          updateCorrection(
                            position,
                            "corrected_text",
                            event.target.value
                          )
                        }
                        rows={3}
                      />
                      <InputTypeIn
                        aria-label={`Correction reason for element ${position}`}
                        value={correction?.reason ?? ""}
                        onChange={(event) =>
                          updateCorrection(
                            position,
                            "reason",
                            event.target.value
                          )
                        }
                        placeholder="How the frozen original supports this correction"
                      />
                    </div>
                    {correction && (
                      <div className="flex justify-end">
                        <Button
                          size="sm"
                          prominence="secondary"
                          onClick={() =>
                            setCorrections((current) =>
                              current.filter(
                                (item) => item.position !== position
                              )
                            )
                          }
                        >
                          {`Remove correction for element ${position}`}
                        </Button>
                      </div>
                    )}
                  </div>
                );
              })}
              {correctionsDirty && (
                <div
                  role="status"
                  className="rounded-08 border border-status-warning-02 bg-status-warning-01 p-2"
                >
                  <Text as="p" font="main-ui-body" color="text-05">
                    Local correction changes are not submitted. Revalidate the
                    full group or discard them before approving this frozen
                    revision.
                  </Text>
                </div>
              )}
              <div className="flex flex-wrap justify-end gap-2">
                {correctionsDirty && (
                  <Button
                    prominence="secondary"
                    onClick={() => setCorrections(payload.corrections)}
                  >
                    Discard local edits
                  </Button>
                )}
                <Button
                  prominence="secondary"
                  disabled={working || !correctionsDirty || !correctionsValid}
                  onClick={() =>
                    void runAction(
                      () =>
                        onEdit
                          ? onEdit(corrections)
                          : editAnnexReview(
                              review.batch_id,
                              review.id,
                              review.review_sha256,
                              corrections
                            ),
                      "A new immutable review revision was created."
                    )
                  }
                >
                  Revalidate full group
                </Button>
              </div>
            </>
          )}
        </ReviewSection>
      )}

      <div className="flex flex-wrap justify-end gap-2">
        {review.status === "pending" && (
          <Button
            variant="danger"
            prominence="secondary"
            disabled={working}
            onClick={() =>
              void runAction(
                () =>
                  onReject
                    ? onReject(review)
                    : rejectAnnexReview(
                        review.batch_id,
                        review.id,
                        review.review_sha256
                      ),
                "Annex group rejected."
              )
            }
          >
            Reject group
          </Button>
        )}
        {frozenReviewReadyToApprove && (
          <Button
            disabled={working || correctionsDirty}
            onClick={() =>
              void runAction(
                () =>
                  onApprove
                    ? onApprove(review)
                    : approveAnnexReview(
                        review.batch_id,
                        review.id,
                        review.review_sha256
                      ),
                "Approval queued; publication status will update here."
              )
            }
          >
            Approve group
          </Button>
        )}
        {(["blocked", "rejected"].includes(review.status) ||
          (review.status === "failed" && !reconciliationRequired)) && (
          <Button
            disabled={working}
            onClick={() =>
              void runAction(
                () =>
                  onRetry
                    ? onRetry(review)
                    : retryAnnexReview(
                        review.batch_id,
                        review.id,
                        review.review_sha256
                      ),
                retryResultMessage
              )
            }
          >
            {review.publication_generation > 0
              ? "Retry publication"
              : "Retry review"}
          </Button>
        )}
      </div>
    </article>
  );
}
