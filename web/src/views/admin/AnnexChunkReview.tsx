"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { Button, Text } from "@opal/components";
import { toast } from "@opal/layouts";
import ChunkChangeCard from "@/sections/cards/ChunkChangeCard";
import AnnexChangeReview from "@/views/admin/AnnexChangeReview";
import {
  type AnnexReview,
  approveAnnexReview,
  rejectAnnexReview,
  retryAnnexReview,
  getAnnexReview,
  getAnnexChunkPage,
  prepareAnnexSelection,
  getAnnexEvidenceUrl,
} from "@/lib/regulatory/amendments";

interface AnnexChunkReviewProps {
  review: AnnexReview;
  onUpdated: (review: AnnexReview) => void;
}

export default function AnnexChunkReview({
  review,
  onUpdated,
}: AnnexChunkReviewProps) {
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [details, setDetails] = useState(false);
  const { data, error, mutate } = useSWR(
    ["annex-chunks", review.id, review.review_sha256, offset],
    () => getAnnexChunkPage(review, offset),
    { refreshInterval: 5000, revalidateOnFocus: true }
  );
  const { data: detail, error: detailError } = useSWR(
    details ? ["annex-details", review.id, review.review_sha256] : null,
    () => getAnnexReview(review)
  );
  useEffect(() => {
    setSelected(new Set());
    setOffset(0);
    setDetails(false);
  }, [review.id, review.review_sha256]);
  const label = review.review_payload.annex_label.replace(/^ek:/i, "EK-");
  const canSelect =
    review.publication_generation === 0 &&
    !["approved", "rejected"].includes(review.status) &&
    !review.review_payload.date_resolution?.effective_end_date;
  async function run(action: () => Promise<AnnexReview>) {
    setBusy(true);
    try {
      await action();
      await mutate();
      setSelected(new Set());
    } catch (failure) {
      toast.error(
        failure instanceof Error
          ? failure.message
          : "Could not update chunk review"
      );
    } finally {
      setBusy(false);
    }
  }
  function toggle(id: string) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }
  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Text
          as="h2"
          font="heading-h3"
        >{`${label} · ${data?.total ?? "…"} chunk changes`}</Text>
        <Button prominence="tertiary" onClick={() => setDetails(!details)}>
          {details ? "Hide source details" : "Source details / Edit"}
        </Button>
      </div>
      {error && <Text color="text-05">{error.message}</Text>}
      {review.review_payload.issues.map((issue) => (
        <Text key={issue}>
          {issue === "source_package_missing"
            ? "Source package missing"
            : issue.replaceAll("_", " ")}
        </Text>
      ))}
      {!data && !error && <Text>Loading chunk changes…</Text>}
      {data?.items.map((item) => {
        const decision = item.selection;
        const preparing =
          decision &&
          ["queued", "running"].includes(decision.preparation?.status ?? "");
        const counts = decision?.review_payload.publication?.counts;
        const ready =
          decision?.status === "pending" &&
          !preparing &&
          decision.review_payload.impact?.ready &&
          decision.review_payload.publication &&
          decision.review_payload.impact_strategy === "source_dependencies_v1";
        return (
          <ChunkChangeCard
            key={item.id}
            title={`${item.position + 1}. ${item.operation} · ${item.new_chunks[0]?.heading_path.join(" › ") || item.old_chunks[0]?.heading_path.join(" › ") || label}`}
            before={item.old_chunks.map((chunk) => chunk.text)}
            after={item.new_chunks.map((chunk) => chunk.text)}
            beforeImages={item.old_image_evidence_ids.map((id) =>
              getAnnexEvidenceUrl(review.batch_id, review.id, id)
            )}
            afterImages={item.new_image_evidence_ids.map((id) =>
              getAnnexEvidenceUrl(review.batch_id, review.id, id)
            )}
            actions={
              <>
                {!decision && canSelect && (
                  <>
                    <Button
                      disabled={busy}
                      prominence={
                        selected.has(item.id) ? "primary" : "secondary"
                      }
                      onClick={() => toggle(item.id)}
                    >
                      {selected.has(item.id) ? "Selected" : "Select"}
                    </Button>
                    <Button
                      disabled={busy}
                      onClick={() =>
                        run(() => prepareAnnexSelection(review, [item.id]))
                      }
                    >
                      Prepare this change
                    </Button>
                  </>
                )}
                {ready && decision && (
                  <Button
                    disabled={busy}
                    onClick={() =>
                      run(() =>
                        approveAnnexReview(
                          decision.batch_id,
                          decision.id,
                          decision.review_sha256
                        )
                      )
                    }
                  >
                    Approve selection
                  </Button>
                )}
                {decision &&
                  !preparing &&
                  decision.publication_generation === 0 &&
                  ["pending", "blocked"].includes(decision.status) && (
                    <Button
                      disabled={busy}
                      variant="danger"
                      prominence="secondary"
                      onClick={() =>
                        run(() =>
                          rejectAnnexReview(
                            decision.batch_id,
                            decision.id,
                            decision.review_sha256
                          )
                        )
                      }
                    >
                      Reject selection
                    </Button>
                  )}
                {decision &&
                  !preparing &&
                  ["blocked", "failed", "rejected"].includes(
                    decision.status
                  ) && (
                    <Button
                      disabled={busy}
                      prominence="secondary"
                      onClick={() =>
                        run(() =>
                          retryAnnexReview(
                            decision.batch_id,
                            decision.id,
                            decision.review_sha256
                          )
                        )
                      }
                    >
                      Recheck selection
                    </Button>
                  )}
              </>
            }
          >
            {(item.old_chunks.length > 1 || item.new_chunks.length > 1) && (
              <Text font="secondary-body">
                These linked chunks are applied together.
              </Text>
            )}
            {decision && (
              <Text font="secondary-body">
                {preparing
                  ? `Preparing ${decision.preparation?.completed_chunks ?? 0}/${decision.preparation?.total_chunks ?? 0}`
                  : decision.status}
              </Text>
            )}
            {counts && (
              <Text font="secondary-body">{`${counts.canonical_changes} changes · ${counts.embeddings} new embedding inputs · ${counts.preserved_vectors ?? counts.exact_vector_reuses} preserved vectors · ${counts.metadata_updates ?? 0} metadata updates`}</Text>
            )}
            {decision?.error_message && <Text>{decision.error_message}</Text>}
            {decision?.review_payload.impact?.ready === false && (
              <Text font="secondary-body">
                {`${decision.dependency_summary?.unresolved ?? 0} source dependencies need verification before publication. ${decision.dependency_summary?.reasons.join("; ") ?? ""}`}
              </Text>
            )}
          </ChunkChangeCard>
        );
      })}
      {selected.size > 0 && (
        <Button
          disabled={busy}
          onClick={() =>
            run(() => prepareAnnexSelection(review, Array.from(selected)))
          }
        >{`Prepare ${selected.size} selected changes`}</Button>
      )}
      {data && data.total > 10 && (
        <div className="flex items-center gap-2">
          <Button
            prominence="secondary"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - 10))}
          >
            Previous
          </Button>
          <Text>{`${offset + 1}–${Math.min(offset + 10, data.total)} / ${data.total}`}</Text>
          <Button
            prominence="secondary"
            disabled={offset + 10 >= data.total}
            onClick={() => setOffset(offset + 10)}
          >
            Next
          </Button>
        </div>
      )}
      {details &&
        (detail ? (
          <AnnexChangeReview
            review={detail}
            onUpdated={onUpdated}
            decisionsLocked={Boolean(data?.selection_count)}
          />
        ) : (
          <Text>
            {detailError
              ? "Could not load source details"
              : "Loading source details…"}
          </Text>
        ))}
    </section>
  );
}
