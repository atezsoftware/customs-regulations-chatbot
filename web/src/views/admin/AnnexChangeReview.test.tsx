import { render, screen, setupUser } from "@tests/setup/test-utils";
import { toast } from "@opal/layouts";

import type { AnnexReview } from "@/lib/regulatory/amendments";
import AnnexChangeReview from "@/views/admin/AnnexChangeReview";

const locator = {
  page: 2,
  sheet: null,
  cell: null,
  row: null,
  column: null,
  path: null,
  original_box: [10, 20, 100, 40] as [number, number, number, number],
  normalized_box: [0.1, 0.2, 0.8, 0.3] as [number, number, number, number],
  original_width: 612,
  original_height: 792,
  coordinate_system: "top_left_points" as const,
};

function reviewFixture(overrides: Partial<AnnexReview> = {}): AnnexReview {
  return {
    id: "review-1",
    logical_group_id: "group-1",
    review_revision: 3,
    batch_id: 42,
    status: "pending",
    review_sha256: "a".repeat(64),
    publication_generation: 0,
    error_message: null,
    created_at: "2026-09-10T00:00:00Z",
    review_payload: {
      instruction_indices: [0, 1],
      instruction_texts: ["Replace EK-1.", "Keep the temporary window."],
      annex_label: "EK-1",
      effective_date: "2026-10-01",
      source_package_id: "package-1",
      source_text_sha256: "b".repeat(64),
      source_manifest_sha256: "c".repeat(64),
      original_source_text_sha256: "d".repeat(64),
      submitted_source_text: "EK-1 ekteki şekilde değiştirilmiştir.",
      raw_new_extraction: null,
      old_evidence_kind: "visual",
      old_extraction: {
        evidence_view: {
          sha256: "old-view",
          label: "EK-1",
          parents: [
            {
              file_id: "old-file",
              sha256: "old-sha",
              mime_type: "application/pdf",
              extraction_sha256: "old-extraction",
              element_count: 1,
              page_count: 4,
              canonical_chunk_ids: ["old-chunk"],
            },
          ],
          pages: [
            {
              parent_index: 0,
              original_page: 2,
              view_page: 1,
              normalized_box: [0, 0, 1, 1],
            },
          ],
          selected_positions: [0],
          element_mappings: [
            {
              parent_index: 0,
              original_position: 8,
              original_locator: locator,
              view_position: 0,
            },
          ],
          boundary_positions: [0],
          selection_method: "native_boundaries",
        },
        page_count: 1,
        schema_version: 1,
        source_sha256: "old-source",
        mime_type: "application/pdf",
        elements: [
          {
            extraction_method: "native",
            canonical_chunk_id: "old-chunk",
            bound_to_regulatory_chunk_id: "old-chunk",
            canonical_role: "authoritative",
            kind: "table_cell",
            text: "Old rate: 10%",
            semantic_key: "rate",
            locator,
            formula: null,
            value: "10%",
            evidence_kind: "original",
            status: "readable",
            issues: [],
            aggregate: false,
            image_file_id: null,
            source_asset_id: null,
          },
        ],
        issues: [],
      },
      new_extraction: {
        evidence_view: null,
        page_count: 1,
        schema_version: 1,
        source_sha256: "new-source",
        mime_type:
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        elements: [
          {
            extraction_method: "native",
            canonical_chunk_id: null,
            bound_to_regulatory_chunk_id: null,
            canonical_role: "authoritative",
            kind: "table_cell",
            text: "New rate: 12%",
            semantic_key: "rate",
            locator: { ...locator, page: null, sheet: "EK-1", cell: "B4" },
            formula: null,
            value: "12%",
            evidence_kind: "original",
            status: "readable",
            issues: [],
            aggregate: false,
            image_file_id: null,
            source_asset_id: "asset-1",
          },
        ],
        issues: [],
      },
      corrections: [],
      correction_reconciliation: null,
      baseline_scope: [],
      comparison: {
        changes: [
          {
            operation: "replace",
            old: [{ position: 0, text: "Old rate: 10%", locator }],
            new: [
              {
                position: 0,
                text: "New rate: 12%",
                locator: { ...locator, page: null, sheet: "EK-1", cell: "B4" },
              },
            ],
            explanation: "The customs rate changed.",
            uncertain: false,
          },
        ],
        issues: [],
        ready: true,
      },
      patch_plan: {
        patches: [
          {
            old_chunk_id: "old-chunk",
            old_text: "Old canonical rate: 10%",
            new_text: "New canonical rate: 12%",
            old_positions: [0],
            new_positions: [0],
            operation: "replace",
          },
        ],
        direct_canonical_changes: ["old-chunk"],
        metadata_only: ["meta-chunk"],
        retire_history: [],
        unchanged: ["unchanged-chunk"],
        issues: [],
        ready: true,
      },
      items: [],
      impact: {
        direct_canonical_changes: ["old-chunk"],
        contextual_candidates: ["consumer-1", "consumer-2"],
        embedding_changes: ["old-chunk", "consumer-1"],
        context_only: ["consumer-1"],
        metadata_only: ["meta-chunk"],
        retire_history: [],
        unchanged: ["unchanged-chunk"],
        reasons: {
          "consumer-1": ["context source changed"],
          "meta-chunk": ["metadata window changed"],
        },
        ready: true,
      },
      evidence: [
        {
          id: "evidence-old",
          side: "old",
          kind: "original",
          file_id: "old-file",
          sha256: "old-sha",
          mime_type: "application/pdf",
          byte_count: 120,
          parent_file_id: "old-file",
          parent_sha256: "old-sha",
          source_asset_id: null,
          locator,
        },
        {
          id: "evidence-new",
          side: "new",
          kind: "comparison_region",
          file_id: "new-region",
          sha256: "new-sha",
          mime_type: "image/png",
          byte_count: 220,
          parent_file_id: "new-file",
          parent_sha256: "new-parent-sha",
          source_asset_id: "asset-1",
          locator,
        },
      ],
      source_only_canonical_ids: ["source-only-1"],
      after_window_authority: {
        kind: "restore_predecessor",
        effective_date: "2027-01-01",
        source_text_sha256: "e".repeat(64),
        source_start: 3,
        source_end: 30,
        source_quote: "Previous schedule resumes.",
        predecessor_ids: ["old-chunk"],
        successor_ids: [],
      },
      publication: {
        artifact_file_id: "artifact-internal",
        artifact_sha256: "f".repeat(64),
        artifact_byte_count: 9000,
        scope: {
          tenant_id: "tenant",
          environment: "annex-local",
          database_identity: "database",
        },
        indexes: [
          {
            index_name: "primary",
            index_uuid: "index-1",
            search_settings_id: 1,
            model_provider: "provider",
            model_name: "model",
            vector_dimension: 768,
            embedding_config_sha256: "0".repeat(64),
            multitenant: true,
          },
        ],
        counts: {
          canonical_changes: 1,
          context_consumers: 2,
          embeddings: 2,
          exact_vector_reuses: 1,
          historical_projections: 2,
          retired_projections: 1,
          total_projections: 5,
        },
        effective_dates: ["2026-10-01", "2027-01-01"],
        projection_ids: ["projection-1"],
      },
      issues: [],
    },
    ...overrides,
  };
}

test("renders frozen evidence, separated impact, and final publication windows", async () => {
  const user = setupUser();
  render(<AnnexChangeReview review={reviewFixture()} onUpdated={jest.fn()} />);

  expect(screen.getByText("Old rate: 10%")).toBeVisible();
  expect(screen.getAllByText("New rate: 12%")[0]).toBeVisible();
  expect(screen.getByText(/Original page 2 → review page 1/)).not.toBeVisible();
  await user.click(screen.getAllByText("Extraction details")[0]!);
  expect(screen.getByText(/Original page 2 → review page 1/)).toBeVisible();
  expect(screen.getByText("Direct legal changes")).toBeVisible();
  expect(screen.getByText("Context reevaluation candidates")).toBeVisible();
  expect(screen.getByText("Exact vector reuse")).toBeVisible();
  expect(screen.getByText("Embeddings").parentElement).toHaveTextContent("2");
  expect(
    screen.getByText(/Effective windows: 2026-10-01 → 2027-01-01/)
  ).toBeVisible();
  expect(screen.getByRole("link", { name: /OLD original/ })).toHaveAttribute(
    "href",
    "/api/regulatory/amendments/batches/42/annex-groups/review-1/evidence/evidence-old"
  );
  expect(screen.queryByText("artifact-internal")).not.toBeInTheDocument();
});

test("keeps supplementary PDF extraction out of the content view without losing evidence", async () => {
  const user = setupUser();
  const review = reviewFixture();
  const extraction = review.review_payload.new_extraction!;
  const original = extraction.elements[0]!;
  extraction.elements.unshift({
    ...original,
    aggregate: true,
    text: "1GHMGKR*PMQR:>R'R",
  });
  const frozen = JSON.stringify(extraction);

  render(<AnnexChangeReview review={review} onUpdated={jest.fn()} />);

  expect(screen.getByText("1GHMGKR*PMQR:>R'R")).not.toBeVisible();
  expect(screen.getAllByText(original.text)[0]).toBeVisible();
  expect(
    screen.queryByRole("textbox", { name: "Correct NEW element 0" })
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("textbox", { name: "Correct NEW element 1" })
  ).toBeVisible();
  await user.click(screen.getAllByText("Extraction details")[1]!);
  expect(screen.getByText("1GHMGKR*PMQR:>R'R")).toBeVisible();
  expect(JSON.stringify(extraction)).toBe(frozen);
});

test("submits a corrected NEW element as a full immutable revalidation", async () => {
  const user = setupUser();
  const onEdit = jest
    .fn()
    .mockResolvedValue(reviewFixture({ review_revision: 4 }));
  render(
    <AnnexChangeReview
      review={reviewFixture()}
      onUpdated={jest.fn()}
      onEdit={onEdit}
    />
  );

  await user.clear(
    screen.getByRole("textbox", { name: "Correct NEW element 0" })
  );
  await user.type(
    screen.getByRole("textbox", { name: "Correct NEW element 0" }),
    "New rate: 11%"
  );
  await user.type(
    screen.getByRole("textbox", { name: "Correction reason for element 0" }),
    "Verified against the frozen cell."
  );
  await user.click(
    screen.getByRole("button", { name: "Revalidate full group" })
  );

  expect(onEdit).toHaveBeenCalledWith([
    {
      position: 0,
      before_text: "New rate: 12%",
      corrected_text: "New rate: 11%",
      reason: "Verified against the frozen cell.",
    },
  ]);
});

test("cannot approve the frozen hash while a local correction is unsubmitted", async () => {
  const user = setupUser();
  const onApprove = jest.fn().mockResolvedValue(reviewFixture());
  render(
    <AnnexChangeReview
      review={reviewFixture()}
      onUpdated={jest.fn()}
      onApprove={onApprove}
    />
  );

  expect(screen.getByRole("button", { name: "Approve group" })).toBeEnabled();
  await user.clear(
    screen.getByRole("textbox", { name: "Correct NEW element 0" })
  );
  await user.type(
    screen.getByRole("textbox", { name: "Correct NEW element 0" }),
    "New rate: 11%"
  );

  expect(
    screen.getByText(/local correction changes are not submitted/i)
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "Approve group" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Approve group" }));
  expect(onApprove).not.toHaveBeenCalled();
});

test("removes an applied correction through an empty immutable revalidation", async () => {
  const user = setupUser();
  const correctedReview = reviewFixture({
    review_payload: {
      ...reviewFixture().review_payload,
      raw_new_extraction: reviewFixture().review_payload.new_extraction,
      corrections: [
        {
          position: 0,
          before_text: "New rate: 12%",
          corrected_text: "New rate: 11%",
          reason: "Verified correction.",
        },
      ],
      correction_reconciliation: {
        supported: true,
        rationale: "Correction matches the frozen cell.",
        input_sha256: "c".repeat(64),
      },
    },
  });
  const onEdit = jest.fn().mockResolvedValue(
    reviewFixture({
      review_revision: 4,
      review_payload: {
        ...correctedReview.review_payload,
        corrections: [],
      },
    })
  );
  render(
    <AnnexChangeReview
      review={correctedReview}
      onUpdated={jest.fn()}
      onEdit={onEdit}
    />
  );

  await user.click(
    screen.getByRole("button", { name: "Remove correction for element 0" })
  );
  await user.click(
    screen.getByRole("button", { name: "Revalidate full group" })
  );

  expect(onEdit).toHaveBeenCalledWith([]);
});

test("keeps raw and applied correction provenance visible after approval", () => {
  const rawExtraction = reviewFixture().review_payload.new_extraction;
  render(
    <AnnexChangeReview
      review={reviewFixture({
        status: "approved",
        publication_generation: 1,
        review_payload: {
          ...reviewFixture().review_payload,
          raw_new_extraction: rawExtraction,
          corrections: [
            {
              position: 0,
              before_text: "New rate: 12%",
              corrected_text: "New rate: 11%",
              reason: "Verified correction.",
            },
          ],
          correction_reconciliation: {
            supported: true,
            rationale: "Correction matches the frozen cell.",
            input_sha256: "c".repeat(64),
          },
        },
      })}
      onUpdated={jest.fn()}
    />
  );

  expect(screen.getByText("Raw extraction 0: New rate: 12%")).toBeVisible();
  expect(screen.getByText(/Applied correction 0: New rate: 11%/)).toBeVisible();
  expect(screen.getByText(/Correction matches the frozen cell/)).toBeVisible();
  expect(
    screen.queryByRole("textbox", { name: "Correct NEW element 0" })
  ).not.toBeInTheDocument();
});

test("reports an unchanged blocked retry from the returned status", async () => {
  const user = setupUser();
  const infoSpy = jest.spyOn(toast, "info").mockImplementation(jest.fn());
  const blocked = reviewFixture({
    status: "blocked",
    review_payload: {
      ...reviewFixture().review_payload,
      publication: null,
      impact: null,
      issues: ["source_package_missing"],
    },
  });
  render(
    <AnnexChangeReview
      review={blocked}
      onUpdated={jest.fn()}
      onRetry={jest.fn().mockResolvedValue(blocked)}
    />
  );

  await user.click(screen.getByRole("button", { name: "Retry review" }));

  expect(infoSpy).toHaveBeenCalledWith(
    "Blocking checks remain; the review was not resumed."
  );
  infoSpy.mockRestore();
});

test("shows precise blocked and provider-reconciliation states", () => {
  const blocked = reviewFixture({
    status: "blocked",
    review_payload: {
      ...reviewFixture().review_payload,
      publication: null,
      impact: null,
      issues: ["source_package_missing"],
    },
  });
  const { rerender } = render(
    <AnnexChangeReview review={blocked} onUpdated={jest.fn()} />
  );
  expect(screen.getByText("Source package missing")).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Approve group" })
  ).not.toBeInTheDocument();

  rerender(
    <AnnexChangeReview
      review={reviewFixture({
        status: "failed",
        publication_generation: 1,
        error_message:
          "Publication needs manual provider reconciliation; no submission will be repeated.",
      })}
      onUpdated={jest.fn()}
    />
  );
  expect(screen.getByText(/manual provider reconciliation/i)).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Retry publication" })
  ).not.toBeInTheDocument();
});

test("shows durable preparation progress and prevents duplicate review actions", () => {
  const review = reviewFixture({
    status: "blocked",
    preparation: {
      status: "running",
      stage: "replacement_context",
      completed_chunks: 12,
      total_chunks: 469,
      error_message: null,
      result_review_id: null,
    },
  });
  render(<AnnexChangeReview review={review} onUpdated={jest.fn()} />);
  expect(screen.getByText("Preparing review")).toBeInTheDocument();
  expect(
    screen.getByText("replacement context · 12/469 chunks")
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Retry review" })).toBeDisabled();
  expect(
    screen.queryByRole("button", { name: "Approve group" })
  ).not.toBeInTheDocument();
});
