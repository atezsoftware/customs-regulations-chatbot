import { SWRConfig } from "swr";
import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import AnnexChunkReview from "@/views/admin/AnnexChunkReview";
import ChunkChangeCard, {
  ChunkContent,
} from "@/sections/cards/ChunkChangeCard";
import {
  type AnnexReview,
  type AnnexChunkReviewPage,
  getAnnexChunkPage,
  getAnnexReview,
  prepareAnnexSelection,
} from "@/lib/regulatory/amendments";

jest.mock("@/lib/regulatory/amendments", () => ({
  ...jest.requireActual("@/lib/regulatory/amendments"),
  getAnnexChunkPage: jest.fn(),
  getAnnexReview: jest.fn(),
  prepareAnnexSelection: jest.fn(),
}));

const review: AnnexReview = {
  id: "parent",
  logical_group_id: "group",
  batch_id: 78,
  review_revision: 2,
  status: "pending",
  review_sha256: "a".repeat(64),
  publication_generation: 0,
  created_at: "2026-09-15",
  error_message: null,
  review_payload: {
    instruction_indices: [0],
    instruction_texts: [],
    annex_label: "ek:3",
    effective_date: "2026-09-15",
    source_package_id: null,
    source_text_sha256: null,
    source_manifest_sha256: null,
    original_source_text_sha256: null,
    old_evidence_kind: "canonical_text",
    submitted_source_text: null,
    raw_new_extraction: null,
    old_extraction: null,
    new_extraction: null,
    corrections: [],
    correction_reconciliation: null,
    comparison: null,
    patch_plan: null,
    items: [],
    impact: null,
    evidence: [],
    source_only_canonical_ids: [],
    after_window_authority: null,
    publication: null,
    issues: [],
  },
};

function page(offset: number): AnnexChunkReviewPage {
  return {
    total: 11,
    offset,
    limit: 10,
    items: [
      {
        id: `item-${offset}`,
        position: offset,
        operation: "replace",
        old_chunks: [],
        new_chunks: [],
        selection: null,
        old_image_evidence_ids: [],
        new_image_evidence_ids: [],
      },
    ],
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  jest
    .mocked(getAnnexChunkPage)
    .mockImplementation(async (_review, offset) => page(offset));
  jest.mocked(prepareAnnexSelection).mockResolvedValue(review);
});

function show() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <AnnexChunkReview review={review} onUpdated={jest.fn()} />
    </SWRConfig>
  );
}

test("loads chunk pages without fetching the full annex evidence and preserves explicit selection across pages", async () => {
  const user = setupUser();
  show();
  await screen.findByText("1. replace · EK-3");
  expect(getAnnexReview).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: /^Select$/ }));
  await user.click(screen.getByRole("button", { name: "Next" }));
  await screen.findByText("11. replace · EK-3");
  await user.click(screen.getByRole("button", { name: /^Select$/ }));
  await user.click(
    screen.getByRole("button", { name: "Prepare 2 selected changes" })
  );
  await waitFor(() =>
    expect(prepareAnnexSelection).toHaveBeenCalledWith(review, [
      "item-0",
      "item-10",
    ])
  );
  expect(getAnnexReview).not.toHaveBeenCalled();
});

test("renders merged table cells while dropping executable and external HTML", async () => {
  const { container } = render(
    <ChunkContent
      text={
        '<table><tr><td rowspan="2" onclick="alert(1)">Certificate</td><td>New value</td></tr><tr><td>Second</td></tr></table><script>alert(1)</script><img src="https://untrusted.invalid/tracker" />'
      }
    />
  );
  await screen.findByRole("table");
  expect(screen.getByText("Certificate")).toHaveAttribute("rowspan", "2");
  expect(container.querySelector("script, img, [onclick]")).toBeNull();
});

test("highlights the changed text without exposing source markup", () => {
  const { container } = render(
    <ChunkChangeCard
      title="Changed paragraph"
      before={["Old rate: 10 percent"]}
      after={["Old rate: 20 percent"]}
    />
  );
  expect(
    Array.from(container.querySelectorAll("mark")).map(
      (node) => node.textContent
    )
  ).toEqual(["1", "2"]);
  expect(screen.getByText("Before")).toBeInTheDocument();
  expect(screen.getByText("After")).toBeInTheDocument();
});
