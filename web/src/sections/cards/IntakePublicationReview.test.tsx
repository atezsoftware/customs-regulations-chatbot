import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import type { DocumentIntakeBatch } from "@/lib/regulatory/documentIntake";
import IntakePublicationReview from "@/sections/cards/IntakePublicationReview";

const batch: DocumentIntakeBatch = {
  id: "intake-1",
  source_package_id: "source-1",
  status: "ready",
  stage: "review",
  target_document_set_id: 7,
  target_document_set_name: "PC Külliyatı",
  review_sha256: "a".repeat(64),
  publication_review_sha256: "b".repeat(64),
  issues: [],
  documents: [
    {
      id: "html",
      asset_id: "html-asset",
      display_name: "Other rendition.html",
      role: "root",
      parent_document_id: null,
      source_url: null,
      status: "ready",
      chunk_count: 2,
      labels: {},
      error: null,
      user_file_id: "file-html",
    },
    {
      id: "pdf",
      asset_id: "pdf-asset",
      display_name: "Original.pdf",
      source_display_name: "uploaded-print.pdf",
      mime_type: "application/pdf",
      role: "root",
      parent_document_id: null,
      source_url: null,
      status: "ready",
      chunk_count: 2,
      labels: {},
      error: null,
      user_file_id: "file-pdf",
    },
    {
      id: "other-html",
      asset_id: "other-html-asset",
      display_name: "Original.pdf",
      source_display_name: "official-page.htm",
      mime_type: "text/html",
      role: "root",
      parent_document_id: null,
      source_url: null,
      status: "ready",
      chunk_count: 2,
      labels: {},
      error: null,
      user_file_id: "file-other-html",
    },
  ],
};

test("keeps the publication receipt frozen while a changed review rejects the user's identity draft", async () => {
  global.fetch = jest
    .fn()
    .mockResolvedValue({ ok: true, json: async () => [] });
  const save = jest
    .fn()
    .mockRejectedValue(
      new Error("The publication review changed. Reload before saving.")
    );
  const dirty = jest.fn();
  const props = {
    batch,
    document: batch.documents[0]!,
    disabled: false,
    onDirtyChange: dirty,
    onSave: save,
  };
  const user = setupUser();
  const view = render(<IntakePublicationReview {...props} />);
  await user.click(
    screen.getByRole("button", { name: "Review document identity" })
  );
  await user.click(
    screen.getByRole("button", {
      name: "Keep as source evidence of Original.pdf (application/pdf)",
    })
  );
  expect(
    screen.getByText("application/pdf · uploaded-print.pdf")
  ).toBeVisible();
  await user.type(
    screen.getByLabelText(
      "Document identity review reason for Other rendition.html"
    ),
    "Same document, alternate rendition."
  );
  view.rerender(
    <IntakePublicationReview
      {...props}
      batch={{ ...batch, publication_review_sha256: "c".repeat(64) }}
    />
  );
  await user.click(
    screen.getByRole("button", { name: "Save document identity review" })
  );
  expect(save).toHaveBeenCalledWith(
    {
      disposition: "source_evidence",
      duplicate_of_document_id: "pdf",
      existing_parent_file_id: null,
      reason: "Same document, alternate rendition.",
    },
    "b".repeat(64)
  );
  expect(
    await screen.findByText(
      "The publication review changed. Reload before saving."
    )
  ).toBeVisible();
  expect(
    screen.getByLabelText(
      "Document identity review reason for Other rendition.html"
    )
  ).toHaveValue("Same document, alternate rendition.");
  expect(dirty).toHaveBeenLastCalledWith("publication:html", true);
  await user.click(
    screen.getByRole("button", { name: "Discard document identity changes" })
  );
  await waitFor(() =>
    expect(dirty).toHaveBeenLastCalledWith("publication:html", false)
  );
  expect(
    screen.queryByLabelText(
      "Document identity review reason for Other rendition.html"
    )
  ).not.toBeInTheDocument();
});

test("prevents busy and approved reviews from changing publication identity", async () => {
  const save = jest.fn();
  const props = {
    batch,
    document: batch.documents[0]!,
    disabled: true,
    onDirtyChange: jest.fn(),
    onSave: save,
  };
  const view = render(<IntakePublicationReview {...props} />);
  expect(
    screen.getByRole("button", { name: "Review document identity" })
  ).toBeDisabled();
  view.rerender(
    <IntakePublicationReview
      {...props}
      disabled={false}
      batch={{ ...batch, status: "completed" }}
      document={{
        ...props.document,
        publication_disposition: "source_evidence",
        duplicate_of_document_id: "pdf",
        publication_review: {
          decision_id: "decision-1",
          reason: "Same source preserved.",
          reviewed_at: "2026-10-27T18:00:00Z",
        },
      }}
    />
  );
  expect(
    screen.getByText("Review reason: Same source preserved.")
  ).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Review document identity" })
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Save document identity review" })
  ).not.toBeInTheDocument();
  expect(save).not.toHaveBeenCalled();
});
