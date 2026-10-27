import { render, screen, setupUser, within } from "@tests/setup/test-utils";
import IntakeLegalMetadataFields from "@/sections/cards/IntakeLegalMetadataFields";

const contextWarning =
  "Document context changed after metadata extraction; model evidence is stale and inferred dates were cleared. Manually reviewed values are retained.";

function staleMetadata(warnings: string[] = [contextWarning, contextWarning]) {
  return {
    document_type: "teblig",
    article_no: "5",
    clause_label: "b",
    document_number: "SGM-2024/4",
    intake_legal_metadata: {
      input_stale: true,
      model: "gemini-3.5-flash",
      warnings,
      fields: {
        document_type: { source: "structural", basis: "chunker", stale: false },
        article_no: {
          source: "llm",
          basis: "explicit",
          stale: true,
          evidence_quote: "MADDE 5",
        },
        clause_label: { source: "llm", basis: "explicit", stale: false },
        document_number: {
          source: "manual",
          basis: "user_review",
          stale: false,
        },
      },
    },
  };
}

test("shows actual stale Gemini fields and one context warning without falsely marking other metadata", () => {
  const metadata = staleMetadata();
  const before = JSON.stringify(metadata);
  render(
    <IntakeLegalMetadataFields
      metadata={metadata}
      editing
      disabled={false}
      validityStart={null}
      validityEnd={null}
    />
  );
  const staleField = screen.getByRole("group", { name: "Metadata article_no" });
  expect(
    within(staleField).getByText("Source changed · review needed")
  ).toBeVisible();
  expect(within(staleField).getByText("Evidence: MADDE 5")).toBeVisible();
  for (const field of [
    "document_title",
    "document_type",
    "clause_label",
    "document_number",
  ])
    expect(
      within(
        screen.getByRole("group", { name: `Metadata ${field}` })
      ).queryByText("Source changed · review needed")
    ).not.toBeInTheDocument();
  expect(screen.getAllByText(contextWarning)).toHaveLength(1);
  expect(screen.getAllByText("Source changed · review needed")).toHaveLength(1);
  expect(JSON.stringify(metadata)).toBe(before);
});

test("retains document context provenance while a stale field has an unsaved manual replacement", async () => {
  const metadata = staleMetadata([]);
  const user = setupUser();
  render(
    <IntakeLegalMetadataFields
      metadata={metadata}
      editing
      disabled={false}
      validityStart={null}
      validityEnd={null}
    />
  );
  const field = screen.getByRole("group", { name: "Metadata article_no" });
  await user.clear(within(field).getByLabelText("After metadata article_no"));
  await user.type(
    within(field).getByLabelText("After metadata article_no"),
    "6"
  );
  expect(within(field).getByText("Manual · unsaved")).toBeVisible();
  expect(
    within(field).queryByText("Source changed · review needed")
  ).not.toBeInTheDocument();
  expect(
    screen.getAllByText(/Document context changed after metadata extraction/)
  ).toHaveLength(1);
  expect(metadata.intake_legal_metadata.input_stale).toBe(true);
  expect(metadata.intake_legal_metadata.fields.article_no.stale).toBe(true);
});
