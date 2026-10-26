import { render, screen, setupUser, within } from "@tests/setup/test-utils";
import IntakeChunkReviewCard from "@/sections/cards/IntakeChunkReviewCard";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";

const chunk: RegulatoryChunk = {
  id: "chunk-1",
  user_file_id: "file-1",
  text: "Original rule.",
  position: 0,
  chunk_type: "madde",
  heading_path: ["MADDE 1"],
  chunk_metadata: { document_number: "4000" },
  validity_start_date: null,
  validity_end_date: null,
  status: "active",
  source: "indexed",
  supersedes_chunk_id: null,
  superseded_by_chunk_id: null,
  created_at: "",
  updated_at: "",
};

test("preserves the user's draft and original review receipt when another reviewer changes the chunk", async () => {
  const save = jest.fn(async () => {
    throw new Error("The review changed.");
  });
  const dirty = jest.fn();
  const user = setupUser();
  const view = render(
    <IntakeChunkReviewCard
      chunk={chunk}
      reviewSha256={"f".repeat(64)}
      editable
      disabled={false}
      onSave={save}
      onDirtyChange={dirty}
    >
      Preview
    </IntakeChunkReviewCard>
  );
  await user.type(screen.getByLabelText("After text"), " My edit.");
  await user.clear(screen.getByLabelText("After metadata document_number"));
  await user.type(
    screen.getByLabelText("After metadata document_number"),
    "5000"
  );
  view.rerender(
    <IntakeChunkReviewCard
      chunk={{
        ...chunk,
        text: "Other reviewer's rule.",
        chunk_metadata: { document_number: "6000" },
      }}
      reviewSha256={"a".repeat(64)}
      editable
      disabled={false}
      onSave={save}
      onDirtyChange={dirty}
    >
      Updated preview
    </IntakeChunkReviewCard>
  );
  expect(screen.getByLabelText("After text")).toHaveValue(
    "Original rule. My edit."
  );
  expect(screen.getByLabelText("After metadata document_number")).toHaveValue(
    "5000"
  );
  await user.click(screen.getByRole("button", { name: "Save chunk changes" }));
  expect(save).toHaveBeenCalledWith(
    expect.objectContaining({
      text: "Original rule. My edit.",
      chunk_metadata: { document_number: "5000" },
    }),
    "f".repeat(64)
  );
  expect(await screen.findByText("The review changed.")).toBeVisible();
  expect(dirty).toHaveBeenLastCalledWith("chunk-1", true);
  await user.click(
    screen.getByRole("button", { name: "Discard chunk changes" })
  );
  expect(screen.getByText("Updated preview")).toBeVisible();
  expect(dirty).toHaveBeenLastCalledWith("chunk-1", false);
});

test("shows per-field chunker and Gemini evidence while technical provenance stays read-only", async () => {
  const metadata = {
    document_number: "4000",
    document_date: "2026-10-26",
    article_title: "Gümrük hükümleri",
    source_links: { "0": "https://example.gov/main.pdf" },
    source_asset_ids: ["asset-1"],
    intake_legal_metadata: {
      version: 1,
      model: "gemini-3.5-flash",
      request_hash: "protected-request",
      input_sha256: "protected-input",
      input_stale: false,
      warnings: [],
      fields: {
        document_number: {
          source: "structural",
          basis: "chunker",
          evidence_quote: null,
          source_chunk_id: "chunk-1",
          stale: false,
        },
        document_date: {
          source: "llm",
          basis: "explicit",
          evidence_quote: "26 Ekim 2026 tarihinde kabul edilmiştir.",
          source_chunk_id: "chunk-1",
          stale: false,
        },
        validity_start_date: {
          source: "llm",
          basis: "relative_publication",
          evidence_quote: "Yayımını izleyen gün yürürlüğe girer.",
          source_chunk_id: "chunk-1",
          stale: false,
        },
      },
    },
  };
  const user = setupUser();
  render(
    <IntakeChunkReviewCard
      chunk={{
        ...chunk,
        validity_start_date: "2026-10-27",
        chunk_metadata: metadata,
      }}
      reviewSha256={"f".repeat(64)}
      editable
      disabled={false}
      onSave={jest.fn()}
      onDirtyChange={jest.fn()}
    >
      Full new legal text.
    </IntakeChunkReviewCard>
  );
  expect(
    within(
      screen.getByRole("group", { name: "Metadata document_number" })
    ).getByText("Chunker")
  ).toBeVisible();
  const date = screen.getByRole("group", { name: "Metadata document_date" });
  expect(within(date).getByText("Gemini · gemini-3.5-flash")).toBeVisible();
  expect(
    within(date).getByText("Evidence: 26 Ekim 2026 tarihinde kabul edilmiştir.")
  ).toBeVisible();
  expect(
    screen.getByText("Calculated relative to publication date.")
  ).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Source details" }));
  expect(screen.getByLabelText("Read-only chunk ID")).toHaveAttribute(
    "readonly"
  );
  const technical = screen.getByLabelText("Read-only technical metadata");
  expect(technical).toHaveAttribute("readonly");
  expect(JSON.parse((technical as HTMLTextAreaElement).value)).toEqual(
    expect.objectContaining({
      source_links: metadata.source_links,
      source_asset_ids: ["asset-1"],
      intake_legal_metadata: metadata.intake_legal_metadata,
    })
  );
  expect(
    screen.queryByLabelText("After metadata source_asset_ids")
  ).not.toBeInTheDocument();
  expect(
    screen.queryByLabelText("After metadata intake_legal_metadata")
  ).not.toBeInTheDocument();
});

test("saves only edited legal metadata keys, supports explicit clears and preserves structured strings", async () => {
  const save = jest.fn(async () => undefined);
  const dirty = jest.fn();
  const metadata = {
    document_number: "4000",
    article_title: "Old title",
    table_index: 1,
    source_asset_ids: ["protected"],
    labels: ["Generated label"],
  };
  const user = setupUser();
  render(
    <IntakeChunkReviewCard
      chunk={{ ...chunk, chunk_metadata: metadata }}
      reviewSha256={"f".repeat(64)}
      editable
      disabled={false}
      onSave={save}
      onDirtyChange={dirty}
    >
      Full text
    </IntakeChunkReviewCard>
  );
  await user.clear(screen.getByLabelText("After metadata document_number"));
  await user.type(
    screen.getByLabelText("After metadata document_number"),
    "5000"
  );
  await user.clear(screen.getByLabelText("After metadata article_title"));
  await user.clear(screen.getByLabelText("After metadata table_index"));
  await user.type(screen.getByLabelText("After metadata table_index"), "1.5");
  expect(
    screen.getByRole("button", { name: "Save chunk changes" })
  ).toBeDisabled();
  expect(
    screen.getAllByText("Table index must be a whole number.").length
  ).toBeGreaterThan(0);
  await user.clear(screen.getByLabelText("After metadata table_index"));
  await user.type(screen.getByLabelText("After metadata table_index"), "2");
  await user.click(screen.getByLabelText("After metadata annex_title"));
  await user.paste('{"title":"EK-1","rows":[1,2]}');
  expect(dirty).toHaveBeenLastCalledWith("chunk-1", true);
  await user.click(screen.getByRole("button", { name: "Save chunk changes" }));
  expect(save).toHaveBeenCalledWith(
    expect.objectContaining({
      chunk_metadata: {
        document_number: "5000",
        article_title: null,
        table_index: 2,
        annex_title: '{"title":"EK-1","rows":[1,2]}',
      },
    }),
    "f".repeat(64)
  );
  const payload = save.mock.calls[0] as unknown as [
    { chunk_metadata: Record<string, unknown> },
  ];
  expect(payload[0].chunk_metadata).not.toHaveProperty("source_asset_ids");
  expect(payload[0].chunk_metadata).not.toHaveProperty("labels");
});

test("shows the failed labeling reason on its chunk while keeping partial review read-only", () => {
  render(
    <IntakeChunkReviewCard
      chunk={{
        ...chunk,
        labeling_status: "failed",
        labeling_error:
          "The model abstained: no supported label was identified.",
      }}
      reviewSha256={null}
      editable={false}
      disabled={false}
      onSave={jest.fn()}
      onDirtyChange={jest.fn()}
    >
      Unresolved chunk preview
    </IntakeChunkReviewCard>
  );
  const alert = screen.getByRole("alert");
  expect(within(alert).getByText("Labeling: failed")).toBeVisible();
  expect(
    within(alert).getByText(
      "The model abstained: no supported label was identified."
    )
  ).toBeVisible();
  expect(screen.getByLabelText("After text")).toHaveAttribute("readonly");
  expect(
    screen.queryByRole("button", { name: "Save chunk changes" })
  ).not.toBeInTheDocument();
});
