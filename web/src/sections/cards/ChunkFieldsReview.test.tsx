import { render, screen, within } from "@tests/setup/test-utils";
import { InputTypeIn } from "@opal/components";
import ChunkFieldsReview, {
  ReadOnlyFieldValue,
} from "@/sections/cards/ChunkFieldsReview";

const fields = {
  id: "chunk-1",
  user_file_id: "file-1",
  position: 0,
  text: "Legal source text.",
  chunk_type: "article",
  heading_path: ["MADDE 1"],
  metadata: { article_no: "1" },
  validity_start_date: null,
  validity_end_date: null,
  status: "active",
  source: "indexed",
  supersedes_chunk_id: null,
  superseded_by_chunk_id: null,
  created_at: "2026-10-26",
  updated_at: "2026-10-26",
};

test("uses the Update Before/After row layout for a new chunk without loading an old comparison", () => {
  render(
    <ChunkFieldsReview
      before={null}
      after={fields}
      renderAfterValue={(key, value) =>
        key === "text" ? (
          <InputTypeIn
            aria-label="After text"
            value={String(value)}
            onChange={() => undefined}
          />
        ) : (
          <ReadOnlyFieldValue value={value} />
        )
      }
    />
  );
  expect(screen.getByText("(new — no existing chunk)")).toBeVisible();
  expect(
    screen.queryByRole("table", { name: "Before chunk fields" })
  ).not.toBeInTheDocument();
  const after = screen.getByRole("table", { name: "After chunk fields" });
  expect(
    within(after)
      .getAllByRole("rowheader")
      .map((header) => header.textContent)
  ).toEqual([
    "id",
    "user_file_id",
    "position",
    "text",
    "chunk_type",
    "heading_path",
    "metadata",
    "validity_start_date",
    "validity_end_date",
    "status",
    "source",
    "created_at",
    "updated_at",
  ]);
  expect(within(after).getByLabelText("After text")).toBeEnabled();
  expect(after.parentElement?.parentElement).toHaveClass(
    "grid",
    "grid-cols-1",
    "gap-3",
    "lg:grid-cols-2"
  );
});

test("keeps normal Update before fields read-only and delegates After controls unchanged", () => {
  render(
    <ChunkFieldsReview
      before={fields}
      after={{ ...fields, text: "Reviewed replacement." }}
      renderAfterValue={(key, value) =>
        key === "text" ? (
          <InputTypeIn
            aria-label="After text"
            value={String(value)}
            onChange={() => undefined}
          />
        ) : (
          <ReadOnlyFieldValue value={value} />
        )
      }
    />
  );
  const before = screen.getByRole("table", { name: "Before chunk fields" });
  expect(within(before).getByText("Legal source text.")).toBeVisible();
  expect(within(before).queryByRole("textbox")).not.toBeInTheDocument();
  expect(
    within(
      screen.getByRole("table", { name: "After chunk fields" })
    ).getByLabelText("After text")
  ).toHaveValue("Reviewed replacement.");
});
