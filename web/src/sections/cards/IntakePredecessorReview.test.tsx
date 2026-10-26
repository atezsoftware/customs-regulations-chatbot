import { render, screen, setupUser } from "@tests/setup/test-utils";
import IntakePredecessorReview from "@/sections/cards/IntakePredecessorReview";
import type {
  DocumentIntakeBatch,
  IntakeDocument,
} from "@/lib/regulatory/documentIntake";

const document: IntakeDocument = {
  id: "new",
  asset_id: "asset",
  display_name: "5000 sayılı Gümrük Kanunu",
  role: "root",
  parent_document_id: null,
  source_url: null,
  status: "ready",
  chunk_count: 1,
  labels: {},
  error: null,
  user_file_id: "new-file",
};
const batch: DocumentIntakeBatch = {
  id: "batch",
  source_package_id: "source",
  status: "ready",
  stage: "review",
  target_document_set_id: 7,
  target_document_set_name: "PC Külliyatı",
  review_sha256: "f".repeat(64),
  issues: [],
  documents: [document],
};

test("does not replace a predecessor draft's original review receipt after a background refresh", async () => {
  global.fetch = jest.fn(
    async () =>
      ({
        ok: true,
        json: async () => [
          {
            user_file_id: "old-file",
            title: "4000 sayılı Gümrük Kanunu",
            reason: "Similar title",
            score: 0.9,
            status: "current",
          },
        ],
      }) as Response
  );
  const save = jest.fn(async () => {
    throw new Error("The document review changed.");
  });
  const dirty = jest.fn();
  const user = setupUser();
  const view = render(
    <IntakePredecessorReview
      batch={batch}
      document={document}
      disabled={false}
      onSave={save}
      onDirtyChange={dirty}
    />
  );
  await user.click(
    screen.getByRole("button", { name: "Review previous legislation" })
  );
  await user.click(
    await screen.findByRole("button", {
      name: "Select 4000 sayılı Gümrük Kanunu",
    })
  );
  view.rerender(
    <IntakePredecessorReview
      batch={{ ...batch, review_sha256: "a".repeat(64) }}
      document={document}
      disabled={false}
      onSave={save}
      onDirtyChange={dirty}
    />
  );
  expect(
    screen.getByText("Selected predecessor: 4000 sayılı Gümrük Kanunu")
  ).toBeVisible();
  await user.click(
    screen.getByRole("button", {
      name: "Save previous legislation relationship",
    })
  );
  expect(save).toHaveBeenLastCalledWith(
    expect.objectContaining({
      predecessor_user_file_id: "old-file",
      relation: "successor",
    }),
    "f".repeat(64)
  );
  expect(await screen.findByText("The document review changed.")).toBeVisible();
  await user.click(
    screen.getByRole("button", { name: "Discard relationship changes" })
  );
  await user.click(
    screen.getByRole("button", { name: "Select 4000 sayılı Gümrük Kanunu" })
  );
  await user.click(
    screen.getByRole("button", {
      name: "Save previous legislation relationship",
    })
  );
  expect(save).toHaveBeenLastCalledWith(
    expect.objectContaining({
      predecessor_user_file_id: "old-file",
      relation: "successor",
    }),
    "a".repeat(64)
  );
});
