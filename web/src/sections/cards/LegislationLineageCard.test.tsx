import { render, screen, setupUser } from "@tests/setup/test-utils";
import LegislationLineageCard from "@/sections/cards/LegislationLineageCard";

test("shows the successor and explicit repeal on the old legislation and opens the related file", async () => {
  global.fetch = jest.fn(
    async () =>
      ({
        ok: true,
        json: async () => ({
          user_file_id: "old-law",
          title: "4000 sayılı Gümrük Kanunu",
          status: "repealed",
          effective_date: "2026-10-26",
          predecessors: [],
          successors: [
            {
              user_file_id: "new-law",
              title: "5000 sayılı Gümrük Kanunu",
              relation: "repeals",
              effective_date: "2026-10-26",
              evidence: "4000 sayılı Kanun yürürlükten kaldırılmıştır.",
            },
          ],
        }),
      }) as Response
  );
  const openRelated = jest.fn();
  const user = setupUser();
  render(
    <LegislationLineageCard fileId="old-law" onOpenRelatedFile={openRelated} />
  );
  expect(await screen.findByText("Repealed")).toBeVisible();
  expect(
    screen.getByText("Successor: 5000 sayılı Gümrük Kanunu")
  ).toBeVisible();
  expect(
    screen.getByText("4000 sayılı Kanun yürürlükten kaldırılmıştır.")
  ).toBeVisible();
  await user.click(
    screen.getByRole("button", { name: "Open 5000 sayılı Gümrük Kanunu" })
  );
  expect(openRelated).toHaveBeenCalledWith("new-law");
  expect(global.fetch).toHaveBeenCalledWith(
    "/api/regulatory/document-intake/files/old-law/lineage",
    undefined
  );
});
