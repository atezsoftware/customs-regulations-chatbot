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

test("opens published parents and attachments as navigation without treating them as succession or repeal", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      user_file_id: "current-file",
      title: "Existing regulation",
      status: "current",
      effective_date: null,
      predecessors: [],
      successors: [],
      existing_parents: [
        {
          user_file_id: "parent-file",
          display_name: "Parent regulation",
          source_url: "https://example.gov/20220618-8.htm",
        },
      ],
      attachments: [
        {
          user_file_id: "child-file",
          display_name: "Published EK-1",
          source_url: "https://example.gov/20220618-8-1.pdf",
        },
      ],
    }),
  });
  const open = jest.fn();
  const user = setupUser();
  render(
    <LegislationLineageCard fileId="current-file" onOpenRelatedFile={open} />
  );
  expect(await screen.findByText("Current")).toBeVisible();
  expect(screen.getByText("Existing parent: Parent regulation")).toBeVisible();
  expect(screen.getByText("Attachment: Published EK-1")).toBeVisible();
  expect(screen.getByText("https://example.gov/20220618-8.htm")).toBeVisible();
  expect(screen.queryByText("Repealed")).not.toBeInTheDocument();
  expect(screen.queryByText(/^Successor:/)).not.toBeInTheDocument();
  await user.click(
    screen.getByRole("button", { name: "Open Parent regulation" })
  );
  await user.click(screen.getByRole("button", { name: "Open Published EK-1" }));
  expect(open.mock.calls).toEqual([["parent-file"], ["child-file"]]);
  expect(global.fetch).toHaveBeenCalledTimes(1);
  expect(global.fetch).toHaveBeenCalledWith(
    "/api/regulatory/document-intake/files/current-file/lineage",
    undefined
  );
});
