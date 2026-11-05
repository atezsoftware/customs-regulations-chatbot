import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import type { LegislationFile } from "@/lib/legistations/interfaces";
import LegislationSourceIdentity from "@/sections/legistations/LegislationSourceIdentity";

const original = {
  asset_id: "asset",
  sha256: "a".repeat(64),
  original_name: "annex.pdf",
  original_url: null,
  final_url: null,
};
const file: LegislationFile = {
  id: "annex",
  name: "Başlık belirlenemedi",
  folder_path: "",
  user_id: "owner",
  file_id: "raw",
  status: "COMPLETED",
  file_type: "application/pdf",
  created_at: "2026-11-05T00:00:00Z",
  chunk_count: 1,
  document_sets: [{ id: 15, name: "PC Külliyatı" }],
  pending_review: false,
  source_lineage: {
    version: 1,
    intake_id: "intake",
    package_id: "package",
    document_set_id: 15,
    document_id: "annex-member",
    role: "annex",
    source: original,
    parents: [
      {
        user_file_id: "parent",
        document_id: "parent-member",
        title: "Parent title",
        source: {
          ...original,
          asset_id: "parent-source",
          original_name: "parent.pdf",
        },
      },
    ],
    attachments: [],
  },
};

afterEach(() => jest.restoreAllMocks());

test("keeps long legal titles in wrapping text with a short navigation action", () => {
  const title = "Çevreye Duyarlı Tasarım Gereklilikleri Tebliği "
    .repeat(8)
    .trim();
  const longTitleFile = {
    ...file,
    source_lineage: {
      ...file.source_lineage!,
      parents: [{ ...file.source_lineage!.parents[0]!, title }],
    },
  };
  render(
    <LegislationSourceIdentity file={longTitleFile} onSelectFile={jest.fn()} />
  );
  expect(
    screen.getByRole("button", { name: `Ana dosya: ${title}` })
  ).toHaveTextContent("Ana dosyayı aç");
  expect(screen.getByText(`Ana dosya: ${title}`)).toBeVisible();
});

test("opens the parent returned by the catalog API", async () => {
  const parent = {
    ...file,
    id: "parent",
    name: "Parent title",
    source_lineage: null,
  };
  const fetcher = jest
    .spyOn(global, "fetch")
    .mockResolvedValue({ ok: true, json: async () => parent } as Response);
  const select = jest.fn();
  render(<LegislationSourceIdentity file={file} onSelectFile={select} />);
  await setupUser().click(
    screen.getByRole("button", { name: "Ana dosya: Parent title" })
  );
  await waitFor(() => expect(select).toHaveBeenCalledWith(parent));
  expect(fetcher).toHaveBeenCalledWith("/api/regulatory/catalog/files/parent");
});

test("keeps a denied parent navigation visible without switching files", async () => {
  jest.spyOn(global, "fetch").mockResolvedValue({
    ok: false,
    status: 403,
    json: async () => ({ detail: "Parent access denied" }),
  } as Response);
  const select = jest.fn();
  render(<LegislationSourceIdentity file={file} onSelectFile={select} />);
  await setupUser().click(
    screen.getByRole("button", { name: "Ana dosya: Parent title" })
  );
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Parent access denied"
  );
  expect(select).not.toHaveBeenCalled();
});
