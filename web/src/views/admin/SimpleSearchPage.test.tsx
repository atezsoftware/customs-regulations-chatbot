import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import SimpleSearchPage from "@/views/admin/SimpleSearchPage";
import type { LegislationFile } from "@/lib/legistations/interfaces";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";

jest.mock("next/navigation", () => ({
  useRouter: () => ({ back: jest.fn() }),
  usePathname: () => "/admin/simple-search",
}));

const file: LegislationFile = {
  id: "file-summary",
  name: "Kanun.md",
  folder_path: "",
  user_id: null,
  file_id: "stored-1",
  status: "COMPLETED",
  file_type: "text/markdown",
  created_at: "2026-11-01T00:00:00Z",
  chunk_count: 30,
  document_sets: [],
  pending_review: false,
};
const chunk: RegulatoryChunk = {
  id: "summary-chunk",
  user_file_id: file.id,
  text: "Bulunan kaynak parçası.",
  position: 80,
  chunk_type: "article",
  heading_path: ["Madde 80"],
  chunk_metadata: {},
  validity_start_date: null,
  validity_end_date: null,
  status: "active",
  source: "indexed",
  supersedes_chunk_id: null,
  superseded_by_chunk_id: null,
  created_at: file.created_at,
  updated_at: file.created_at,
};
let fetchSpy: jest.SpyInstance;
let summaryError: string | null;
function jsonResponse(body: unknown): Response {
  return { ok: true, json: async () => body } as Response;
}
beforeEach(() => {
  summaryError = null;
  fetchSpy = jest.spyOn(global, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), "http://localhost:3000");
    // POST /api/regulatory/catalog/simple-search: source summary and ranked chunks.
    if (url.pathname.endsWith("/simple-search"))
      return jsonResponse({
        search: {
          items: [{ file, chunk, score: 0.9 }],
          total: 1,
          offset: 0,
          limit: 20,
          has_more: false,
          total_is_exact: false,
          candidate_limit: 100,
          result_limit: 20,
          per_file_limit: 8,
        },
        summary: summaryError
          ? null
          : "Bu dosyada gümrük vergisi hakkında bir bölüm buldum. [1]",
        summary_model: "nvidia/nemotron-3-ultra-550b-a55b",
        summary_error: summaryError,
      });
    // GET catalog file: refresh selected file metadata.
    if (url.pathname === `/api/regulatory/catalog/files/${file.id}`)
      return jsonResponse(file);
    // GET catalog chunk location: locate exact match beyond first page.
    if (url.pathname.endsWith("/location")) return jsonResponse({ offset: 25 });
    // GET canonical chunk page: include matching and neighboring chunk.
    if (url.pathname.endsWith("/chunks/page"))
      return jsonResponse({
        items: [
          chunk,
          { ...chunk, id: "neighbor", position: 81, text: "Komşu chunk." },
        ],
        total: 30,
        offset: 25,
        limit: 25,
      });
    throw new Error(`Unexpected request: ${String(input)}`);
  });
});
afterEach(() => fetchSpy.mockRestore());

async function runSearch() {
  const user = setupUser();
  render(<SimpleSearchPage />);
  await user.type(
    screen.getByRole("textbox", { name: "Basit mevzuat araması" }),
    "gümrük vergisi"
  );
  await user.click(screen.getByRole("button", { name: "Ara" }));
  return user;
}

test("shows a source summary and numbered chunk cards with highlighted file navigation", async () => {
  const user = await runSearch();
  expect(await screen.findByText(/Bu dosyada gümrük vergisi/)).toBeVisible();
  expect(screen.getByText("Kaynak [1] · Chunk #80")).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Düzenle" })
  ).not.toBeInTheDocument();
  expect(fetchSpy).toHaveBeenCalledWith(
    "/api/regulatory/catalog/simple-search",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ query: "gümrük vergisi" }),
    })
  );
  await user.click(screen.getByRole("button", { name: "Dosyaya git" }));
  expect(
    await screen.findByLabelText("Arama eşleşmesi · Chunk #80")
  ).toHaveTextContent(chunk.text);
  expect(screen.getByText("Komşu chunk.")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Sonuçlara dön" }));
  expect(await screen.findByText(/Bu dosyada gümrük vergisi/)).toBeVisible();
  expect(
    fetchSpy.mock.calls.filter(([input]) =>
      String(input).endsWith("/simple-search")
    )
  ).toHaveLength(1);
});

test("keeps retrieved chunks accessible when the summary model fails", async () => {
  summaryError = "Özet modeli kullanılamıyor; kaynakları inceleyebilirsiniz.";
  await runSearch();
  expect(await screen.findByRole("alert")).toHaveTextContent(summaryError);
  expect(screen.getByText(chunk.text)).toBeVisible();
  expect(screen.getByRole("button", { name: "Dosyaya git" })).toBeEnabled();
});

test("renders the empty search and draft without fetching until submission", async () => {
  const user = setupUser();
  render(<SimpleSearchPage />);
  expect(screen.getByRole("button", { name: "Ara" })).toBeDisabled();
  expect(
    screen.getByText(/Arama yapın; ilgili kaynakların özeti/)
  ).toBeVisible();
  await user.type(
    screen.getByRole("textbox", { name: "Basit mevzuat araması" }),
    "gümrük"
  );
  expect(screen.getByRole("button", { name: "Ara" })).toBeEnabled();
  await user.clear(
    screen.getByRole("textbox", { name: "Basit mevzuat araması" })
  );
  expect(screen.getByRole("button", { name: "Ara" })).toBeDisabled();
  await waitFor(() => expect(fetchSpy).not.toHaveBeenCalled());
});
