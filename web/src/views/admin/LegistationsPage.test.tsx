import {
  render,
  screen,
  setupUser,
  waitFor,
  within,
} from "@tests/setup/test-utils";
import LegistationsPage from "@/views/admin/LegistationsPage";
import type { LegislationFile } from "@/lib/legistations/interfaces";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";

jest.mock("next/navigation", () => ({
  useRouter: () => ({ back: jest.fn() }),
  usePathname: () => "/admin/legistations",
}));

const file: LegislationFile = {
  id: "file-1",
  name: "Mevzuat/Kanun.md",
  folder_path: "Mevzuat",
  user_id: null,
  file_id: "stored-1",
  status: "COMPLETED",
  file_type: "text/markdown",
  created_at: "2026-11-01T00:00:00Z",
  chunk_count: 2,
  document_sets: [{ id: 7, name: "PC Külliyatı" }],
  pending_review: false,
};

const chunk: RegulatoryChunk = {
  id: "chunk-1",
  user_file_id: file.id,
  text: "Gümrük vergisinin hesaplanması.",
  position: 1,
  chunk_type: "article",
  heading_path: ["Madde 1"],
  chunk_metadata: { article: "1" },
  validity_start_date: null,
  validity_end_date: null,
  status: "active",
  source: "indexed",
  supersedes_chunk_id: null,
  superseded_by_chunk_id: null,
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
};

let fetchSpy: jest.SpyInstance;
let latestSearchBody: Record<string, unknown> | null;
let pendingReview: boolean;
let aggregate: boolean;
let patchFailure: number | null;
let fileStatus: string;
let indexQueued: boolean;

function jsonResponse(body: unknown): Response {
  return { ok: true, json: async () => body } as Response;
}

beforeEach(() => {
  latestSearchBody = null;
  pendingReview = false;
  aggregate = false;
  patchFailure = null;
  fileStatus = "COMPLETED";
  indexQueued = false;
  fetchSpy = jest
    .spyOn(global, "fetch")
    .mockImplementation(async (input, init) => {
      const url = new URL(String(input), "http://localhost:3000");
      // GET /api/regulatory/catalog/folders: global immediate child folders.
      if (url.pathname === "/api/regulatory/catalog/folders") {
        return jsonResponse({
          items: url.searchParams.get("parent")
            ? []
            : [{ name: "Mevzuat", path: "Mevzuat", file_count: 30 }],
          total: url.searchParams.get("parent") ? 0 : 1,
          offset: 0,
          limit: 50,
        });
      }
      // GET /api/regulatory/catalog/files: paginated catalog.
      if (url.pathname === "/api/regulatory/catalog/files") {
        const offset = Number(url.searchParams.get("offset"));
        return jsonResponse({
          items: [
            {
              ...file,
              name: offset ? "Mevzuat/İkinci.md" : file.name,
              pending_review: pendingReview,
              status: fileStatus,
            },
          ],
          total: url.searchParams.has("folder") ? 1 : 26,
          offset,
          limit: 25,
        });
      }
      // GET catalog file and POST initial indexing share the refreshed file snapshot.
      if (
        url.pathname === `/api/regulatory/catalog/files/${file.id}` ||
        url.pathname === `/api/regulatory/catalog/files/${file.id}/index`
      ) {
        if (init?.method === "POST") indexQueued = true;
        return jsonResponse({
          ...file,
          pending_review: pendingReview,
          status: fileStatus,
          index_request: indexQueued
            ? {
                attempt_id: "index-attempt-1",
                status: "PENDING",
                updated_at: chunk.updated_at,
                failure_code: null,
              }
            : null,
        });
      }
      // POST /api/regulatory/catalog/search: raw retrieval, no chat route.
      if (url.pathname === "/api/regulatory/catalog/search") {
        latestSearchBody = JSON.parse(String(init?.body));
        return jsonResponse({
          items: [
            {
              file: { ...file, pending_review: pendingReview },
              chunk: aggregate
                ? { ...chunk, chunk_type: "hierarchical_aggregate" }
                : chunk,
              score: 0.8,
            },
          ],
          total: 1,
          offset: 0,
          limit: 25,
          has_more: false,
          total_is_exact: false,
        });
      }
      // GET /api/regulatory/files/:id/chunks/page: all chunks of the selected file.
      if (url.pathname === `/api/regulatory/files/${file.id}/chunks/page`) {
        return jsonResponse({
          items: [
            aggregate
              ? { ...chunk, chunk_type: "hierarchical_aggregate" }
              : chunk,
            {
              ...chunk,
              id: "chunk-2",
              position: 2,
              text: "Dosyanın diğer chunkı.",
            },
          ],
          total: 2,
          offset: 0,
          limit: 25,
        });
      }
      // PATCH /api/regulatory/chunks/:id: individual chunk edit and reindex.
      if (
        url.pathname === `/api/regulatory/chunks/${chunk.id}` &&
        init?.method === "PATCH"
      ) {
        if (patchFailure !== null)
          return {
            ok: false,
            status: patchFailure,
            json: async () => ({ detail: "Chunk changed." }),
          } as Response;
        return jsonResponse({ ...chunk, ...JSON.parse(String(init.body)) });
      }
      throw new Error(`Unexpected request: ${String(input)}`);
    });
});

afterEach(() => fetchSpy.mockRestore());

test("paginates files and selects a global folder independent of the current page", async () => {
  const user = setupUser();
  render(<LegistationsPage />);
  expect(await screen.findByRole("heading", { name: file.name })).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Sonraki" }));
  expect(
    await screen.findByRole("heading", { name: "Mevzuat/İkinci.md" })
  ).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Mevzuat (30)" }));
  expect(await screen.findByRole("heading", { name: file.name })).toBeVisible();
  await waitFor(() =>
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining("folder=Mevzuat")
    )
  );
  expect(screen.getByRole("button", { name: "Sonraki" })).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Mevzuat klasörünü daralt" })
  ).toHaveAttribute("aria-expanded", "true");
});

test("opens all file chunks from an embedding result and returns to the same search", async () => {
  const user = setupUser();
  render(<LegistationsPage />);
  await user.click(screen.getByRole("button", { name: "Chunk · Embedding" }));
  await user.type(
    screen.getByRole("textbox", { name: "Mevzuat araması" }),
    "gümrük vergisi"
  );
  await user.click(screen.getByRole("button", { name: "Ara" }));
  await user.click(await screen.findByRole("button", { name: "Dosyaya git" }));
  expect(await screen.findByText("Dosyanın diğer chunkı.")).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Sonuçlara dön" }));
  expect(
    await screen.findByRole("button", { name: "Dosyaya git" })
  ).toBeVisible();
  expect(latestSearchBody).toEqual({
    query: "gümrük vergisi",
    mode: "semantic",
    offset: 0,
    limit: 25,
  });
  expect(screen.getByRole("textbox", { name: "Mevzuat araması" })).toHaveValue(
    "gümrük vergisi"
  );
  expect(
    fetchSpy.mock.calls.every(([url]) => !String(url).includes("/chat"))
  ).toBe(true);
});

test("edits a search chunk with its revision and reindexes only that chunk", async () => {
  const user = setupUser();
  render(<LegistationsPage />);
  await user.click(screen.getByRole("button", { name: "Chunk · BM25" }));
  await user.type(
    screen.getByRole("textbox", { name: "Mevzuat araması" }),
    "gümrük"
  );
  await user.click(screen.getByRole("button", { name: "Ara" }));
  await user.click(await screen.findByRole("button", { name: "Düzenle" }));
  const dialog = await screen.findByRole("dialog");
  const textInput = within(dialog).getAllByRole("textbox")[0]!;
  await user.clear(textInput);
  await user.type(textInput, "Düzeltilen chunk metni.");
  await user.click(
    within(dialog).getByRole("button", { name: "Save & Re-index" })
  );
  await waitFor(() =>
    expect(fetchSpy).toHaveBeenCalledWith(
      `/api/regulatory/chunks/${chunk.id}`,
      expect.objectContaining({ method: "PATCH", body: expect.any(String) })
    )
  );
  const patchCall = fetchSpy.mock.calls.find(
    ([, init]) => init?.method === "PATCH"
  );
  expect(JSON.parse(patchCall![1].body)).toEqual(
    expect.objectContaining({
      expected_updated_at: chunk.updated_at,
      text: "Düzeltilen chunk metni.",
    })
  );
  expect(
    fetchSpy.mock.calls.filter(([, init]) => init?.method === "PATCH")
  ).toHaveLength(1);
});

test("keeps pending intake chunks visible and directs edits to their review workflow", async () => {
  pendingReview = true;
  const user = setupUser();
  render(<LegistationsPage />);
  await user.click(await screen.findByRole("button", { name: "Dosyayı aç" }));
  expect(await screen.findByText("Dosyanın diğer chunkı.")).toBeVisible();
  expect(screen.getByText(/Updates → New document/)).toBeVisible();
  expect(screen.getByRole("button", { name: "Edit chunk 1" })).toBeDisabled();
});

test("separates root files from all files when selecting folders", async () => {
  const user = setupUser();
  render(<LegistationsPage />);
  await screen.findByRole("heading", { name: file.name });
  await user.click(screen.getByRole("button", { name: "Kökteki dosyalar" }));
  await waitFor(() =>
    expect(
      fetchSpy.mock.calls.some(([input]) => {
        const url = new URL(String(input), "http://localhost:3000");
        return (
          url.pathname.endsWith("/catalog/files") &&
          url.searchParams.get("folder") === "" &&
          url.searchParams.get("recursive") === "false"
        );
      })
    ).toBe(true)
  );
  await user.click(screen.getByRole("button", { name: "Tüm dosyalar" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Sonraki" })).toBeEnabled()
  );
  await waitFor(() => {
    const requests = fetchSpy.mock.calls.filter(([input]) =>
      String(input).startsWith("/api/regulatory/catalog/files")
    );
    expect(
      new URL(requests.at(-1)![0], "http://localhost:3000").searchParams.has(
        "folder"
      )
    ).toBe(false);
  });
});

test("preserves the editor draft and search view after a concurrent edit conflict", async () => {
  patchFailure = 409;
  const user = setupUser();
  render(<LegistationsPage />);
  await user.click(screen.getByRole("button", { name: "Chunk · BM25" }));
  await user.type(
    screen.getByRole("textbox", { name: "Mevzuat araması" }),
    "gümrük"
  );
  await user.click(screen.getByRole("button", { name: "Ara" }));
  await user.click(await screen.findByRole("button", { name: "Düzenle" }));
  const dialog = await screen.findByRole("dialog");
  const textInput = within(dialog).getByRole("textbox", {
    name: "Chunk metni",
  });
  await user.clear(textInput);
  await user.type(textInput, "Kaybolmaması gereken taslak.");
  await user.click(
    within(dialog).getByRole("button", { name: "Save & Re-index" })
  );
  expect(await within(dialog).findByText(/Chunk değişmiş/)).toBeVisible();
  expect(textInput).toHaveValue("Kaybolmaması gereken taslak.");
  expect(
    within(dialog).getByRole("button", { name: "Save & Re-index" })
  ).toBeEnabled();
  expect(screen.queryByText("Sonuçlara dön")).not.toBeInTheDocument();
  expect(
    fetchSpy.mock.calls.filter(([, init]) => init?.method === "PATCH")
  ).toHaveLength(1);
});

test("shows derived chunks in search and file views while directing edits to their source chunks", async () => {
  aggregate = true;
  const user = setupUser();
  render(<LegistationsPage />);
  await user.click(screen.getByRole("button", { name: "Chunk · BM25" }));
  await user.type(
    screen.getByRole("textbox", { name: "Mevzuat araması" }),
    "gümrük"
  );
  await user.click(screen.getByRole("button", { name: "Ara" }));
  expect(await screen.findByRole("button", { name: "Düzenle" })).toBeDisabled();
  expect(screen.getByText(/Bu chunk alt chunklardan türetilir/)).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Dosyaya git" }));
  expect(
    await screen.findByRole("button", { name: "Edit chunk 1" })
  ).toBeDisabled();
  expect(screen.getByRole("button", { name: "Edit chunk 2" })).toBeEnabled();
});

test("saves unindexed chunk edits before explicit initial indexing and refreshes its queued status", async () => {
  fileStatus = "CHUNKED";
  const user = setupUser();
  render(<LegistationsPage />);
  await user.click(await screen.findByRole("button", { name: "Dosyayı aç" }));
  expect(
    await screen.findByText(/henüz arama indeksine yayımlanmadı/)
  ).toBeVisible();
  await user.click(await screen.findByRole("button", { name: "Edit chunk 1" }));
  const dialog = await screen.findByRole("dialog");
  const textInput = within(dialog).getByRole("textbox", {
    name: "Chunk metni",
  });
  await user.clear(textInput);
  await user.type(textInput, "İndekslemeden önce düzeltilen içerik.");
  expect(
    within(dialog).queryByRole("button", { name: "Save & Re-index" })
  ).not.toBeInTheDocument();
  await user.click(within(dialog).getByRole("button", { name: "Kaydet" }));
  await waitFor(() =>
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  );
  expect(
    fetchSpy.mock.calls.some(([url]) => String(url).endsWith("/index"))
  ).toBe(false);
  await user.click(screen.getByRole("button", { name: "İndeksle" }));
  expect(
    await screen.findByRole("button", { name: "İndeksleniyor…" })
  ).toBeDisabled();
  expect(screen.getByRole("button", { name: "Edit chunk 1" })).toBeDisabled();
  expect(fetchSpy).toHaveBeenCalledWith(
    `/api/regulatory/catalog/files/${file.id}/index`,
    { method: "POST" }
  );
  fileStatus = "COMPLETED";
  indexQueued = false;
  await user.click(screen.getByRole("button", { name: "Durumu yenile" }));
  expect(await screen.findByText("Dosya durumu: COMPLETED")).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "İndeksleniyor…" })
  ).not.toBeInTheDocument();
  expect(
    await screen.findByRole("button", { name: "Edit chunk 1" })
  ).toBeEnabled();
});
