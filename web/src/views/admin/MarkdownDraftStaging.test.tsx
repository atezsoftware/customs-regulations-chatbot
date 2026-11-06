import { StrictMode } from "react";
import {
  act,
  deferred,
  render,
  screen,
  setupUser,
  waitFor,
  within,
} from "@tests/setup/test-utils";
import NewDocumentIntake from "@/views/admin/NewDocumentIntake";

const base = "/api/regulatory/document-intake";
const draftsBase = `${base}/markdown-drafts`;
interface TestDocument {
  id: string;
  display_name: string;
  parent_document_id: string | null;
  byte_count: number;
  sha256: string;
}
interface TestDraft {
  id: string;
  revision: number;
  status: "draft" | "prepared";
  documents: TestDocument[];
  source_package_id: string | null;
  updated_at: string;
}
const root: TestDocument = {
  id: "root-1",
  display_name: "ana.md",
  parent_document_id: null,
  byte_count: 8,
  sha256: "a".repeat(64),
};
function draft(documents: TestDocument[] = []): TestDraft {
  return {
    id: "draft-1",
    revision: documents.length,
    status: "draft",
    documents,
    source_package_id: null,
    updated_at: "2026-11-06T10:00:00Z",
  };
}
function json(value: unknown): Response {
  return { ok: true, status: 200, json: async () => value } as Response;
}

// Only HTTP is mocked; production components and request builders run.
function server(initial?: TestDraft, configured = true) {
  let current = initial ?? draft();
  let exists = Boolean(initial);
  const uploadedRequests = new Map<string, TestDraft>();
  const packageSnapshot = {
    id: "source-staged",
    document_set_id: 7,
    status: "ready",
    assets: [],
    issues: [],
    asset_count: 2,
    total_bytes: 16,
    manifest_sha256: "b".repeat(64),
    created_at: "2026-11-06",
    updated_at: "2026-11-06",
  };
  const batch = {
    id: "intake-staged",
    source_package_id: packageSnapshot.id,
    status: "ready",
    stage: "review",
    target_document_set_id: 7,
    target_document_set_name: "PC Külliyatı",
    review_sha256: "c".repeat(64),
    documents: [],
    issues: [],
  };
  const fetchMock = jest.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === `${base}/capabilities`)
        return json({
          enabled: true,
          target_document_set_id: 7,
          target_document_set_name: "PC Külliyatı",
          error: null,
        });
      if (url === `${base}/labeling-setup`)
        return json({
          providers: configured ? [{ id: 1 }] : [],
          configuration_errors: [],
          default_label_count: 255,
        });
      if (url === `${base}/batches` && init?.method === "POST")
        return json(batch);
      if (url === `${base}/batches`) return json([]);
      if (url === draftsBase && init?.method === "POST") {
        exists = true;
        return json(current);
      }
      if (url === draftsBase) return json(exists ? [current] : []);
      if (url === `${draftsBase}/draft-1`) return json(current);
      if (url === `${draftsBase}/draft-1/documents`) {
        const body = init!.body as FormData;
        const key = String(body.get("idempotency_key"));
        if (uploadedRequests.has(key)) return json(uploadedRequests.get(key));
        current = {
          ...current,
          revision: current.revision + 1,
          documents: [
            ...current.documents,
            ...(body.getAll("files") as File[]).map((file, index) => ({
              id:
                current.documents.length === 0 && index === 0
                  ? "root-1"
                  : `doc-${current.documents.length + index}`,
              display_name: file.name,
              parent_document_id: body.get("parent_document_id") as
                | string
                | null,
              byte_count: file.size,
              sha256: "a".repeat(64),
            })),
          ],
        };
        uploadedRequests.set(key, current);
        return json(current);
      }
      if (
        url.startsWith(`${draftsBase}/draft-1/documents/`) &&
        init?.method === "DELETE"
      ) {
        const id = url.split("/documents/")[1]!.split("?")[0];
        current = {
          ...current,
          revision: current.revision + 1,
          documents: current.documents.filter(
            (doc) => doc.id !== id && doc.parent_document_id !== id
          ),
        };
        return json(current);
      }
      if (url === `${draftsBase}/draft-1/prepare`) {
        current = {
          ...current,
          status: "prepared",
          source_package_id: packageSnapshot.id,
        };
        return json(packageSnapshot);
      }
      if (url === `${base}/batches/intake-staged`) return json(batch);
      if (url.endsWith("/reprepare")) return json({ eligible: false });
      throw new Error(`Unexpected request: ${init?.method ?? "GET"} ${url}`);
    }
  );
  global.fetch = fetchMock;
  return fetchMock;
}
async function openMarkdown() {
  const user = setupUser();
  await user.click(await screen.findByRole("button", { name: "Markdown" }));
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Markdown belgelerini ekle" })
    ).toBeInTheDocument()
  );
  return user;
}
function file(name: string) {
  return new File(["# Belge"], name, { type: "text/markdown" });
}
function writes(mock: ReturnType<typeof server>) {
  return mock.mock.calls.filter(([, init]) => init?.method === "POST");
}
beforeEach(() => {
  localStorage.clear();
});

test("stages multiple roots and annexes without processing until explicit prepare", async () => {
  const fetchMock = server();
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  expect(screen.queryByText(/manifest.json/)).not.toBeInTheDocument();
  await waitFor(() =>
    expect(screen.getByLabelText("Markdown belgeleri")).toBeEnabled()
  );
  await user.upload(screen.getByLabelText("Markdown belgeleri"), [
    file("ana.md"),
    file("ikinci.md"),
  ]);
  const main = await screen.findByRole("region", { name: "ana.md" });
  expect(
    await screen.findByRole("region", { name: "ikinci.md" })
  ).toBeVisible();
  const rootBody = writes(fetchMock).find(([url]) =>
    String(url).endsWith("/documents")
  )![1]!.body as FormData;
  expect(rootBody.getAll("files")).toHaveLength(2);
  expect(rootBody.has("parent_document_id")).toBe(false);
  expect(rootBody.get("expected_revision")).toBe("0");
  await user.click(
    within(main).getByRole("button", { name: "Belgeye ek ekle" })
  );
  await user.upload(screen.getByLabelText("ana.md için ek belgeler"), [
    file("ek-1.md"),
    file("ek-2.md"),
  ]);
  expect(await within(main).findByText("ek-1.md")).toBeVisible();
  const annexBody = writes(fetchMock).filter(([url]) =>
    String(url).endsWith("/documents")
  )[1]![1]!.body as FormData;
  expect(annexBody.get("parent_document_id")).toBe("root-1");
  expect(annexBody.getAll("files")).toHaveLength(2);
  expect(
    writes(fetchMock).every(
      ([url]) =>
        String(url).startsWith(draftsBase) && !String(url).endsWith("/prepare")
    )
  ).toBe(true);
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  expect(await screen.findByText("New documents: ready")).toBeVisible();
  expect(
    JSON.parse(
      writes(fetchMock).find(([url]) => String(url).endsWith("/prepare"))![1]!
        .body as string
    )
  ).toEqual({ expected_revision: 2 });
  expect(
    writes(fetchMock).filter(([url]) => url === `${base}/batches`)
  ).toHaveLength(1);
});

test("restores stored roots after reload and removes an entire attachment subtree", async () => {
  const fetchMock = server(
    draft([
      root,
      {
        ...root,
        id: "annex-1",
        display_name: "ek.md",
        parent_document_id: root.id,
      },
    ])
  );
  const view = render(<NewDocumentIntake />);
  const user = await openMarkdown();
  expect(await screen.findByText("ek.md")).toBeVisible();
  view.unmount();
  render(<NewDocumentIntake />);
  await openMarkdown();
  const main = await screen.findByRole("region", { name: "ana.md" });
  await user.click(
    within(main).getAllByRole("button", { name: "Belgeyi kaldır" })[0]!
  );
  await waitFor(() =>
    expect(screen.queryByText("ek.md")).not.toBeInTheDocument()
  );
  expect(fetchMock).toHaveBeenCalledWith(
    `${draftsBase}/draft-1/documents/root-1?expected_revision=2`,
    { method: "DELETE" }
  );
  expect(writes(fetchMock)).toHaveLength(0);
});

test("reuses create and upload identities after lost responses", async () => {
  const fetchMock = server();
  const dispatch = fetchMock.getMockImplementation()!;
  let createLost = true;
  let uploadLost = true;
  fetchMock.mockImplementation(async (input, init) => {
    if (String(input) === draftsBase && init?.method === "POST" && createLost) {
      createLost = false;
      await dispatch(input, init);
      throw new TypeError("Lost response");
    }
    if (String(input).endsWith("/documents") && uploadLost) {
      uploadLost = false;
      await dispatch(input, init);
      throw new TypeError("Lost response");
    }
    return dispatch(input, init);
  });
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await waitFor(() =>
    expect(screen.getByLabelText("Markdown belgeleri")).toBeEnabled()
  );
  await user.upload(
    screen.getByLabelText("Markdown belgeleri"),
    file("ana.md")
  );
  await user.click(
    await screen.findByRole("button", { name: "Yüklemeyi tekrar dene" })
  );
  await user.click(
    await screen.findByRole("button", { name: "Yüklemeyi tekrar dene" })
  );
  expect(await screen.findByRole("region", { name: "ana.md" })).toBeVisible();
  expect(screen.getAllByRole("region", { name: "ana.md" })).toHaveLength(1);
  const creates = writes(fetchMock).filter(([url]) => url === draftsBase);
  expect(JSON.parse(creates[0]![1]!.body as string).idempotency_key).toBe(
    JSON.parse(creates[1]![1]!.body as string).idempotency_key
  );
  const uploads = writes(fetchMock).filter(([url]) =>
    String(url).endsWith("/documents")
  );
  expect((uploads[0]![1]!.body as FormData).get("idempotency_key")).toBe(
    (uploads[1]![1]!.body as FormData).get("idempotency_key")
  );
});

test("stores documents without labeling configuration while blocking preparation", async () => {
  server(undefined, false);
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await waitFor(() =>
    expect(screen.getByLabelText("Markdown belgeleri")).toBeEnabled()
  );
  await user.upload(
    screen.getByLabelText("Markdown belgeleri"),
    file("ana.md")
  );
  expect(await screen.findByRole("region", { name: "ana.md" })).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Chunkları hazırla" })
  ).toBeDisabled();
});

test("freezes mutations during preparation and retries admission using the same prepared source", async () => {
  const fetchMock = server(draft([root]));
  const dispatch = fetchMock.getMockImplementation()!;
  const pending = deferred<Response>();
  let admissions = 0;
  fetchMock.mockImplementation(async (input, init) => {
    if (String(input).endsWith("/prepare")) return pending.promise;
    if (
      String(input) === `${base}/batches` &&
      init?.method === "POST" &&
      admissions++ === 0
    )
      throw new TypeError("Lost admission response");
    return dispatch(input, init);
  });
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await screen.findByText("ana.md");
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  expect(screen.getByLabelText("Markdown belgeleri")).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Belgeye ek ekle" })
  ).toBeDisabled();
  await act(async () =>
    pending.resolve(
      await dispatch(`${draftsBase}/draft-1/prepare`, { method: "POST" })
    )
  );
  const retry = await screen.findByRole("button", {
    name: "Hazırlamayı tekrar dene",
  });
  expect(screen.getByRole("button", { name: "Belgeyi kaldır" })).toBeDisabled();
  await user.click(retry);
  expect(await screen.findByText("New documents: ready")).toBeVisible();
  expect(
    writes(fetchMock).filter(([url]) => String(url).endsWith("/prepare"))
  ).toHaveLength(1);
  const requests = writes(fetchMock).filter(
    ([url]) => url === `${base}/batches`
  );
  expect(requests[0]![1]!.body).toBe(requests[1]![1]!.body);
});

test("resumes a prepared draft after reload without uploading or creating another source", async () => {
  const fetchMock = server({
    ...draft([root]),
    status: "prepared",
    source_package_id: "source-staged",
  });
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await screen.findByText("ana.md");
  expect(screen.getByLabelText("Markdown belgeleri")).toBeDisabled();
  await user.click(
    screen.getByRole("button", { name: "Hazırlamayı tekrar dene" })
  );
  expect(await screen.findByText("New documents: ready")).toBeVisible();
  expect(writes(fetchMock).map(([url]) => url)).toEqual([
    `${draftsBase}/draft-1/prepare`,
    `${base}/batches`,
  ]);
});

test("lets users replace a rejected upload instead of trapping them in an impossible retry", async () => {
  const fetchMock = server();
  const dispatch = fetchMock.getMockImplementation()!;
  let rejectUpload = true;
  fetchMock.mockImplementation(async (input, init) => {
    if (String(input).endsWith("/documents") && rejectUpload) {
      rejectUpload = false;
      return { ok: false, status: 400, json: async () => ({}) } as Response;
    }
    return dispatch(input, init);
  });
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await waitFor(() =>
    expect(screen.getByLabelText("Markdown belgeleri")).toBeEnabled()
  );
  await user.upload(
    screen.getByLabelText("Markdown belgeleri"),
    file("invalid.md")
  );
  await screen.findByText(/UTF-8/);
  expect(screen.getByLabelText("Markdown belgeleri")).toBeEnabled();
  await user.upload(
    screen.getByLabelText("Markdown belgeleri"),
    file("corrected.md")
  );
  expect(
    await screen.findByRole("region", { name: "corrected.md" })
  ).toBeVisible();
});

test("drops late preparation responses after the user leaves Markdown mode", async () => {
  const fetchMock = server(draft([root]));
  const dispatch = fetchMock.getMockImplementation()!;
  const pending = deferred<Response>();
  fetchMock.mockImplementation(async (input, init) =>
    String(input).endsWith("/prepare") ? pending.promise : dispatch(input, init)
  );
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await screen.findByText("ana.md");
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  await user.click(screen.getByRole("button", { name: "File" }));
  await act(async () =>
    pending.resolve(
      await dispatch(`${draftsBase}/draft-1/prepare`, { method: "POST" })
    )
  );
  expect(
    screen.getByRole("button", { name: "Choose document or ZIP" })
  ).toBeVisible();
  expect(writes(fetchMock).some(([url]) => url === `${base}/batches`)).toBe(
    false
  );
});

test("shows the server's actionable Turkish file-size error", async () => {
  const fetchMock = server();
  const dispatch = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation(async (input, init) =>
    String(input).endsWith("/documents")
      ? ({
          ok: false,
          status: 413,
          json: async () => ({
            detail: "Her Markdown belgesi en fazla 2 MiB olabilir.",
          }),
        } as Response)
      : dispatch(input, init)
  );
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await waitFor(() =>
    expect(screen.getByLabelText("Markdown belgeleri")).toBeEnabled()
  );
  await user.upload(
    screen.getByLabelText("Markdown belgeleri"),
    file("large.md")
  );
  expect(
    await screen.findByText("Her Markdown belgesi en fazla 2 MiB olabilir.")
  ).toBeVisible();
  expect(screen.getByLabelText("Markdown belgeleri")).toBeEnabled();
});

test("keeps newer uploaded documents when an older Strict Mode load resolves late", async () => {
  const fetchMock = server(draft([root]));
  const dispatch = fetchMock.getMockImplementation()!;
  const olderLoad = deferred<Response>();
  let listCalls = 0;
  fetchMock.mockImplementation(async (input, init) => {
    if (String(input) === draftsBase && !init?.method && listCalls++ === 0)
      return olderLoad.promise;
    return dispatch(input, init);
  });
  render(
    <StrictMode>
      <NewDocumentIntake />
    </StrictMode>
  );
  const user = await openMarkdown();
  await screen.findByText("ana.md");
  await user.upload(
    screen.getByLabelText("Markdown belgeleri"),
    file("new.md")
  );
  expect(await screen.findByRole("region", { name: "new.md" })).toBeVisible();
  await act(async () => olderLoad.resolve(json([draft([root])])));
  expect(screen.getByRole("region", { name: "new.md" })).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  const prepare = writes(fetchMock).find(([url]) =>
    String(url).endsWith("/prepare")
  )!;
  expect(JSON.parse(prepare[1]!.body as string)).toEqual({
    expected_revision: 2,
  });
});

test("makes a draft editable again after rejected preparation is confirmed by a fresh snapshot", async () => {
  const fetchMock = server(draft([root]));
  const dispatch = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation(async (input, init) =>
    String(input).endsWith("/prepare")
      ? ({ ok: false, status: 400, json: async () => ({}) } as Response)
      : dispatch(input, init)
  );
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await screen.findByText("ana.md");
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  await screen.findByText(/UTF-8/);
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Belgeye ek ekle" })
    ).toBeEnabled()
  );
  await user.click(screen.getByRole("button", { name: "Belgeye ek ekle" }));
  await user.upload(
    screen.getByLabelText("ana.md için ek belgeler"),
    file("missing.md")
  );
  expect(await screen.findByText("missing.md")).toBeVisible();
});

test("keeps an ambiguous preparation frozen until a later authoritative reload", async () => {
  const fetchMock = server(draft([root]));
  const dispatch = fetchMock.getMockImplementation()!;
  let snapshotUnavailable = true;
  fetchMock.mockImplementation(async (input, init) => {
    if (String(input).endsWith("/prepare"))
      throw new TypeError("Lost preparation response");
    if (String(input) === `${draftsBase}/draft-1` && snapshotUnavailable)
      throw new TypeError("Snapshot unavailable");
    return dispatch(input, init);
  });
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await screen.findByText("ana.md");
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  await screen.findByRole("button", { name: "Hazırlamayı tekrar dene" });
  expect(
    screen.getByRole("button", { name: "Belgeye ek ekle" })
  ).toBeDisabled();
  snapshotUnavailable = false;
  await user.click(screen.getByRole("button", { name: "Taslakları yenile" }));
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Belgeye ek ekle" })
    ).toBeEnabled()
  );
});

test("recovers the latest revision before retrying preparation after a conflict", async () => {
  const fresh = { ...draft([root]), revision: 7 };
  const fetchMock = server(fresh);
  const dispatch = fetchMock.getMockImplementation()!;
  let attempts = 0;
  fetchMock.mockImplementation(async (input, init) => {
    if (String(input) === draftsBase && !init?.method)
      return json([draft([root])]);
    if (String(input).endsWith("/prepare") && attempts++ === 0)
      return { ok: false, status: 409, json: async () => ({}) } as Response;
    return dispatch(input, init);
  });
  render(<NewDocumentIntake />);
  const user = await openMarkdown();
  await screen.findByText("ana.md");
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  await screen.findByText(/Güncel taslağı/);
  await user.click(screen.getByRole("button", { name: "Chunkları hazırla" }));
  expect(await screen.findByText("New documents: ready")).toBeVisible();
  expect(
    writes(fetchMock)
      .filter(([url]) => String(url).endsWith("/prepare"))
      .map(([, init]) => JSON.parse(init!.body as string).expected_revision)
  ).toEqual([1, 7]);
});
