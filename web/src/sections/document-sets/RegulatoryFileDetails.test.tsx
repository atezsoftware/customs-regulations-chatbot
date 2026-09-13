import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import RegulatoryFileDetails from "@/sections/document-sets/RegulatoryFileDetails";
import { UserFileStatus } from "@/lib/projects/types";

const mockUseFileChunks = jest.fn();

jest.mock("@/lib/regulatory/hooks", () => ({
  useFileChunks: (...args: unknown[]) => mockUseFileChunks(...args),
}));

const FILE = {
  id: "file-1",
  name: "mevzuat/1975_tir_sozlesmesi.md",
  status: UserFileStatus.CHUNKED,
  chunk_count: 2,
};

const CHUNK = {
  id: "chunk-abc",
  position: 3,
  chunk_type: "madde",
  heading_path: ["BİRİNCİ BÖLÜM", "Madde 5"],
  text: "Gümrük müşavirliği şirketleri.",
  status: "active",
  source: "import",
  validity_start_date: null,
  validity_end_date: null,
  chunk_metadata: {},
};

beforeEach(() => {
  mockUseFileChunks.mockReturnValue({
    chunks: [CHUNK],
    total: 1,
    error: undefined,
    isLoading: false,
    refreshChunks: jest.fn(),
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

function renderDetails() {
  return render(
    <RegulatoryFileDetails file={FILE as never} onFileRenamed={jest.fn()} />
  );
}

test("links to the whole document as a PDF", async () => {
  renderDetails();

  const link = await screen.findByRole("link", { name: /document pdf/i });

  expect(link).toHaveAttribute("href", "/api/regulatory/files/file-1/pdf");
  // Opens in the browser's viewer rather than replacing the admin page.
  expect(link).toHaveAttribute("target", "_blank");
});

test("links to each chunk as its own PDF", async () => {
  renderDetails();

  const link = await screen.findByRole("link", { name: /chunk pdf/i });

  expect(link).toHaveAttribute("href", "/api/regulatory/chunks/chunk-abc/pdf");
});

test("requests and renders one chunk page at a time", async () => {
  const user = setupUser();
  const secondPageChunk = {
    ...CHUNK,
    id: "chunk-page-2",
    position: 26,
    text: "Only the second page chunk is rendered.",
  };
  mockUseFileChunks.mockImplementation(
    (_fileId: string, offset: number, limit: number) => ({
      chunks: offset === 25 ? [secondPageChunk] : [CHUNK],
      total: 26,
      error: undefined,
      isLoading: false,
      refreshChunks: jest.fn(),
      offset,
      limit,
    })
  );

  renderDetails();

  expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-1", 0, 25);
  expect(screen.getByText("1–1 of 26")).toBeInTheDocument();
  expect(
    screen.queryByText("Only the second page chunk is rendered.")
  ).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Next" }));

  await waitFor(() =>
    expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-1", 25, 25)
  );
  expect(screen.getByText("26–26 of 26")).toBeInTheDocument();
  expect(
    screen.getByText("Only the second page chunk is rendered.")
  ).toBeInTheDocument();
  expect(screen.queryByText(CHUNK.text)).not.toBeInTheDocument();
});

test("resets the chunk page when the selected file changes", async () => {
  const user = setupUser();
  mockUseFileChunks.mockImplementation(
    (fileId: string, offset: number, limit: number) => ({
      chunks: [{ ...CHUNK, id: `${fileId}-${offset}` }],
      total: 26,
      error: undefined,
      isLoading: false,
      refreshChunks: jest.fn(),
      offset,
      limit,
    })
  );
  const view = renderDetails();
  await user.click(screen.getByRole("button", { name: "Next" }));
  await waitFor(() =>
    expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-1", 25, 25)
  );
  await user.click(screen.getByRole("button", { name: "Edit chunk 3" }));
  expect(screen.getByText("Edit Chunk #3")).toBeInTheDocument();

  view.rerender(
    <RegulatoryFileDetails
      file={{ ...FILE, id: "file-2", name: "second.md" } as never}
      onFileRenamed={jest.fn()}
    />
  );

  await waitFor(() =>
    expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-2", 0, 25)
  );
  expect(screen.queryByText("Edit Chunk #3")).not.toBeInTheDocument();
});

test("clamps to the last available page when the total shrinks", async () => {
  const user = setupUser();
  let total = 26;
  mockUseFileChunks.mockImplementation(
    (_fileId: string, offset: number, limit: number) => ({
      chunks: offset < total ? [CHUNK] : [],
      total,
      error: undefined,
      isLoading: false,
      refreshChunks: jest.fn(),
      offset,
      limit,
    })
  );
  const view = renderDetails();
  await user.click(screen.getByRole("button", { name: "Next" }));
  await waitFor(() =>
    expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-1", 25, 25)
  );

  total = 20;
  view.rerender(
    <RegulatoryFileDetails file={FILE as never} onFileRenamed={jest.fn()} />
  );

  await waitFor(() =>
    expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-1", 0, 25)
  );
  expect(screen.getByText("1–1 of 20")).toBeInTheDocument();
});

test("keeps the current offset when a later page fails to load", async () => {
  const user = setupUser();
  mockUseFileChunks.mockImplementation(
    (_fileId: string, offset: number, limit: number) => ({
      chunks: offset === 0 ? [CHUNK] : [],
      total: offset === 0 ? 26 : 0,
      error: offset === 0 ? undefined : new Error("Page fetch failed"),
      isLoading: false,
      refreshChunks: jest.fn(),
      offset,
      limit,
    })
  );
  renderDetails();

  await user.click(screen.getByRole("button", { name: "Next" }));

  await waitFor(() =>
    expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-1", 25, 25)
  );
  expect(
    screen.getByText("Failed to load chunks for this file.")
  ).toBeInTheDocument();
});

test("refreshes the current page after editing a chunk", async () => {
  const user = setupUser();
  const refreshChunks = jest.fn().mockResolvedValue(undefined);
  mockUseFileChunks.mockImplementation(
    (_fileId: string, offset: number, limit: number) => ({
      chunks: [{ ...CHUNK, position: offset + 1 }],
      total: 26,
      error: undefined,
      isLoading: false,
      refreshChunks,
      offset,
      limit,
    })
  );
  jest.spyOn(global, "fetch").mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => CHUNK,
  } as Response);
  renderDetails();
  await user.click(screen.getByRole("button", { name: "Next" }));
  await user.click(screen.getByRole("button", { name: "Edit chunk 26" }));
  await user.click(screen.getByRole("button", { name: "Save & Re-index" }));

  await waitFor(() => expect(refreshChunks).toHaveBeenCalledTimes(1));
  expect(mockUseFileChunks).toHaveBeenLastCalledWith("file-1", 25, 25);
});
