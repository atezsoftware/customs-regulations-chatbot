import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import DocumentSetFiles from "@/sections/document-sets/DocumentSetFiles";
import { UserFileStatus } from "@/lib/projects/types";

const mockUploadDocumentSetFiles = jest.fn();
const mockGetDocumentSetFiles = jest.fn();
const mockIndexDocumentSetFile = jest.fn();
const mockIndexDocumentSetChunkedFiles = jest.fn();

jest.mock("@/app/admin/documents/sets/lib", () => ({
  __esModule: true,
  getDocumentSetFiles: (...args: unknown[]) => mockGetDocumentSetFiles(...args),
  getDocumentSetFilesKey: (documentSetId: number) =>
    `document-set-files-${documentSetId}`,
  unlinkFileFromDocumentSet: jest.fn().mockResolvedValue(undefined),
  uploadDocumentSetFiles: (...args: unknown[]) =>
    mockUploadDocumentSetFiles(...args),
  indexDocumentSetFile: (...args: unknown[]) =>
    mockIndexDocumentSetFile(...args),
  indexDocumentSetChunkedFiles: (...args: unknown[]) =>
    mockIndexDocumentSetChunkedFiles(...args),
}));

const mockUseSettings = jest.fn();

jest.mock("@/lib/settings/hooks", () => ({
  useSettings: () => mockUseSettings(),
}));

jest.mock("@/sections/document-sets/RegulatoryFileDetails", () => ({
  __esModule: true,
  default: ({ file }: { file: { name: string } }) => (
    <div>{`File details for ${file.name}`}</div>
  ),
}));

function renderFiles() {
  return render(
    <DocumentSetFiles documentSetId={7} documentSetName="Gumruk Mevzuati" />
  );
}

beforeEach(() => {
  mockUseSettings.mockReturnValue({
    document_import_enabled: true,
    markdown_import_enabled: true,
  });
  mockGetDocumentSetFiles.mockReset();
  mockGetDocumentSetFiles.mockResolvedValue([]);
  mockIndexDocumentSetFile.mockReset();
  mockIndexDocumentSetFile.mockResolvedValue(undefined);
  mockIndexDocumentSetChunkedFiles.mockReset();
  mockIndexDocumentSetChunkedFiles.mockResolvedValue({ queued: 1 });
  mockUploadDocumentSetFiles.mockReset();
  mockUploadDocumentSetFiles.mockResolvedValue({
    user_files: [],
    rejected_files: [],
  });
});

test("offers an archive upload control restricted to zip files", async () => {
  renderFiles();

  const archiveInput = await screen.findByLabelText(
    "Upload archive to document set"
  );

  expect(archiveInput).toHaveAttribute("accept", ".zip");
});

test("uploads a selected archive through the document set upload endpoint", async () => {
  const user = setupUser();
  renderFiles();

  const archiveInput = await screen.findByLabelText(
    "Upload archive to document set"
  );
  const archive = new File(
    [new Uint8Array([0x50, 0x4b, 0x03, 0x04])],
    "mevzuat.zip",
    { type: "application/zip" }
  );

  await user.upload(archiveInput, archive);

  await waitFor(() => {
    expect(mockUploadDocumentSetFiles).toHaveBeenCalledWith(7, [archive]);
  });
});

describe("when the runtime can only ingest markdown", () => {
  beforeEach(() => {
    mockUseSettings.mockReturnValue({
      document_import_enabled: false,
      markdown_import_enabled: true,
    });
  });

  test("still offers the archive upload control", async () => {
    renderFiles();

    expect(
      await screen.findByLabelText("Upload archive to document set")
    ).toBeInTheDocument();
  });

  test("restricts the file picker to markdown", async () => {
    renderFiles();

    const fileInput = await screen.findByLabelText(
      "Upload files to document set"
    );

    expect(fileInput).toHaveAttribute("accept", ".md,.mdx,.zip");
  });

  test("says which formats this deployment accepts", async () => {
    renderFiles();

    expect(
      await screen.findByText(/only Markdown documents and \.zip archives/i)
    ).toBeInTheDocument();
  });
});

describe("when the runtime cannot ingest anything", () => {
  beforeEach(() => {
    mockUseSettings.mockReturnValue({
      document_import_enabled: false,
      markdown_import_enabled: false,
    });
  });

  test("hides the upload controls and points at the importer deployment", async () => {
    renderFiles();

    expect(
      await screen.findByText(/separate importer deployment/i)
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Upload archive to document set")
    ).not.toBeInTheDocument();
  });
});

describe("indexing reviewed chunks", () => {
  const chunkedFile = {
    id: "file-1",
    name: "mevzuat/1975_tir_sozlesmesi.md",
    status: UserFileStatus.CHUNKED,
    chunk_count: 223,
  };

  beforeEach(() => {
    mockGetDocumentSetFiles.mockResolvedValue([chunkedFile]);
  });

  test("offers to index a file whose chunks are ready for review", async () => {
    renderFiles();

    expect(
      await screen.findByRole("button", { name: /^index$/i })
    ).toBeInTheDocument();
  });

  test("offers distinguishable PDF links for files in every status", async () => {
    mockGetDocumentSetFiles.mockResolvedValue([
      chunkedFile,
      {
        id: "file-2",
        name: "mevzuat/gumruk-kanunu.md",
        status: UserFileStatus.COMPLETED,
        chunk_count: 40,
      },
    ]);
    renderFiles();

    const chunkedPdfLink = await screen.findByRole("link", {
      name: "Open mevzuat/1975_tir_sozlesmesi.md as PDF in a new tab",
    });
    const completedPdfLink = screen.getByRole("link", {
      name: "Open mevzuat/gumruk-kanunu.md as PDF in a new tab",
    });

    expect(chunkedPdfLink).toHaveAttribute(
      "href",
      "/api/regulatory/files/file-1/pdf"
    );
    expect(chunkedPdfLink).toHaveAttribute("target", "_blank");
    expect(completedPdfLink).toHaveAttribute(
      "href",
      "/api/regulatory/files/file-2/pdf"
    );
  });

  test("keeps the PDF action separate from chunk-detail navigation", async () => {
    const user = setupUser();
    renderFiles();

    const pdfLink = await screen.findByRole("link", {
      name: "Open mevzuat/1975_tir_sozlesmesi.md as PDF in a new tab",
    });
    pdfLink.addEventListener("click", (event) => event.preventDefault(), {
      once: true,
    });

    await user.click(pdfLink);
    expect(
      screen.queryByText("File details for mevzuat/1975_tir_sozlesmesi.md")
    ).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("button", {
        name: /mevzuat\/1975_tir_sozlesmesi\.md/i,
      })
    );
    expect(
      await screen.findByText("File details for mevzuat/1975_tir_sozlesmesi.md")
    ).toBeInTheDocument();
  });

  test("offers a bulk action for every chunked file in the set", async () => {
    renderFiles();

    expect(
      await screen.findByRole("button", { name: /index all chunked \(1\)/i })
    ).toBeInTheDocument();
  });

  test("indexes the file through the document set endpoint", async () => {
    const user = setupUser();
    renderFiles();

    await user.click(await screen.findByRole("button", { name: /^index$/i }));

    await waitFor(() => {
      expect(mockIndexDocumentSetFile).toHaveBeenCalledWith(7, "file-1");
    });
  });

  test("hides the indexing actions once a file is indexed", async () => {
    mockGetDocumentSetFiles.mockResolvedValue([
      { ...chunkedFile, status: UserFileStatus.COMPLETED },
    ]);
    renderFiles();

    expect(await screen.findByText(/223 chunks/i)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^index$/i })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /index all chunked/i })
    ).not.toBeInTheDocument();
  });
});

test("shows durable indexing stage and trustworthy item progress without an ETA", async () => {
  mockGetDocumentSetFiles.mockResolvedValue([
    {
      id: "file-2",
      name: "mevzuat/gumruk-kanunu.md",
      status: UserFileStatus.INDEXING,
      chunk_count: 40,
      regulatory_indexing_progress: {
        job_id: "job-2",
        status: "RETRY_WAIT",
        stage: "CONTEXT_WAIT",
        total_items: 40,
        completed_items: 12,
        context_ready_items: 12,
        embedded_items: 0,
        failed_items: 0,
        attempt_count: 2,
        next_retry_at: "2026-08-20T12:00:00Z",
        error_summary: "Dizinleme otomatik yeniden denemeyi bekliyor.",
        provider_batch_state: "vertex:SUBMITTED",
      },
    },
  ]);

  renderFiles();

  expect(
    await screen.findByText(/Bağlam işi bekleniyor · 12\/40 \(30%\)/i)
  ).toBeInTheDocument();
  expect(
    screen.getByText(/Deneme 2 · Yeniden deneme planlandı/i)
  ).toBeInTheDocument();
  expect(screen.queryByText(/ETA/i)).not.toBeInTheDocument();
});
