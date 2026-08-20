import { render, screen } from "@tests/setup/test-utils";
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
    error: undefined,
    isLoading: false,
    refreshChunks: jest.fn(),
  });
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
