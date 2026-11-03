import { render, screen } from "@tests/setup/test-utils";
import RetainedPdfEvidenceCard from "@/sections/cards/RetainedPdfEvidenceCard";
import type { RetainedPdfEvidencePage } from "@/lib/regulatory/interfaces";

const mockUseSWR = jest.fn();

jest.mock("swr", () => ({
  ...jest.requireActual<typeof import("swr")>("swr"),
  __esModule: true,
  default: (...args: unknown[]) => mockUseSWR(...args),
}));

function evidencePage(
  relation: "equivalent_rendition" | "historical_rendition"
): RetainedPdfEvidencePage {
  return {
    document_set_id: 42,
    user_file_id: "file-1",
    canonical_sha256: "a".repeat(64),
    canonical_generation_hash: "b".repeat(64),
    items: [
      {
        id: "evidence-1",
        original_filename: "original.pdf",
        relation,
        reason: "The original PDF is retained with its legislation record.",
        source_url: null,
        byte_count: 1024,
        created_at: "2026-11-03T00:00:00Z",
        canonical_sha256: "a".repeat(64),
        canonical_generation_hash: "b".repeat(64),
      },
    ],
  };
}

function mockPage(page: RetainedPdfEvidencePage) {
  mockUseSWR.mockImplementation((key: unknown) => ({
    data:
      key === "document-intake-capability"
        ? { enabled: true, target_document_set_id: 42 }
        : page,
    error: undefined,
    isLoading: false,
  }));
}

test("identifies a historical PDF and uses the returned document set for download", () => {
  mockPage(evidencePage("historical_rendition"));
  render(<RetainedPdfEvidenceCard fileId="file-1" />);

  expect(screen.getByText("Historical source")).toBeInTheDocument();
  expect(
    screen.getByText(
      "This archived PDF does not replace the current legislation text."
    )
  ).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Open source PDF" })).toHaveAttribute(
    "href",
    "/api/regulatory/files/file-1/retained-pdf-evidence/evidence-1/download?document_set_id=42"
  );
});

test("keeps equivalent source PDFs distinct from historical renditions", () => {
  mockPage(evidencePage("equivalent_rendition"));
  render(<RetainedPdfEvidenceCard fileId="file-1" />);

  expect(screen.getByText("Equivalent source")).toBeInTheDocument();
  expect(screen.queryByText("Historical source")).not.toBeInTheDocument();
});

test("does not show another file's cached source PDF after navigation", () => {
  mockPage(evidencePage("historical_rendition"));
  render(<RetainedPdfEvidenceCard fileId="file-2" />);

  expect(
    screen.queryByRole("region", { name: "Original source PDFs" })
  ).not.toBeInTheDocument();
});

test("does not claim equivalence after the reviewed canonical version changes", () => {
  const page = evidencePage("equivalent_rendition");
  page.canonical_sha256 = "c".repeat(64);
  mockPage(page);
  render(<RetainedPdfEvidenceCard fileId="file-1" />);

  expect(screen.queryByText("Equivalent source")).not.toBeInTheDocument();
  expect(screen.getByText("Earlier retained source")).toBeInTheDocument();
  expect(
    screen.getByText(/Equivalence to the current text has not been reconfirmed/)
  ).toBeInTheDocument();
});
