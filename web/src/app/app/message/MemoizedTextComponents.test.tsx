import { fireEvent, render, screen } from "@testing-library/react";
import { OnyxDocument } from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";
import { MemoizedAnchor, MemoizedLink } from "./MemoizedTextComponents";

const citedDocument: OnyxDocument = {
  document_id: "customs-law",
  semantic_identifier: "Gümrük Kanunu.docx — MADDE 46",
  link: "",
  source_type: ValidSources.UserFile,
  blurb: "",
  boost: 1,
  hidden: false,
  score: 1,
  chunk_ind: 46,
  match_highlights: [],
  metadata: { regulatory_heading_path: "MADDE 46" },
  updated_at: null,
  is_internet: false,
};

describe("inline document citations", () => {
  it("opens an exact chunk preview instead of the complete uploaded file", () => {
    const updatePresentingDocument = jest.fn();

    render(
      <MemoizedLink
        document={citedDocument}
        citation={{
          citation_num: 1,
          document_id: "customs-law",
          chunk_ind: 46,
          semantic_identifier: "Gümrük Kanunu.docx — MADDE 46",
          source_type: ValidSources.UserFile,
        }}
        updatePresentingDocument={updatePresentingDocument}
      >
        [1]
      </MemoizedLink>
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Gümrük Kanunu · 46. Madde" })
    );

    expect(updatePresentingDocument).toHaveBeenCalledWith({
      preview_type: "citation",
      document_id: "customs-law",
      semantic_identifier: "Gümrük Kanunu · 46. Madde",
      citation_number: 1,
      citation_chunk_ind: 46,
    });
  });

  it("opens the exact citation even when the source has a direct file link", () => {
    const updatePresentingDocument = jest.fn();
    const openSpy = jest.spyOn(window, "open").mockImplementation(() => null);

    render(
      <MemoizedLink
        document={{
          ...citedDocument,
          link: "/api/chat/file/customs-law",
          source_type: ValidSources.IngestionApi,
        }}
        citation={{
          citation_num: 8,
          document_id: "customs-law",
          chunk_ind: 46,
          semantic_identifier: "Gümrük Kanunu.docx — MADDE 46",
          source_type: ValidSources.IngestionApi,
        }}
        updatePresentingDocument={updatePresentingDocument}
      >
        [8]
      </MemoizedLink>
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Gümrük Kanunu · 46. Madde" })
    );

    expect(openSpy).not.toHaveBeenCalled();
    expect(updatePresentingDocument).toHaveBeenCalledWith({
      preview_type: "citation",
      document_id: "customs-law",
      semantic_identifier: "Gümrük Kanunu · 46. Madde",
      citation_number: 8,
      citation_chunk_ind: 46,
    });
  });

  it("keeps an incomplete citation in fail-closed citation mode", () => {
    const updatePresentingDocument = jest.fn();
    const openSpy = jest.spyOn(window, "open").mockImplementation(() => null);

    render(
      <MemoizedLink
        document={citedDocument}
        citation={{
          citation_num: 9,
          document_id: "customs-law",
          semantic_identifier: "Gümrük Kanunu.docx — MADDE 46",
          source_type: ValidSources.UserFile,
        }}
        updatePresentingDocument={updatePresentingDocument}
      >
        [9]
      </MemoizedLink>
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Gümrük Kanunu · 46. Madde" })
    );

    expect(openSpy).not.toHaveBeenCalled();
    expect(updatePresentingDocument).toHaveBeenCalledWith({
      preview_type: "citation",
      document_id: "customs-law",
      semantic_identifier: "Gümrük Kanunu · 46. Madde",
      citation_number: 9,
      citation_chunk_ind: undefined,
    });
  });

  it("opens a citation packet even when its document card metadata is absent", () => {
    const updatePresentingDocument = jest.fn();

    render(
      <MemoizedAnchor
        docs={[]}
        citationReferences={[
          {
            citation_num: 12,
            document_id: "customs-law",
            chunk_ind: 46,
            semantic_identifier: "Gümrük Kanunu.docx — MADDE 46",
            source_type: ValidSources.UserFile,
          },
        ]}
        updatePresentingDocument={updatePresentingDocument}
      >
        [12]
      </MemoizedAnchor>
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Gümrük Kanunu · 46. Madde" })
    );

    expect(updatePresentingDocument).toHaveBeenCalledWith({
      preview_type: "citation",
      citation_number: 12,
      document_id: "customs-law",
      semantic_identifier: "Gümrük Kanunu · 46. Madde",
      citation_chunk_ind: 46,
    });
  });
});

it("preserves the source-specific native preview target on a citation chip", () => {
  const updatePresentingDocument = jest.fn();
  render(
    <MemoizedLink
      document={{
        ...citedDocument,
        document_id: "original-pdf",
        chunk_ind: -2,
        semantic_identifier:
          "Garanti şartları.pdf · sayfa 4 (özgün belgeden çıkarılan metin)",
      }}
      citation={{
        citation_num: 2,
        document_id: "original-pdf",
        chunk_ind: -2,
        semantic_identifier:
          "Garanti şartları.pdf · sayfa 4 (özgün belgeden çıkarılan metin)",
        source_type: ValidSources.UserFile,
        preview_url: "/api/asv3/citation/101/2",
      }}
      updatePresentingDocument={updatePresentingDocument}
    >
      [2]
    </MemoizedLink>
  );
  fireEvent.click(screen.getByRole("button"));
  expect(updatePresentingDocument).toHaveBeenCalledWith(
    expect.objectContaining({
      preview_type: "citation",
      document_id: "original-pdf",
      citation_chunk_ind: -2,
      citation_preview_url: "/api/asv3/citation/101/2",
    })
  );
});

it("opens ephemeral internet citations at the original public page without a synthetic chunk preview", () => {
  const updatePresentingDocument = jest.fn();
  const openSpy = jest.spyOn(window, "open").mockImplementation(() => null);
  render(
    <MemoizedLink
      document={{
        ...citedDocument,
        source_type: ValidSources.Web,
        is_internet: true,
        link: "https://official.example/legal/source",
        chunk_ind: 0,
      }}
      citation={{
        citation_num: 3,
        document_id: citedDocument.document_id,
        chunk_ind: 0,
        source_type: ValidSources.Web,
      }}
      updatePresentingDocument={updatePresentingDocument}
    >
      [3]
    </MemoizedLink>
  );
  fireEvent.click(screen.getByRole("button"));
  expect(openSpy).toHaveBeenCalledWith(
    "https://official.example/legal/source",
    "_blank",
    "noopener,noreferrer"
  );
  expect(updatePresentingDocument).not.toHaveBeenCalled();
  openSpy.mockRestore();
});

it.each(["javascript:alert(1)", "file:///private/source", "/api/private/file"])(
  "rejects unsafe ephemeral internet target %s",
  (link) => {
    const updatePresentingDocument = jest.fn();
    const openSpy = jest.spyOn(window, "open").mockImplementation(() => null);
    render(
      <MemoizedLink
        document={{
          ...citedDocument,
          source_type: ValidSources.Web,
          is_internet: true,
          link,
          chunk_ind: 0,
        }}
        citation={{
          citation_num: 3,
          document_id: citedDocument.document_id,
          chunk_ind: 0,
          source_type: ValidSources.Web,
        }}
        updatePresentingDocument={updatePresentingDocument}
      >
        [3]
      </MemoizedLink>
    );
    fireEvent.click(screen.getByRole("button"));
    expect(openSpy).not.toHaveBeenCalled();
    expect(updatePresentingDocument).not.toHaveBeenCalled();
    openSpy.mockRestore();
  }
);

it("keeps indexed web sources on their exact canonical chunk", () => {
  const updatePresentingDocument = jest.fn();
  const openSpy = jest.spyOn(window, "open").mockImplementation(() => null);
  render(
    <MemoizedLink
      document={{
        ...citedDocument,
        source_type: ValidSources.Web,
        is_internet: false,
        link: "https://official.example/legal/source",
      }}
      citation={{
        citation_num: 3,
        document_id: citedDocument.document_id,
        chunk_ind: 46,
        source_type: ValidSources.Web,
      }}
      updatePresentingDocument={updatePresentingDocument}
    >
      [3]
    </MemoizedLink>
  );
  fireEvent.click(screen.getByRole("button"));
  expect(openSpy).not.toHaveBeenCalled();
  expect(updatePresentingDocument).toHaveBeenCalledWith(
    expect.objectContaining({
      preview_type: "citation",
      citation_chunk_ind: 46,
    })
  );
  openSpy.mockRestore();
});
