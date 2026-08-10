import { fireEvent, render, screen } from "@testing-library/react";
import { OnyxDocument } from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";
import { MemoizedLink } from "./MemoizedTextComponents";

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
        updatePresentingDocument={updatePresentingDocument}
      >
        [1]
      </MemoizedLink>
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Gümrük Kanunu · 46. Madde" })
    );

    expect(updatePresentingDocument).toHaveBeenCalledWith({
      document_id: "customs-law",
      semantic_identifier: "Gümrük Kanunu · 46. Madde",
      citation_chunk_ind: 46,
    });
  });
});
