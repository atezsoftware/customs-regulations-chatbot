import { StreamingCitation } from "@/app/app/services/streamingModels";
import { OnyxDocument } from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";
import {
  citationsToSourceInfoArray,
  getDisplayNameForSource,
} from "./sourceTagUtils";

function makeDocument(overrides: Partial<OnyxDocument> = {}): OnyxDocument {
  return {
    document_id: "customs-law",
    semantic_identifier:
      "Gümrük Kanunu.docx — 4458 SAYILI GÜMRÜK KANUNU > İKİNCİ KISIM > MADDE 46 > (1)",
    link: "",
    source_type: ValidSources.UserFile,
    blurb: "Eşya, gümrüğe sunulmasından sonra...",
    boost: 1,
    hidden: false,
    score: 1,
    chunk_ind: 46,
    match_highlights: [],
    metadata: {
      regulatory_heading_path:
        "4458 SAYILI GÜMRÜK KANUNU > İKİNCİ KISIM > MADDE 46 > (1)",
    },
    updated_at: null,
    is_internet: false,
    ...overrides,
  };
}

describe("regulatory citation presentation", () => {
  it("uses the legal document and article heading instead of the source type", () => {
    expect(getDisplayNameForSource(makeDocument())).toBe(
      "Gümrük Kanunu · 46. Madde"
    );
  });

  it("uses the uploaded document name when no legal article heading exists", () => {
    expect(
      getDisplayNameForSource(
        makeDocument({
          semantic_identifier: "İthalat Rehberi.pdf",
          metadata: {},
        })
      )
    ).toBe("İthalat Rehberi.pdf");
  });

  it("keeps the article visible when the legal document title is long", () => {
    expect(
      getDisplayNameForSource(
        makeDocument({
          semantic_identifier:
            "Gümrük İşlemlerinin Kolaylaştırılması Yönetmeliği.docx — MADDE 46",
          metadata: { regulatory_heading_path: "MADDE 46" },
        })
      )
    ).toMatch(/46\. Madde$/);
  });

  it("does not substitute another chunk from the same document", () => {
    const citations: StreamingCitation[] = [
      { citation_num: 1, document_id: "customs-law", chunk_ind: 47 },
    ];
    const documentMap = new Map<string, OnyxDocument>([
      ["customs-law:46", makeDocument()],
    ]);

    expect(citationsToSourceInfoArray(citations, documentMap)).toEqual([]);
  });
});
