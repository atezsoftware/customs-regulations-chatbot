import { OnyxDocument } from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";
import { getDocumentSourceDisplayName } from "@/lib/regulatory/sourceDisplayName";

function documentWith(
  semanticIdentifier: string,
  metadata: OnyxDocument["metadata"] = {}
): OnyxDocument {
  return {
    document_id: "source",
    semantic_identifier: semanticIdentifier,
    link: "",
    source_type: ValidSources.UserFile,
    blurb: "",
    boost: 0,
    hidden: false,
    score: 1,
    chunk_ind: 1,
    match_highlights: [],
    metadata,
    updated_at: null,
    is_internet: false,
  };
}

describe("regulatory source names", () => {
  it("prefers the verified source name without changing the canonical document", () => {
    const document = documentWith("Kanunlar/gumruk_kanunu.docx", {
      asv3_source_display_name: "4458 Sayılı Gümrük Kanunu",
      regulatory_heading_path: ["GÜMRÜK KANUNU", "MADDE 168"],
    });

    expect(getDocumentSourceDisplayName(document)).toBe(
      "4458 Sayılı Gümrük Kanunu"
    );
    expect(document.semantic_identifier).toBe("Kanunlar/gumruk_kanunu.docx");
  });

  it("uses an official heading for historical regulatory documents", () => {
    expect(
      getDocumentSourceDisplayName(
        documentWith("Mevzuat/KDV/katma_deger_vergisi_kanunu.md", {
          regulatory_heading_path: [
            "3065 SAYILI KATMA DEĞER VERGİSİ KANUNU",
            "MADDE 16",
          ],
        })
      )
    ).toBe("3065 SAYILI KATMA DEĞER VERGİSİ KANUNU");
  });

  it("preserves slash year-numbers only with a verified circular type or title", () => {
    expect(
      getDocumentSourceDisplayName(
        documentWith("Genelgeler/genelge_2024-22_geri_gelen_esya.md", {
          document_type: "genelge",
          regulatory_heading_path: ["TİCARET BAKANLIĞI", "(2024/22)"],
        })
      )
    ).toBe("2024/22 sayılı Genelge");
    expect(
      getDocumentSourceDisplayName(
        documentWith("Genelgeler/source.md", {
          regulatory_heading_path: ["Genelge 2024/22"],
        })
      )
    ).toBe("Genelge 2024/22");
  });

  it("uses a readable filename without inventing a circular from a bare number", () => {
    expect(
      getDocumentSourceDisplayName(
        documentWith("Genelgeler/genelge_2024-22_geri_gelen_esya.docx", {
          regulatory_heading_path: ["TİCARET BAKANLIĞI", "(2024/22)"],
        })
      )
    ).toBe("genelge 2024-22 geri gelen esya");
  });

  it("does not display a folder path masquerading as an official heading", () => {
    expect(
      getDocumentSourceDisplayName(
        documentWith(
          "Tebliğler/katma_deger_vergisi_genel_uygulama_tebligi.md",
          {
            regulatory_heading_path: ["Tebliğler/KDV Genel Uygulama Tebliği"],
          }
        )
      )
    ).toBe("katma deger vergisi genel uygulama tebligi");
  });

  it("keeps ordinary uploaded document names unchanged", () => {
    const name = "Projects/customer_notes_v2.docx";
    expect(getDocumentSourceDisplayName(documentWith(name))).toBe(name);
  });
});
