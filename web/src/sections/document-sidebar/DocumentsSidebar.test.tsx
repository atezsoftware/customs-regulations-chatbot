import { fireEvent, render, screen } from "@testing-library/react";
import { Packet, PacketType } from "@/app/app/services/streamingModels";
import {
  useCurrentMessageTree,
  useSelectedNodeForDocDisplay,
} from "@/app/app/stores/useChatSessionStore";
import { OnyxDocument } from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";
import { TooltipProvider } from "@/components/ui/tooltip";
import DocumentsSidebar from "@/sections/document-sidebar/DocumentsSidebar";

jest.mock("@/app/app/stores/useChatSessionStore", () => ({
  useCurrentMessageTree: jest.fn(),
  useSelectedNodeForDocDisplay: jest.fn(),
}));

const mockedUseCurrentMessageTree = useCurrentMessageTree as jest.Mock;
const mockedUseSelectedNodeForDocDisplay =
  useSelectedNodeForDocDisplay as jest.Mock;

function buildDocument(chunkInd: number): OnyxDocument {
  return {
    document_id: "shared-law",
    semantic_identifier: `Customs Act — ARTICLE ${chunkInd}`,
    link: "/api/chat/file/shared-law",
    source_type: ValidSources.UserFile,
    blurb: `Exact article ${chunkInd}`,
    boost: 1,
    hidden: false,
    score: 1,
    chunk_ind: chunkInd,
    match_highlights: [],
    metadata: {},
    updated_at: null,
    is_internet: false,
  };
}

function buildCitationPacket(citationNumber: number, chunkInd: number): Packet {
  return {
    placement: { turn_index: 1 },
    obj: {
      type: PacketType.CITATION_INFO,
      citation_number: citationNumber,
      document_id: "shared-law",
      chunk_ind: chunkInd,
      semantic_identifier: `Customs Act — ARTICLE ${chunkInd}`,
      source_type: ValidSources.UserFile,
    },
  } as Packet;
}

describe("DocumentsSidebar cited sources", () => {
  beforeEach(() => {
    mockedUseSelectedNodeForDocDisplay.mockReturnValue(22);
  });

  it("keeps and opens two cited chunks from the same document independently", () => {
    const documents = [buildDocument(7), buildDocument(12)];
    const selectedMessage = {
      nodeId: 22,
      parentNodeId: null,
      packets: [buildCitationPacket(1, 7), buildCitationPacket(2, 12)],
      documents,
    };
    mockedUseCurrentMessageTree.mockReturnValue(
      new Map([[22, selectedMessage]])
    );
    const setPresentingDocument = jest.fn();

    render(
      <TooltipProvider>
        <DocumentsSidebar
          closeSidebar={jest.fn()}
          modal={false}
          selectedDocuments={null}
          setPresentingDocument={setPresentingDocument}
        />
      </TooltipProvider>
    );

    expect(
      screen.getAllByText("Customs Act — ARTICLE 7").length
    ).toBeGreaterThan(0);
    expect(
      screen.getAllByText("Customs Act — ARTICLE 12").length
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByText("Customs Act — ARTICLE 12")[0]!);

    expect(setPresentingDocument).toHaveBeenCalledWith({
      preview_type: "citation",
      citation_number: 2,
      document_id: "shared-law",
      semantic_identifier: "Customs Act — ARTICLE 12",
      citation_chunk_ind: 12,
    });
  });

  it("keeps citations usable when reloaded document-card metadata is absent", () => {
    const selectedMessage = {
      nodeId: 22,
      parentNodeId: null,
      packets: [buildCitationPacket(3, 46)],
      documents: [],
    };
    mockedUseCurrentMessageTree.mockReturnValue(
      new Map([[22, selectedMessage]])
    );
    const setPresentingDocument = jest.fn();

    render(
      <TooltipProvider>
        <DocumentsSidebar
          closeSidebar={jest.fn()}
          modal={false}
          selectedDocuments={null}
          setPresentingDocument={setPresentingDocument}
        />
      </TooltipProvider>
    );

    fireEvent.click(screen.getAllByText("Customs Act — ARTICLE 46")[0]!);

    expect(setPresentingDocument).toHaveBeenCalledWith({
      preview_type: "citation",
      citation_number: 3,
      document_id: "shared-law",
      semantic_identifier: "Customs Act — ARTICLE 46",
      citation_chunk_ind: 46,
    });
  });
});
