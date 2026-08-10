import { act, render, screen, waitFor } from "@testing-library/react";
import { fetchChatFile } from "@/lib/chat/svc";
import PreviewModal from "./PreviewModal";

jest.mock("@/lib/chat/svc", () => ({
  fetchChatFile: jest.fn(),
}));

const mockedFetchChatFile = fetchChatFile as jest.MockedFunction<
  typeof fetchChatFile
>;

describe("PreviewModal citation chunk mode", () => {
  beforeEach(() => {
    mockedFetchChatFile.mockResolvedValue(
      new Response("the complete uploaded file", {
        headers: { "Content-Type": "text/plain" },
      })
    );
  });

  it("fetches only the exact cited chunk", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(
        new Response(JSON.stringify({ content: "Article 46 exact chunk" }))
      );

    render(
      <PreviewModal
        presentingDocument={{
          document_id: "customs-law",
          semantic_identifier: "Gümrük Kanunu · 46. Madde",
          citation_chunk_ind: 46,
        }}
        onClose={jest.fn()}
      />
    );

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/document/chunk-info?document_id=customs-law&chunk_id=46"
      )
    );
    expect(mockedFetchChatFile).not.toHaveBeenCalled();
    expect(
      await screen.findByText("Article 46 exact chunk")
    ).toBeInTheDocument();
    expect(
      screen.queryByText("the complete uploaded file")
    ).not.toBeInTheDocument();
  });

  it("does not fall back to the full file when the cited chunk is unavailable", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(new Response(null, { status: 404 }));

    render(
      <PreviewModal
        presentingDocument={{
          document_id: "customs-law",
          semantic_identifier: "Gümrük Kanunu · 46. Madde",
          citation_chunk_ind: 46,
        }}
        onClose={jest.fn()}
      />
    );

    expect(
      await screen.findByText(
        "The cited chunk is unavailable or you no longer have access to it."
      )
    ).toBeInTheDocument();
    expect(screen.queryByText("Download File")).not.toBeInTheDocument();
    expect(mockedFetchChatFile).not.toHaveBeenCalled();
  });

  it.each([
    ["missing", {}],
    ["empty", { content: "" }],
    ["non-string", { content: 46 }],
  ])(
    "rejects a %s citation content payload without falling back to the full file",
    async (_case, payload) => {
      jest
        .spyOn(global, "fetch")
        .mockResolvedValue(new Response(JSON.stringify(payload)));

      render(
        <PreviewModal
          presentingDocument={{
            document_id: "customs-law",
            semantic_identifier: "Gümrük Kanunu · 46. Madde",
            citation_chunk_ind: 46,
          }}
          onClose={jest.fn()}
        />
      );

      expect(
        await screen.findByText(
          "The cited chunk is unavailable or you no longer have access to it."
        )
      ).toBeInTheDocument();
      expect(screen.queryByText("Download File")).not.toBeInTheDocument();
      expect(mockedFetchChatFile).not.toHaveBeenCalled();
    }
  );

  it("keeps the newest cited chunk when an older response resolves last", async () => {
    let resolveOlderResponse!: (response: Response) => void;
    const olderResponse = new Promise<Response>((resolve) => {
      resolveOlderResponse = resolve;
    });
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockReturnValueOnce(olderResponse)
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ content: "Current Article 47 chunk" }))
      );

    const { rerender } = render(
      <PreviewModal
        presentingDocument={{
          document_id: "customs-law",
          semantic_identifier: "Gümrük Kanunu · 46. Madde",
          citation_chunk_ind: 46,
        }}
        onClose={jest.fn()}
      />
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    rerender(
      <PreviewModal
        presentingDocument={{
          document_id: "customs-law",
          semantic_identifier: "Gümrük Kanunu · 47. Madde",
          citation_chunk_ind: 47,
        }}
        onClose={jest.fn()}
      />
    );
    expect(
      await screen.findByText("Current Article 47 chunk")
    ).toBeInTheDocument();

    await act(async () => {
      resolveOlderResponse(
        new Response(JSON.stringify({ content: "Stale Article 46 chunk" }))
      );
    });

    await waitFor(() =>
      expect(
        screen.queryByText("Stale Article 46 chunk")
      ).not.toBeInTheDocument()
    );
    expect(screen.getByText("Current Article 47 chunk")).toBeInTheDocument();
    expect(mockedFetchChatFile).not.toHaveBeenCalled();
  });
});
