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
  it.each(["", " \n\t "])(
    "preserves empty or whitespace retained admin content %j",
    async (content) => {
      const endpoint =
        "/api/admin/chat-session-history/9ea01263-7e32-4d85-b0d5-1e951b69f2fc/citation/6202/4";
      const fetchMock = jest
        .spyOn(global, "fetch")
        .mockResolvedValue(new Response(JSON.stringify({ content })));
      render(
        <PreviewModal
          presentingDocument={{
            preview_type: "citation",
            document_id: "source",
            semantic_identifier: "Official title",
            citation_number: 4,
            citation_chunk_ind: -4,
            citation_preview_url: endpoint,
            citation_resolution_status: "retained",
          }}
          onClose={jest.fn()}
        />
      );
      expect((await screen.findByLabelText("Atıf pasajı")).textContent).toBe(
        content
      );
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(mockedFetchChatFile).not.toHaveBeenCalled();
      fetchMock.mockRestore();
    }
  );
  it.each([
    [-4, "retained"],
    [4, "current"],
  ] as const)(
    "uses the authenticated admin target for index %s and labels current text",
    async (index, status) => {
      const endpoint =
        "/api/admin/chat-session-history/9ea01263-7e32-4d85-b0d5-1e951b69f2fc/citation/6202/4";
      const fetchMock = jest
        .spyOn(global, "fetch")
        .mockResolvedValue(
          new Response(JSON.stringify({ content: "Verified passage" }))
        );
      render(
        <PreviewModal
          presentingDocument={{
            preview_type: "citation",
            document_id: "source",
            semantic_identifier: "Official title",
            citation_number: 4,
            citation_chunk_ind: index,
            citation_preview_url: endpoint,
            citation_resolution_status: status,
          }}
          onClose={jest.fn()}
        />
      );
      expect(await screen.findByText("Verified passage")).toBeInTheDocument();
      expect(fetchMock).toHaveBeenCalledWith(endpoint);
      if (status === "current")
        expect(screen.getByRole("dialog")).toHaveAccessibleDescription(
          "Güncel kaynak parçası · Tarihî kayıtlı pasaj değildir"
        );
      expect(mockedFetchChatFile).not.toHaveBeenCalled();
      fetchMock.mockRestore();
    }
  );
  beforeEach(() => {
    mockedFetchChatFile.mockResolvedValue(
      new Response("the complete uploaded file", {
        headers: { "Content-Type": "text/plain" },
      })
    );
  });

  it("shows the supplied exact chunk and official title without requesting a preview", async () => {
    const fetchMock = jest.spyOn(global, "fetch");
    const officialTitle = "Resmî Başlık <script>alert(1)</script>";
    const { rerender } = render(
      <PreviewModal
        presentingDocument={{
          preview_type: "citation",
          document_id: "shared-file",
          semantic_identifier: officialTitle,
          citation_number: 10,
          citation_chunk_id: "canonical-10",
          citation_chunk_ind: 10,
          citation_text: "Article 10 exact original text",
          citation_preview_url: "/api/asv3/citation/101/10",
        }}
        onClose={jest.fn()}
      />
    );
    expect(
      await screen.findByText("Article 10 exact original text")
    ).toBeInTheDocument();
    expect(
      screen.getByRole("dialog", {
        name: (name) => name.startsWith(officialTitle),
      })
    ).toBeInTheDocument();

    rerender(
      <PreviewModal
        presentingDocument={{
          preview_type: "citation",
          document_id: "shared-file",
          semantic_identifier: officialTitle,
          citation_number: 11,
          citation_chunk_id: "native-11",
          citation_chunk_ind: -11,
          citation_text: "Second citation original source text",
        }}
        onClose={jest.fn()}
      />
    );
    expect(
      await screen.findByText("Second citation original source text")
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Article 10 exact original text")
    ).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(mockedFetchChatFile).not.toHaveBeenCalled();
  });

  it("does not fetch a different source when an inline chunk is empty", async () => {
    const fetchMock = jest.spyOn(global, "fetch");
    render(
      <PreviewModal
        presentingDocument={{
          preview_type: "citation",
          document_id: "shared-file",
          semantic_identifier: "Resmî Başlık",
          citation_chunk_ind: -2,
          citation_text: "",
        }}
        onClose={jest.fn()}
      />
    );
    expect(
      await screen.findByRole("dialog", { name: "Resmî Başlık" })
    ).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(mockedFetchChatFile).not.toHaveBeenCalled();
    expect(
      screen.queryByText(
        "The cited chunk is unavailable or you no longer have access to it."
      )
    ).not.toBeInTheDocument();
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
          preview_type: "citation",
          document_id: "customs-law",
          semantic_identifier: "Gümrük Kanunu · 46. Madde",
          citation_chunk_ind: 46,
          citation_text: null,
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
          preview_type: "citation",
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
            preview_type: "citation",
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
          preview_type: "citation",
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
          preview_type: "citation",
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

  it("fails closed when citation metadata has no chunk index", async () => {
    const fetchMock = jest.spyOn(global, "fetch");

    render(
      <PreviewModal
        presentingDocument={{
          preview_type: "citation",
          document_id: "customs-law",
          semantic_identifier: "Gümrük Kanunu · citation",
          citation_number: 9,
        }}
        onClose={jest.fn()}
      />
    );

    expect(
      await screen.findByText(
        "The cited chunk is unavailable or you no longer have access to it."
      )
    ).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(mockedFetchChatFile).not.toHaveBeenCalled();
    expect(screen.queryByText("Download File")).not.toBeInTheDocument();
  });
});

it("opens native-derived citations at their owned-message endpoint", async () => {
  const fetchMock = jest
    .spyOn(global, "fetch")
    .mockResolvedValue(
      new Response(
        JSON.stringify({ content: "Original source page 4 excerpt" })
      )
    );
  render(
    <PreviewModal
      presentingDocument={{
        preview_type: "citation",
        document_id: "original-pdf",
        semantic_identifier: "Original PDF · Page 4",
        citation_chunk_ind: -2,
        citation_preview_url: "/api/asv3/citation/101/2",
      }}
      onClose={jest.fn()}
    />
  );
  expect(
    await screen.findByText("Original source page 4 excerpt")
  ).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledWith("/api/asv3/citation/101/2");
  expect(mockedFetchChatFile).not.toHaveBeenCalled();
});

it.each([
  "https://external.invalid/source",
  "/api/asv3/citation/101/2?extra=true",
  "/api/asv3/citation/0/2",
])(
  "rejects untrusted native citation route %s",
  async (citation_preview_url) => {
    const fetchMock = jest.spyOn(global, "fetch");
    render(
      <PreviewModal
        presentingDocument={{
          preview_type: "citation",
          document_id: "original-pdf",
          semantic_identifier: "Original PDF",
          citation_chunk_ind: -2,
          citation_preview_url,
        }}
        onClose={jest.fn()}
      />
    );
    expect(
      await screen.findByText(
        "The cited chunk is unavailable or you no longer have access to it."
      )
    ).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(mockedFetchChatFile).not.toHaveBeenCalled();
  }
);
