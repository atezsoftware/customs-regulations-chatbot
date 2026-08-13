import { fireEvent, render, screen } from "@testing-library/react";
import type { MessageSnapshot } from "../../usage/types";
import QueryHistoryTranscript from "./QueryHistoryTranscript";

const messages: MessageSnapshot[] = [
  {
    id: 1,
    message: "How do I calculate customs duty?",
    message_type: "user",
    documents: [],
    feedback_type: null,
    feedback_text: null,
    time_created: "2026-08-13T10:00:00Z",
  },
  {
    id: 2,
    message: "## Customs duty\n\nUse the **CIF value** as the basis.",
    message_type: "assistant",
    documents: [
      {
        document_id: "source-1",
        semantic_identifier: "Customs Law · Article 15",
        link: "https://example.com/customs-law",
      },
      {
        document_id: "source-2",
        semantic_identifier: "Internal tariff schedule",
        link: null,
      },
    ],
    feedback_type: "like",
    feedback_text: "Clear and helpful.",
    time_created: "2026-08-13T10:00:01Z",
  },
];

describe("QueryHistoryTranscript", () => {
  it("renders a read-only conversation with chat-style user and assistant messages", () => {
    render(<QueryHistoryTranscript messages={messages} />);

    expect(
      screen.getByRole("region", { name: "Conversation history" })
    ).toBeInTheDocument();
    expect(screen.getByTestId("query-history-user-message")).toHaveTextContent(
      "How do I calculate customs duty?"
    );
    expect(
      screen.getByTestId("query-history-assistant-message")
    ).toHaveTextContent("Use the CIF value as the basis.");
    expect(
      screen.getByRole("heading", { name: "Customs duty" })
    ).toBeInTheDocument();
    expect(
      screen.getByText("Customs duty").closest("div.prose-onyx")
    ).toBeInTheDocument();
  });

  it("keeps sources compact until their document group is opened", () => {
    render(<QueryHistoryTranscript messages={messages} />);

    expect(
      screen.getByRole("button", { name: "Kaynakları göster" })
    ).toHaveTextContent("Kaynaklar · 2 sonuç · 2 belge");
    expect(
      screen.queryByRole("link", { name: "Customs Law · Article 15" })
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Kaynakları göster" }));
    expect(screen.getByText("Customs Law")).toBeInTheDocument();
    expect(
      screen.getByText("Internal tariff schedule")
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Customs Law kaynaklarını göster" })
    );
    const source = screen.getByRole("link", {
      name: "Article 15",
    });
    expect(source).toHaveAttribute("href", "https://example.com/customs-law");
    expect(source).toHaveAttribute("target", "_blank");
    expect(screen.queryByText("Article 15")).toBeInTheDocument();
  });

  it("groups duplicate sources and limits an expanded document to a small preview", () => {
    const sourceDocuments = Array.from({ length: 8 }, (_, index) => ({
      document_id: `source-${index}`,
      semantic_identifier: `Customs Law — Article ${index + 1}`,
      link: `https://example.com/customs-law#${index + 1}`,
    }));
    const manySources: MessageSnapshot[] = [
      {
        ...messages[1]!,
        documents: [...sourceDocuments, sourceDocuments[0]!],
      },
    ];

    render(<QueryHistoryTranscript messages={manySources} />);

    fireEvent.click(screen.getByRole("button", { name: "Kaynakları göster" }));
    expect(
      screen.getByText("Kaynaklar · 8 sonuç · 1 belge")
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Customs Law kaynaklarını göster" })
    );
    expect(screen.getByText("Article 6")).toBeInTheDocument();
    expect(screen.queryByText("Article 7")).not.toBeInTheDocument();
    expect(screen.getByText("2 kaynak daha göster")).toBeInTheDocument();
  });

  it("shows feedback alongside the source control", () => {
    render(<QueryHistoryTranscript messages={messages} />);

    expect(screen.getByText("Like")).toBeInTheDocument();
    expect(screen.getByText("Clear and helpful.")).toBeInTheDocument();
  });
});
