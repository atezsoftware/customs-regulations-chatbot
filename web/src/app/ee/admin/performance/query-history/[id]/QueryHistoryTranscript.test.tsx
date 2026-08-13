import { render, screen } from "@testing-library/react";
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

  it("shows linked and unlinked sources alongside feedback for an assistant response", () => {
    render(<QueryHistoryTranscript messages={messages} />);

    const source = screen.getByRole("link", {
      name: "Customs Law · Article 15",
    });
    expect(source).toHaveAttribute("href", "https://example.com/customs-law");
    expect(source).toHaveAttribute("target", "_blank");
    expect(screen.getByText("Internal tariff schedule")).toBeInTheDocument();
    expect(screen.getByText("Like")).toBeInTheDocument();
    expect(screen.getByText("Clear and helpful.")).toBeInTheDocument();
  });
});
