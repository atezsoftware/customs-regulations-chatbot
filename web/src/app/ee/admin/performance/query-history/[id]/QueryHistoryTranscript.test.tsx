import {
  fireEvent,
  render as renderComponent,
  screen,
  within,
} from "@testing-library/react";
import { setupUser } from "@tests/setup/test-utils";
import { ValidSources } from "@/lib/types";
import { Provider as TooltipProvider } from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";
import type { MessageSnapshot } from "@/app/ee/admin/performance/usage/types";
import QueryHistoryTranscript from "@/app/ee/admin/performance/query-history/[id]/QueryHistoryTranscript";

function render(component: ReactNode) {
  return renderComponent(<TooltipProvider>{component}</TooltipProvider>);
}

let mockIsAdmin = false;

jest.mock("@/providers/UserProvider", () => ({
  useUser: () => ({ isAdmin: mockIsAdmin }),
}));

const messages: MessageSnapshot[] = [
  {
    id: 1,
    message: "How do I calculate customs duty?",
    message_type: "user",
    model_display_name: null,
    documents: [],
    feedback_type: null,
    feedback_text: null,
    time_created: "2026-08-13T10:00:00Z",
  },
  {
    id: 2,
    message: "## Customs duty\n\nUse the **CIF value** as the basis.",
    message_type: "assistant",
    model_display_name: "GPT-5",
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
  it("opens every same-file citation's retained text and keeps unresolved references visible", async () => {
    const user = setupUser();
    const fetchSpy = jest.spyOn(global, "fetch");
    const answer: MessageSnapshot = {
      ...messages[1]!,
      message: "Rule [[4]]() and exception [[10]](). Missing [[99]]().",
      documents: [],
      unresolved_citations: [99],
      citation_info: [4, 10].map((citation_number) => ({
        type: "citation_info",
        citation_number,
        document_id: "same-file",
        chunk_ind: citation_number,
        chunk_id: `rc-${citation_number}`,
        display_title: "Official source title",
        reference: `MADDE ${citation_number === 4 ? 241 : 242}`,
        source_type: ValidSources.File,
        text: `\n Passage ${citation_number} original text \n`,
        resolution_status: "retained",
      })),
    };
    render(<QueryHistoryTranscript messages={[answer]} />);
    const references = screen.getByLabelText("Yanıtta kullanılan atıflar");
    expect(references).toHaveTextContent("Yanıtta kullanılan atıflar · 3");
    expect(within(references).getByText("MADDE 241")).toBeVisible();
    expect(within(references).getByText("MADDE 242")).toBeVisible();
    expect(references).toHaveTextContent("[99] · Kaynak çözümlenemedi");
    const buttons = within(references).getAllByRole("button", {
      name: "Official source title",
    });
    await user.click(buttons[1]!);
    const passage = await screen.findByText("Passage 10 original text");
    expect(passage.textContent).toBe("\n Passage 10 original text \n");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("keeps more than six public web citations reachable through their original links", async () => {
    const user = setupUser();
    const openSpy = jest.spyOn(window, "open").mockImplementation(() => null);
    const answer: MessageSnapshot = {
      ...messages[1]!,
      documents: [],
      citation_info: Array.from({ length: 8 }, (_, index) => ({
        type: "citation_info",
        citation_number: index + 1,
        document_id: `url-${index}`,
        display_title: `Public source ${index + 1}`,
        link: `https://official.example/${index + 1}`,
        source_type: ValidSources.Web,
        resolution_status: "external",
      })),
    };
    render(<QueryHistoryTranscript messages={[answer]} />);
    const references = screen.getByLabelText("Yanıtta kullanılan atıflar");
    expect(within(references).getAllByRole("button")).toHaveLength(8);
    await user.click(
      within(references).getByRole("button", { name: "Public source 8" })
    );
    expect(openSpy).toHaveBeenCalledWith(
      "https://official.example/8",
      "_blank",
      "noopener,noreferrer"
    );
  });

  afterEach(() => {
    mockIsAdmin = false;
    jest.restoreAllMocks();
  });

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
    expect(screen.getByText("Internal tariff schedule")).toBeInTheDocument();

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

  it("offers admins a graph link for each assistant answer", () => {
    mockIsAdmin = true;
    const open = jest.spyOn(window, "open").mockImplementation(() => null);
    render(<QueryHistoryTranscript messages={messages} />);

    fireEvent.click(screen.getByRole("button", { name: "Execution graph" }));
    expect(open).toHaveBeenCalledWith(
      "/admin/answer-graphs/message/2",
      "_blank",
      "noopener,noreferrer"
    );
  });
});
