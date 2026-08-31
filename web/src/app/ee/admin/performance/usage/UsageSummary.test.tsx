import { render, screen } from "@tests/setup/test-utils";
import { UsageSummaryCards } from "./UsageSummary";

describe("UsageSummaryCards", () => {
  test("shows query and session totals with token averages", () => {
    render(
      <UsageSummaryCards
        summary={{
          total_user_queries: 1250,
          total_user_sessions: 500,
          total_query_tokens: 45000,
          average_tokens_per_query: 36,
          average_tokens_per_session: 90,
          average_queries_per_session: 2.5,
        }}
      />
    );

    expect(screen.getByText("1,250")).toBeInTheDocument();
    expect(screen.getByText("500")).toBeInTheDocument();
    expect(screen.getByText("45,000")).toBeInTheDocument();
    expect(screen.getByText("36 tokens / query")).toBeInTheDocument();
    expect(screen.getByText("90 tokens / session")).toBeInTheDocument();
    expect(screen.getByText("2.5 queries / session")).toBeInTheDocument();
  });
});
