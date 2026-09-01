import { render, screen } from "@tests/setup/test-utils";
import { UsageSummaryCards } from "./UsageSummary";

describe("UsageSummaryCards", () => {
  test("shows ledger token and cost totals with query and session averages", () => {
    render(
      <UsageSummaryCards
        summary={{
          total_user_queries: 1250,
          total_user_sessions: 500,
          total_tokens: 4500000,
          total_cost_cents: 1234.56,
          average_tokens_per_query: 3600,
          average_tokens_per_session: 9000,
          average_cost_cents_per_query: 0.987648,
          average_cost_cents_per_session: 2.46912,
          average_queries_per_session: 2.5,
        }}
      />
    );

    expect(screen.getByText("1,250")).toBeInTheDocument();
    expect(screen.getByText("500")).toBeInTheDocument();
    expect(screen.getByText("4,500,000")).toBeInTheDocument();
    expect(screen.getByText("3,600 tokens / query")).toBeInTheDocument();
    expect(screen.getByText("9,000 tokens / session")).toBeInTheDocument();
    expect(screen.getByText("$12.35")).toBeInTheDocument();
    expect(screen.getByText("$0.0099 / query")).toBeInTheDocument();
    expect(screen.getByText("$0.0247 / session")).toBeInTheDocument();
    expect(screen.getByText("2.5 queries / session")).toBeInTheDocument();
  });
});
