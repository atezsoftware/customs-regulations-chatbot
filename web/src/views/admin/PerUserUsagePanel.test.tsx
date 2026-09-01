/**
 * @jest-environment jsdom
 */

import { render, screen } from "@tests/setup/test-utils";
import PerUserUsagePanel from "./PerUserUsagePanel";
import { useUsageExport } from "@/lib/usage/userUsage";

jest.mock("@/lib/usage/userUsage", () => ({
  ...jest.requireActual("@/lib/usage/userUsage"),
  useUsageExport: jest.fn(),
  resetUserUsage: jest.fn(),
}));

const mockUseUsageExport = jest.mocked(useUsageExport);

describe("PerUserUsagePanel", () => {
  it("shows per-user query, session, token, and cost rates", () => {
    mockUseUsageExport.mockReturnValue({
      usage: {
        start: "2026-06-01",
        end: "2026-06-14",
        users: [
          {
            email: "alice@example.com",
            totals: {
              input_tokens: 600,
              output_tokens: 180,
              cache_read_tokens: 5,
              cost_cents: 6,
              total_tokens: 780,
              total_user_queries: 3,
              total_user_sessions: 2,
              average_tokens_per_query: 260,
              average_tokens_per_session: 390,
              average_cost_cents_per_query: 2,
              average_cost_cents_per_session: 3,
              average_queries_per_session: 1.5,
            },
          },
        ],
      },
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    });

    render(
      <PerUserUsagePanel
        timeRange={{
          from: new Date(2026, 5, 1),
          to: new Date(2026, 5, 14),
          selectValue: "custom",
        }}
      />
    );

    expect(screen.getByText("Queries")).toBeInTheDocument();
    expect(screen.getByText("Sessions")).toBeInTheDocument();
    expect(screen.getByText("Total tokens")).toBeInTheDocument();
    expect(screen.getByText("Tokens / query")).toBeInTheDocument();
    expect(screen.getByText("Tokens / session")).toBeInTheDocument();
    expect(screen.getByText("Cost / query")).toBeInTheDocument();
    expect(screen.getByText("Cost / session")).toBeInTheDocument();
    expect(screen.getByText("Queries / session")).toBeInTheDocument();

    const row = screen.getByTestId("usage-row-alice@example.com");
    expect(row).toHaveTextContent("780");
    expect(row).toHaveTextContent("260");
    expect(row).toHaveTextContent("390");
    expect(row).toHaveTextContent("$0.0200");
    expect(row).toHaveTextContent("$0.0300");
    expect(row).toHaveTextContent("1.5");
  });
});
