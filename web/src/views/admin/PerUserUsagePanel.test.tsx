/**
 * @jest-environment jsdom
 */

import { render, screen, setupUser, within } from "@tests/setup/test-utils";
import PerUserUsagePanel from "@/views/admin/PerUserUsagePanel";
import { useUsageExport } from "@/lib/usage/userUsage";
import type { UsageExportUser, WorkflowUsage } from "@/lib/usage/interfaces";

jest.mock("@/lib/usage/userUsage", () => ({
  ...jest.requireActual("@/lib/usage/userUsage"),
  useUsageExport: jest.fn(),
  resetUserUsage: jest.fn(),
}));

const mockUseUsageExport = jest.mocked(useUsageExport);

const timeRange = {
  from: new Date(2026, 5, 1),
  to: new Date(2026, 5, 14),
  selectValue: "custom",
};

const alice: UsageExportUser = {
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
};

const deepUsage: WorkflowUsage = {
  session_count: 2,
  query_count: 4,
  cost_cents: 80,
  known_cost_cents: 80,
  average_cost_cents_per_session: 40,
  average_cost_cents_per_query: 20,
  cost_status: "complete",
  unpriced_call_count: 0,
};

function mockUsers(users: UsageExportUser[]) {
  mockUseUsageExport.mockReturnValue({
    usage: { start: "2026-06-01", end: "2026-06-14", users },
    isLoading: false,
    error: undefined,
    refetch: jest.fn(),
  });
}

describe("PerUserUsagePanel", () => {
  it("preserves every existing metric and supports responses without workflow data", () => {
    mockUsers([alice]);
    render(<PerUserUsagePanel timeRange={timeRange} />);

    for (const label of [
      "Input",
      "Output",
      "Cache",
      "Total tokens",
      "Queries",
      "Sessions",
      "Tokens / query",
      "Tokens / session",
      "Cost / query",
      "Cost / session",
      "Queries / session",
      "Cost",
    ]) {
      expect(
        screen.getByRole("button", {
          name: `Sort All workflows ${label}`,
        })
      ).toBeInTheDocument();
    }

    const row = screen.getByRole("row", { name: /alice@example.com/ });
    const cells = within(row).getAllByRole("cell");
    expect(cells.slice(1, 13).map((cell) => cell.textContent)).toEqual([
      "600",
      "180",
      "5",
      "780",
      "3",
      "2",
      "260",
      "390",
      "$0.0200",
      "$0.0300",
      "1.5",
      "$0.06",
    ]);
    expect(cells.slice(13, 23).map((cell) => cell.textContent)).toEqual(
      Array(10).fill("—")
    );
    expect(
      within(row).getByRole("button", { name: "Reset" })
    ).toBeInTheDocument();
  });

  it("shows independent deep and fast workflow counts, costs, and averages", () => {
    mockUsers([
      {
        ...alice,
        workflow_usage: {
          asv3: deepUsage,
          asv3_exp_cropped: {
            ...deepUsage,
            session_count: 1,
            query_count: 3,
            cost_cents: 12,
            known_cost_cents: 12,
            average_cost_cents_per_session: 12,
            average_cost_cents_per_query: 4,
          },
        },
      },
    ]);
    render(<PerUserUsagePanel timeRange={timeRange} />);

    expect(
      screen.getByRole("columnheader", { name: "Derin Mod · ASv3 Tek" })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: "Hızlı Mod · ASv3 Exp Cropped" })
    ).toBeInTheDocument();
    const row = screen.getByRole("row", { name: /alice@example.com/ });
    expect(
      within(row)
        .getAllByRole("cell")
        .slice(13, 23)
        .map((cell) => cell.textContent)
    ).toEqual([
      "2",
      "$0.80",
      "$0.4000",
      "4",
      "$0.2000",
      "1",
      "$0.12",
      "$0.1200",
      "3",
      "$0.0400",
    ]);
  });

  it("shows a partial captured lower bound without presenting it as a complete average", () => {
    mockUsers([
      {
        ...alice,
        workflow_usage: {
          asv3: {
            ...deepUsage,
            cost_cents: null,
            cost_status: "partial",
            average_cost_cents_per_query: null,
            average_cost_cents_per_session: null,
            unpriced_call_count: 1,
          },
          asv3_exp_cropped: {
            ...deepUsage,
            cost_cents: null,
            known_cost_cents: 0,
            cost_status: "unavailable",
            average_cost_cents_per_query: null,
            average_cost_cents_per_session: null,
          },
        },
      },
    ]);
    render(<PerUserUsagePanel timeRange={timeRange} />);

    const row = screen.getByRole("row", { name: /alice@example.com/ });
    expect(within(row).getByText("≥$0.80")).toBeInTheDocument();
    expect(within(row).getByText("Partial")).toBeInTheDocument();
    expect(within(row).getByText("Unavailable")).toBeInTheDocument();
    expect(within(row).getAllByRole("cell")[15]).toHaveTextContent("—");
    expect(within(row).getAllByRole("cell")[17]).toHaveTextContent("—");
  });

  it("sorts by workflow usage in both directions and keeps missing metrics last", async () => {
    mockUsers([
      { ...alice, email: "legacy@example.com" },
      { ...alice, workflow_usage: { asv3: deepUsage } },
      {
        ...alice,
        email: "bob@example.com",
        workflow_usage: { asv3: { ...deepUsage, session_count: 5 } },
      },
    ]);
    render(<PerUserUsagePanel timeRange={timeRange} />);
    const user = setupUser();
    const sort = screen.getByRole("button", {
      name: "Sort Derin Mod · ASv3 Tek Sessions",
    });
    const emails = () =>
      screen
        .getAllByRole("link")
        .map((link) => link.getAttribute("aria-label"));

    await user.click(sort);
    expect(emails()).toEqual([
      "Query history for bob@example.com",
      "Query history for alice@example.com",
      "Query history for legacy@example.com",
    ]);
    await user.click(sort);
    expect(emails()).toEqual([
      "Query history for alice@example.com",
      "Query history for bob@example.com",
      "Query history for legacy@example.com",
    ]);
  });

  it("opens all query history for the exact user without filtering out older sessions", () => {
    mockUsers([alice]);
    render(<PerUserUsagePanel timeRange={timeRange} />);

    const link = screen.getByRole("link", {
      name: "Query history for alice@example.com",
    });
    const url = new URL(link.getAttribute("href")!, "http://localhost:3000");
    expect(url.pathname).toBe("/admin/performance/query-history");
    expect(url.searchParams.get("user_email")).toBe("alice@example.com");
    expect(url.searchParams.has("start_time")).toBe(false);
    expect(url.searchParams.has("end_time")).toBe(false);
  });
});
