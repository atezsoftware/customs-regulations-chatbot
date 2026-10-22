"use client";

import { useEffect, useMemo, useState } from "react";
import { Button, Card, InputTypeIn, MessageCard, Text } from "@opal/components";
import { SvgChevronLeft, SvgChevronRight, SvgX } from "@opal/icons";
import { PageLoader, toast } from "@opal/layouts";
import { cn } from "@opal/utils";
import { resetUserUsage, useUsageExport } from "@/lib/usage/userUsage";
import type {
  UsageExportTotals,
  UsageExportUser,
  WorkflowUsage,
  WorkflowUsageKey,
} from "@/lib/usage/interfaces";
import { DateRangePickerValue } from "@/components/dateRangeSelectors/AdminDateRangeSelector";
import { ADMIN_ROUTES } from "@/lib/admin-routes";
import { buildApiPath } from "@/lib/urlBuilder";

const PAGE_SIZE = 10;

type MetricSortKey = keyof UsageExportTotals;
type WorkflowMetricKey =
  | "session_count"
  | "cost_cents"
  | "average_cost_cents_per_session"
  | "query_count"
  | "average_cost_cents_per_query";
type SortKey =
  | "email"
  | MetricSortKey
  | `${WorkflowUsageKey}.${WorkflowMetricKey}`;
type SortDir = "asc" | "desc";

function formatTokens(n: number): string {
  return n.toLocaleString();
}

function formatCost(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

function formatAverageCost(cents: number): string {
  return `$${(cents / 100).toFixed(4)}`;
}

function formatAverage(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

const METRIC_COLUMNS: {
  label: string;
  sortKey: MetricSortKey;
  format: (value: number) => string;
  muted?: boolean;
}[] = [
  {
    label: "Input",
    sortKey: "input_tokens",
    format: formatTokens,
    muted: true,
  },
  {
    label: "Output",
    sortKey: "output_tokens",
    format: formatTokens,
    muted: true,
  },
  {
    label: "Cache",
    sortKey: "cache_read_tokens",
    format: formatTokens,
    muted: true,
  },
  { label: "Total tokens", sortKey: "total_tokens", format: formatTokens },
  { label: "Queries", sortKey: "total_user_queries", format: formatTokens },
  {
    label: "Sessions",
    sortKey: "total_user_sessions",
    format: formatTokens,
  },
  {
    label: "Tokens / query",
    sortKey: "average_tokens_per_query",
    format: formatAverage,
    muted: true,
  },
  {
    label: "Tokens / session",
    sortKey: "average_tokens_per_session",
    format: formatAverage,
    muted: true,
  },
  {
    label: "Cost / query",
    sortKey: "average_cost_cents_per_query",
    format: formatAverageCost,
    muted: true,
  },
  {
    label: "Cost / session",
    sortKey: "average_cost_cents_per_session",
    format: formatAverageCost,
    muted: true,
  },
  {
    label: "Queries / session",
    sortKey: "average_queries_per_session",
    format: formatAverage,
    muted: true,
  },
  { label: "Cost", sortKey: "cost_cents", format: formatCost },
];

const WORKFLOWS: { key: WorkflowUsageKey; label: string }[] = [
  { key: "asv3", label: "Derin Mod · ASv3 Tek" },
  { key: "asv3_exp_cropped", label: "Hızlı Mod · ASv3 Exp Cropped" },
];

const WORKFLOW_COLUMNS: {
  label: string;
  sortKey: WorkflowMetricKey;
  format: (value: number) => string;
}[] = [
  { label: "Sessions", sortKey: "session_count", format: formatTokens },
  { label: "Cost", sortKey: "cost_cents", format: formatCost },
  {
    label: "Cost / session",
    sortKey: "average_cost_cents_per_session",
    format: formatAverageCost,
  },
  { label: "Queries", sortKey: "query_count", format: formatTokens },
  {
    label: "Cost / query",
    sortKey: "average_cost_cents_per_query",
    format: formatAverageCost,
  },
];

function workflowCostDescription(usage: WorkflowUsage): string | undefined {
  if (usage.cost_status === "complete") return undefined;
  const reason = {
    partial: "Some research calls have missing or unpriced cost data.",
    unavailable: "Research cost data was not captured for these queries.",
    running: "Some queries are still running; cost capture is incomplete.",
  }[usage.cost_status];
  return `${reason} Captured cost: ${formatCost(usage.known_cost_cents)}. Averages are unavailable until all query costs are complete.`;
}

function sortValue(
  user: UsageExportUser,
  key: SortKey
): number | string | null {
  if (key === "email") return user.email.toLowerCase();
  if (!key.includes(".")) return user.totals[key as MetricSortKey];
  const [workflowKey, metricKey] = key.split(".") as [
    WorkflowUsageKey,
    WorkflowMetricKey,
  ];
  const usage = user.workflow_usage?.[workflowKey];
  if (!usage) return null;
  if (metricKey === "cost_cents" && usage.cost_status !== "unavailable") {
    return usage.cost_cents ?? usage.known_cost_cents;
  }
  return usage[metricKey];
}

interface UsageRowProps {
  user: UsageExportUser;
  onReset: () => void;
}

function UsageRow({ user, onReset }: UsageRowProps) {
  const [resetting, setResetting] = useState(false);
  const totals = user.totals;

  async function handleReset() {
    setResetting(true);
    try {
      await resetUserUsage(user.email);
      toast.success(`Reset usage for ${user.email}.`);
      onReset();
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown error";
      toast.error(`Failed to reset usage: ${message}`);
    } finally {
      setResetting(false);
    }
  }

  return (
    <tr data-testid={`usage-row-${user.email}`}>
      <td className="sticky left-0 z-10 min-w-64 max-w-64 border-b border-r border-border-01 bg-background-neutral-01 px-3 py-2">
        <div className="flex w-56 flex-col items-start gap-1">
          <div className="w-full truncate" title={user.email}>
            <Text font="main-ui-body">{user.email}</Text>
          </div>
          <Button
            href={buildApiPath(ADMIN_ROUTES.QUERY_HISTORY.path, {
              user_email: user.email,
            })}
            prominence="tertiary"
            size="sm"
            aria-label={`Query history for ${user.email}`}
          >
            Query history
          </Button>
        </div>
      </td>
      {METRIC_COLUMNS.map((column) => (
        <td
          className="min-w-28 border-b border-border-01 px-3 py-2 text-right"
          key={column.sortKey}
        >
          <Text
            font="main-ui-body"
            color={column.muted ? "text-03" : undefined}
          >
            {column.format(totals[column.sortKey])}
          </Text>
        </td>
      ))}
      {WORKFLOWS.map((workflow) => {
        const usage = user.workflow_usage?.[workflow.key];
        return WORKFLOW_COLUMNS.map((column, index) => {
          const value = usage?.[column.sortKey];
          const incompleteCost =
            column.sortKey === "cost_cents" &&
            usage &&
            usage.cost_status !== "complete";
          const displayValue =
            incompleteCost && usage.cost_status !== "unavailable"
              ? `≥${formatCost(usage.known_cost_cents)}`
              : value == null
                ? "—"
                : column.format(value);
          return (
            <td
              key={`${workflow.key}.${column.sortKey}`}
              className={cn(
                "min-w-28 border-b border-border-01 px-3 py-2 text-right",
                index === 0 && "border-l"
              )}
              title={usage ? workflowCostDescription(usage) : undefined}
            >
              <div className="flex flex-col items-end gap-1">
                <Text font="main-ui-body">{displayValue}</Text>
                {incompleteCost && (
                  <Text font="secondary-body" color="text-03">
                    {usage.cost_status === "running"
                      ? "In progress"
                      : usage.cost_status === "partial"
                        ? "Partial"
                        : "Unavailable"}
                  </Text>
                )}
              </div>
            </td>
          );
        });
      })}
      <td className="border-b border-border-01 px-3 py-2">
        <Button
          variant="default"
          prominence="tertiary"
          size="sm"
          disabled={resetting}
          onClick={handleReset}
        >
          Reset
        </Button>
      </td>
    </tr>
  );
}

interface SortHeaderProps {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  dir: SortDir;
  onSort: (key: SortKey) => void;
  align: "left" | "right";
  groupLabel: string;
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  dir,
  onSort,
  align,
  groupLabel,
}: SortHeaderProps) {
  const active = activeKey === sortKey;
  const indicator = active ? (dir === "desc" ? " ↓" : " ↑") : "";
  return (
    <th
      scope="col"
      aria-sort={
        active ? (dir === "desc" ? "descending" : "ascending") : "none"
      }
      className="min-w-28 border-b border-border-01 px-2 py-2"
    >
      <div className={cn("flex", align === "right" && "justify-end")}>
        <Button
          onClick={() => onSort(sortKey)}
          prominence={active ? "secondary" : "tertiary"}
          size="sm"
          aria-label={`Sort ${groupLabel} ${label}`}
        >
          {`${label}${indicator}`}
        </Button>
      </div>
    </th>
  );
}

interface PerUserUsagePanelProps {
  timeRange: DateRangePickerValue;
}

/** Searchable, sortable admin per-user usage totals. */
export default function PerUserUsagePanel({
  timeRange,
}: PerUserUsagePanelProps) {
  const { usage, isLoading, error, refetch } = useUsageExport(timeRange);
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("cost_cents");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const users = usage?.users ?? [];

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q
      ? users.filter((u) => u.email.toLowerCase().includes(q))
      : users;
    return [...filtered].sort((a, b) => {
      const av = sortValue(a, sortKey);
      const bv = sortValue(b, sortKey);
      if (av === null) return bv === null ? 0 : 1;
      if (bv === null) return -1;
      const cmp =
        typeof av === "string" && typeof bv === "string"
          ? av.localeCompare(bv)
          : (av as number) - (bv as number);
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [users, query, sortKey, sortDir]);

  const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));

  // Jump back to the first page whenever the filter or sort reshapes the list.
  useEffect(() => {
    setPage(0);
  }, [query, sortKey, sortDir]);

  // Clamp the page when the list shrinks (e.g. a reset drops a user off).
  useEffect(() => {
    if (page > pageCount - 1) setPage(pageCount - 1);
  }, [page, pageCount]);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      // Numeric columns lead high→low (leaderboard); email reads A→Z.
      setSortDir(key === "email" ? "asc" : "desc");
    }
  }

  if (isLoading) return <PageLoader />;
  if (error) {
    return (
      <MessageCard
        variant="error"
        icon={SvgX}
        title="Failed to load per-user usage."
      />
    );
  }

  const pageUsers = visible.slice(
    page * PAGE_SIZE,
    page * PAGE_SIZE + PAGE_SIZE
  );

  return (
    <Card border="solid" rounding="lg" padding="sm">
      <div className="flex flex-col gap-2">
        <Text font="heading-h3">Per-user usage</Text>
        <Text font="secondary-body" color="text-03">
          Query, session, token, and cost rates per user for the selected
          period. Total tokens are input + output; cache reads remain separate.
          Click a column to rank by it, or search by email. Reset clears usage
          from every currently active limit window.
        </Text>
        <Text
          font="secondary-body"
          color="text-03"
          title="Workflow estimates include nested research LLM calls, excluding embeddings, reranking, images, and voice. Cost / session and Cost / query divide the same captured cost by their counts."
        >
          Workflow columns include only queries with a recorded workflow;
          sessions used in both modes appear in both groups. Costs are captured
          research estimates and can differ from existing usage totals. Partial
          costs show a captured lower bound; — means complete cost data is
          unavailable.
        </Text>

        <InputTypeIn
          value={query}
          placeholder="Search users by email…"
          onChange={(e) => setQuery(e.target.value)}
        />

        {users.length === 0 ? (
          <Text font="main-ui-body" color="text-03">
            No usage recorded yet.
          </Text>
        ) : visible.length === 0 ? (
          <Text font="main-ui-body" color="text-03">
            {`No users match "${query}".`}
          </Text>
        ) : (
          <>
            <div
              className="overflow-x-auto"
              role="region"
              tabIndex={0}
              aria-label="Per-user usage table; scroll horizontally for workflow columns"
            >
              <table
                className="w-max min-w-full border-separate border-spacing-0"
                aria-label="Per-user usage"
              >
                <thead>
                  <tr>
                    <th
                      className="sticky left-0 z-20 border-b border-r border-border-01 bg-background-neutral-01 px-3 py-2"
                      scope="colgroup"
                    >
                      <Text font="main-ui-action">Account</Text>
                    </th>
                    <th
                      colSpan={METRIC_COLUMNS.length}
                      scope="colgroup"
                      className="border-b border-border-01 px-3 py-2 text-left"
                    >
                      <Text font="main-ui-action">All workflows</Text>
                    </th>
                    {WORKFLOWS.map((workflow) => (
                      <th
                        key={workflow.key}
                        colSpan={WORKFLOW_COLUMNS.length}
                        scope="colgroup"
                        className="border-b border-l border-border-01 px-3 py-2 text-left"
                      >
                        <Text font="main-ui-action">{workflow.label}</Text>
                      </th>
                    ))}
                    <th
                      scope="col"
                      className="border-b border-border-01 px-3 py-2"
                    >
                      <Text font="main-ui-action">Actions</Text>
                    </th>
                  </tr>
                  <tr>
                    <th
                      scope="col"
                      aria-sort={
                        sortKey === "email"
                          ? sortDir === "desc"
                            ? "descending"
                            : "ascending"
                          : "none"
                      }
                      className="sticky left-0 z-20 border-b border-r border-border-01 bg-background-neutral-01 px-3 py-2 text-left"
                    >
                      <Button
                        onClick={() => handleSort("email")}
                        prominence="tertiary"
                        size="sm"
                        aria-label="Sort by user email"
                      >
                        {`User${sortKey === "email" ? (sortDir === "desc" ? " ↓" : " ↑") : ""}`}
                      </Button>
                    </th>
                    {METRIC_COLUMNS.map((column) => (
                      <SortHeader
                        key={column.sortKey}
                        label={column.label}
                        sortKey={column.sortKey}
                        activeKey={sortKey}
                        dir={sortDir}
                        onSort={handleSort}
                        align="right"
                        groupLabel="All workflows"
                      />
                    ))}
                    {WORKFLOWS.map((workflow) =>
                      WORKFLOW_COLUMNS.map((column) => (
                        <SortHeader
                          key={`${workflow.key}.${column.sortKey}`}
                          label={column.label}
                          sortKey={`${workflow.key}.${column.sortKey}`}
                          activeKey={sortKey}
                          dir={sortDir}
                          onSort={handleSort}
                          align="right"
                          groupLabel={workflow.label}
                        />
                      ))
                    )}
                    <th
                      scope="col"
                      aria-label="Usage actions"
                      className="border-b border-border-01"
                    />
                  </tr>
                </thead>
                <tbody>
                  {pageUsers.map((user) => (
                    <UsageRow key={user.email} user={user} onReset={refetch} />
                  ))}
                </tbody>
              </table>
            </div>

            {pageCount > 1 && (
              <div className="flex flex-row items-center justify-end gap-3 pt-2">
                <Button
                  variant="default"
                  prominence="tertiary"
                  size="sm"
                  icon={SvgChevronLeft}
                  tooltip="Previous page"
                  aria-label="Previous page"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                />
                <Text font="main-ui-body" color="text-03">
                  {`Page ${page + 1} of ${pageCount} · ${visible.length} users`}
                </Text>
                <Button
                  variant="default"
                  prominence="tertiary"
                  size="sm"
                  icon={SvgChevronRight}
                  tooltip="Next page"
                  aria-label="Next page"
                  disabled={page >= pageCount - 1}
                  onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                />
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
}
