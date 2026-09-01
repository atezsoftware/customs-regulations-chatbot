"use client";

import { useEffect, useMemo, useState } from "react";
import { Button, Card, InputTypeIn, MessageCard, Text } from "@opal/components";
import { SvgChevronLeft, SvgChevronRight, SvgX } from "@opal/icons";
import { PageLoader, toast } from "@opal/layouts";
import {
  resetUserUsage,
  useUsageExport,
  UsageExportTotals,
  UsageExportUser,
} from "@/lib/usage/userUsage";
import { DateRangePickerValue } from "@/components/dateRangeSelectors/AdminDateRangeSelector";

const PAGE_SIZE = 10;

type MetricSortKey = keyof UsageExportTotals;
type SortKey = "email" | MetricSortKey;
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

function sortValue(user: UsageExportUser, key: SortKey): number | string {
  if (key === "email") return user.email.toLowerCase();
  return user.totals[key];
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
    <div
      className="flex flex-row items-center gap-4 py-2"
      data-testid={`usage-row-${user.email}`}
    >
      <div className="flex-1 truncate">
        <Text font="main-ui-body">{user.email}</Text>
      </div>
      {METRIC_COLUMNS.map((column) => (
        <div className="w-28 text-right" key={column.sortKey}>
          <Text
            font="main-ui-body"
            color={column.muted ? "text-03" : undefined}
          >
            {column.format(totals[column.sortKey])}
          </Text>
        </div>
      ))}
      <Button
        variant="default"
        prominence="tertiary"
        size="sm"
        disabled={resetting}
        onClick={handleReset}
      >
        Reset
      </Button>
    </div>
  );
}

interface SortHeaderProps {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  dir: SortDir;
  onSort: (key: SortKey) => void;
  align: "left" | "right";
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  dir,
  onSort,
  align,
}: SortHeaderProps) {
  const active = activeKey === sortKey;
  const indicator = active ? (dir === "desc" ? " ↓" : " ↑") : "";
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSort(sortKey)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSort(sortKey);
        }
      }}
      className={`cursor-pointer select-none ${
        align === "right" ? "w-28 text-right" : "flex-1"
      }`}
    >
      <Text font="main-ui-action" color={active ? "text-05" : "text-03"}>
        {`${label}${indicator}`}
      </Text>
    </div>
  );
}

/** Searchable, sortable admin per-user usage totals. */
export default function PerUserUsagePanel({
  timeRange,
}: {
  timeRange: DateRangePickerValue;
}) {
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
            <div className="overflow-x-auto">
              <div className="min-w-[1900px] flex flex-col divide-y divide-border-01">
                <div className="flex flex-row items-center gap-4 py-2">
                  <SortHeader
                    label="User"
                    sortKey="email"
                    activeKey={sortKey}
                    dir={sortDir}
                    onSort={handleSort}
                    align="left"
                  />
                  {METRIC_COLUMNS.map((column) => (
                    <SortHeader
                      key={column.sortKey}
                      label={column.label}
                      sortKey={column.sortKey}
                      activeKey={sortKey}
                      dir={sortDir}
                      onSort={handleSort}
                      align="right"
                    />
                  ))}
                  <div className="w-[68px]" />
                </div>
                {pageUsers.map((user) => (
                  <UsageRow key={user.email} user={user} onReset={refetch} />
                ))}
              </div>
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
