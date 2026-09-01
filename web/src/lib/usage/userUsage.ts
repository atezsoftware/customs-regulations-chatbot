"use client";

import useSWR from "swr";
import { errorHandlingFetcher } from "@/lib/fetcher";
import { SWR_KEYS } from "@/lib/swr-keys";
import { buildApiPath } from "@/lib/urlBuilder";
import {
  convertDateToEndOfDay,
  convertDateToStartOfDay,
} from "@/components/dateRangeSelectors/dateUtils";

interface UsageExportDateRange {
  from: Date;
  to: Date;
}

export interface UsageExportTotals {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cost_cents: number;
  total_tokens: number;
  total_user_queries: number;
  total_user_sessions: number;
  average_tokens_per_query: number;
  average_tokens_per_session: number;
  average_cost_cents_per_query: number;
  average_cost_cents_per_session: number;
  average_queries_per_session: number;
}

export interface UsageExportUser {
  email: string;
  totals: UsageExportTotals;
}

export interface UsageExportResponse {
  start: string;
  end: string;
  users: UsageExportUser[];
}

export function buildUsageExportUrl(timeRange?: UsageExportDateRange): string {
  if (!timeRange) return SWR_KEYS.adminUsageExport;
  return buildApiPath(SWR_KEYS.adminUsageExport, {
    period_from: convertDateToStartOfDay(timeRange.from)?.toISOString(),
    period_to: convertDateToEndOfDay(timeRange.to)?.toISOString(),
  });
}

/** Company-wide per-user usage with a revalidation callback. */
export function useUsageExport(timeRange?: UsageExportDateRange) {
  const url = buildUsageExportUrl(timeRange);
  const { data, error, isLoading, mutate } = useSWR<UsageExportResponse>(
    url,
    errorHandlingFetcher,
    { revalidateOnFocus: false }
  );

  return { usage: data, isLoading, error, refetch: mutate };
}

/** Clears a user's usage across active enforcement windows. */
export async function resetUserUsage(userEmail: string): Promise<void> {
  const response = await fetch(SWR_KEYS.adminUsageReset, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_email: userEmail }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.detail || data?.error_code || response.statusText);
  }
}
