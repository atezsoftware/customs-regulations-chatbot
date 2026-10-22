"use client";

import useSWR from "swr";
import { errorHandlingFetcher } from "@/lib/fetcher";
import { SWR_KEYS } from "@/lib/swr-keys";
import { buildApiPath } from "@/lib/urlBuilder";
import type {
  UsageExportDateRange,
  UsageExportResponse,
} from "@/lib/usage/interfaces";
import {
  convertDateToEndOfDay,
  convertDateToStartOfDay,
} from "@/components/dateRangeSelectors/dateUtils";

export type {
  UsageExportResponse,
  UsageExportTotals,
  UsageExportUser,
} from "@/lib/usage/interfaces";

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
