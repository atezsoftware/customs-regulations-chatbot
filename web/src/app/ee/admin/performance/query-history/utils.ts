import type { DateRange } from "@/components/dateRangeSelectors/AdminDateRangeSelector";
import { START_QUERY_HISTORY_EXPORT_URL } from "@/app/ee/admin/performance/query-history/constants";
import type { QueryHistoryFilters } from "@/app/ee/admin/performance/query-history/filters";

export const withRequestId = (url: string, requestId: string): string =>
  `${url}?request_id=${encodeURIComponent(requestId)}`;

export function withDateRange(
  dateRange: DateRange,
  filters: QueryHistoryFilters = {}
): string {
  const { start_time, end_time, ...selection } = filters;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(selection)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const start = dateRange?.from.toISOString() ?? start_time;
  const end = dateRange?.to.toISOString() ?? end_time;
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  const query = params.toString();
  return query
    ? `${START_QUERY_HISTORY_EXPORT_URL}?${query}`
    : START_QUERY_HISTORY_EXPORT_URL;
}
