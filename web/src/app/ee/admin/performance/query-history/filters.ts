import type { Feedback } from "@/lib/types";
import type { SessionFeedbackRating } from "@/app/app/interfaces";

export interface QueryHistoryFilters {
  user_id?: string;
  user_email?: string;
  user_query?: string;
  api_key_query?: string;
  q?: string;
  feedback_type?: Feedback;
  session_rating?: SessionFeedbackRating;
  start_time?: string;
  end_time?: string;
}

const FILTER_KEYS = [
  "user_id",
  "user_email",
  "user_query",
  "api_key_query",
  "q",
  "feedback_type",
  "session_rating",
  "start_time",
  "end_time",
] as const;

export function readQueryHistoryFilters(
  params: Pick<URLSearchParams, "get">,
  allowIdentity = true
): QueryHistoryFilters {
  const filters: QueryHistoryFilters = {};
  for (const key of ["q", "start_time", "end_time"] as const) {
    const value = params.get(key)?.trim();
    if (value) filters[key] = value;
  }
  if (allowIdentity) {
    for (const key of [
      "user_id",
      "user_email",
      "user_query",
      "api_key_query",
    ] as const) {
      const value = params.get(key)?.trim();
      if (value) filters[key] = value;
    }
  }
  const feedback = params.get("feedback_type");
  if (feedback === "like" || feedback === "dislike" || feedback === "mixed") {
    filters.feedback_type = feedback;
  }
  const rating = Number(params.get("session_rating"));
  if (rating === 1 || rating === 2 || rating === 3 || rating === 4) {
    filters.session_rating = rating;
  }
  return filters;
}

export function readQueryHistoryPage(params: Pick<URLSearchParams, "get">) {
  const page = Number(params.get("page") || 1);
  return Number.isSafeInteger(page) && page > 0 ? page : 1;
}

export function queryHistorySearchParams(
  current: string,
  filters: QueryHistoryFilters,
  page = 1
) {
  const params = new URLSearchParams(current);
  FILTER_KEYS.forEach((key) => params.delete(key));
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  params.set("page", String(page));
  return params;
}

export function queryHistoryApiUrl(
  filters: QueryHistoryFilters,
  page: number,
  pageSize: number
) {
  const params = queryHistorySearchParams("", filters);
  params.delete("page");
  params.set("page_num", String(page - 1));
  params.set("page_size", String(pageSize));
  return `/api/admin/chat-session-history?${params.toString()}`;
}
