import { withDateRange } from "@/app/ee/admin/performance/query-history/utils";

test("CSV export combines date range with every applied history filter", () => {
  const url = new URL(
    withDateRange(
      {
        from: new Date("2026-11-01T00:00:00Z"),
        to: new Date("2026-11-09T00:00:00Z"),
      },
      {
        user_query: "Çağrı",
        api_key_query: "Tariff",
        feedback_type: "like",
        session_rating: 4,
        q: "50%_",
      }
    ),
    "http://localhost"
  );
  expect(Object.fromEntries(url.searchParams)).toEqual({
    start: "2026-11-01T00:00:00.000Z",
    end: "2026-11-09T00:00:00.000Z",
    user_query: "Çağrı",
    api_key_query: "Tariff",
    feedback_type: "like",
    session_rating: "4",
    q: "50%_",
  });
});
