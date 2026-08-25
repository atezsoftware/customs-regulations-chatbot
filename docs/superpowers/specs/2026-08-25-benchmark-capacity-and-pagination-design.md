# Benchmark Capacity and Pagination Design

## Problem

Benchmark runs currently reject more than 50 questions even when fewer models
keep the run's total work reasonable. The API also returns every item of every
historical run from `GET /api/regulatory/benchmark/runs`; the admin page polls
that payload every three seconds while a run is active. Increasing the run
limit without changing this read path would amplify API and browser memory use.

## Capacity contract

- Accept at most 100 selected questions.
- Accept at most 6 candidate models.
- Accept at most 300 run items, where run items are the unique candidate model
  count multiplied by the selected question count.
- Therefore both 100 questions x 3 models and 50 questions x 6 models are
  valid, while 100 questions x 6 models is rejected before rows are created.
- Keep regulatory benchmark worker concurrency and per-item child-process
  isolation unchanged. Total run size must not increase simultaneous model
  execution.
- Return the configured limits to the web client so UI validation and backend
  validation cannot silently drift.

## Read API

`GET /api/regulatory/benchmark/runs` returns lightweight run summaries only.
Summaries contain run identity, status, progress counts, timing, failure data,
and cost totals needed by the history list, but no item payloads, judgments,
execution steps, cited sources, or run report.

`GET /api/regulatory/benchmark/runs/{run_id}` returns run-level detail and
aggregates without embedding all items. Item results move to
`GET /api/regulatory/benchmark/runs/{run_id}/items?offset=<n>&limit=<n>`. The
item endpoint uses a stable item-id order, accepts a page size from 1 through
50, defaults to 20, and returns `items`, `total`, `offset`, and `limit`.

Create, start, cancel, and retry operations return the run-level detail shape.
They do not serialize the full item graph.

## Web behavior

The benchmark admin page loads run summaries for history and polling. It loads
run detail plus the first item page only for the selected run. Selecting another
run or page requests only that run/page. Active polling refreshes summaries,
the selected run detail, and its visible item page; it never downloads every
historical item.

The create panel shows the total run-item count and configured maximum. It
disables creation when question count, candidate count, or their cross-product
exceeds the server-provided limits, with a concise explanation of the violated
limit.

## Error handling and compatibility

The backend remains authoritative and rejects invalid combinations with an
`INVALID_INPUT` error that includes the selected and configured item counts.
Existing run data and database schema are unchanged. No migration is required.
The admin web client changes in the same deployment as the API contract.

## Tests

- Backend model/API tests prove 100 x 3 and 50 x 6 are accepted and 100 x 6 is
  rejected without creating a run.
- DB/API tests prove run listing does not load or serialize item details and
  item pagination is stable and bounded.
- Frontend tests prove limits are displayed, an oversized combination is
  blocked, and polling requests only summaries plus the selected item page.
- Existing benchmark recovery, execution, retry, cancellation, and scoring
  tests remain green.
