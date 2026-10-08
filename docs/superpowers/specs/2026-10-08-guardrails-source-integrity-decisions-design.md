# Guardrails Source Integrity and Decisions Design

## Problem

`Experimental Guardrails` is an isolated ASv3 workflow, but it currently uses
the standard retrieval limits. A reranked independent source can therefore be
excluded from the small delivery window before its original is hydrated and
recorded. The final answer cannot cite or display an original that never enters
the evidence ledger.

The workflow must preserve independent supporting sources without changing the
existing ASv3, Experimental, or Experimental ASv3 routes. The extra work must
remain bounded in latency and cost.

## Chosen Design

1. Add a retrieval-policy resolver that returns a dedicated high-recall,
   diversity-preserving override only for `asv3_guarded_experimental`.
   Its values are lower than the legacy tuned policy: 192 candidates per lane,
   256 rerank candidates, 256 regulatory candidates, and 32 delivered LLM
   chunks. This makes source diversity available without using the tuned
   workflow's 384/50 budget.
2. Add a source-integrity delivery rule in the search adapter. It records a
   deterministic receipt containing candidate, selected, hydrated, and dropped
   source counts. A hydrated independent source selected through diversity is
   retained as evidence. A malformed/unmapped source remains a partial result;
   it is never silently represented as corpus absence.
3. Add an optional OpenAI Decisions-based advisory reranker for Guardrails
   only. It is off unless `ASV3_GUARDED_DECISIONS_ENABLED=true` and an OpenAI
   default key are present. It examines only a bounded shortlist around the
   deterministic cutoff, uses one API request with predicate questions, a
   1.5-second timeout, no retries, and leaves the deterministic ordering intact
   on any refusal, timeout, malformed response, or provider error.
4. The advisory can only promote already retrieved candidates into the
   Guardrails delivery window; it cannot remove deterministic top results or
   invent citations. It is therefore a recall guard, not an authority or answer
   generator.
5. Call the documented Decisions endpoint through the existing `httpx` runtime
   dependency. The pinned LiteLLM version excludes OpenAI SDK 3.x, so a direct
   request avoids a broad model-stack upgrade while retaining the endpoint
   contract. The call is tagged with a dedicated tracing flow and never logs
   the API key or passage text.

## Isolation and Performance Constraints

- Existing workflow variants receive the exact prior retrieval overrides and
  never invoke Decisions.
- The default Guardrails path makes no extra provider call.
- Decisions evaluates at most 12 boundary candidates and may promote at most 4.
- A failure adds no retry and no delay beyond the 1.5-second local timeout.
- Search source identity, citation numbering, and evidence-ledger publication
  remain host controlled.

## Tests

- Workflow-policy tests prove only Guardrails gets the dedicated override.
- Retrieval tests prove a late independent source is hydrated and retained by
  Guardrails diversity while legacy variants retain their current behavior.
- Decisions unit tests cover disabled mode, a bounded successful promotion,
  timeout/error fallback, malformed/refusal fallback, and the no-demotion
  invariant.
- The existing ASv3 workflow, native adapter, and search-adapter suites guard
  citation publication and legacy routing.
