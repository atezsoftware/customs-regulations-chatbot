# Experimental Guardrails v3 Design

## Intent

Add an isolated `Experimental Guardrails v3` workflow on top of ASv3 Single
Research. It must reduce silent omission of requested legal outcomes and recover
useful sources lost at the reranking boundary without changing any existing
button, request contract, workflow policy, prompt, retrieval limit, checkpoint,
or publication behavior.

Success means that v3 can:

- preserve every material requested outcome and alternative scenario;
- distinguish missing evidence from evidence that was found but excluded,
  delivered but omitted, or applied incorrectly;
- run one combined Method 1 + Method 2 JEV review stage;
- use the cheapest valid repair route before starting a new search;
- perform at most one bounded repair cycle and a targeted recheck; and
- expose enough telemetry to measure accuracy, cost, latency, false alarms, and
  repair regressions.

The user explicitly selected a new experimental button as the isolation and
rollout boundary. Existing ASv3, ASv3 Single Research, Experimental Research,
Experimental Parallel Research, Experimental Guardrails, Experimental
Guardrails v2, Legal Composite, and Supersearch workflows remain unchanged.

## Chosen Approach

Use Method 1 as the state and provenance foundation and Method 2 as the final
professional review checklist. They share one frozen packet, one logical JEV
review stage, one finding normalizer, and one repair controller. They do not run
as two independent reviewers.

The alternatives were rejected for these reasons:

- Method 2 alone can identify a missing outcome but cannot determine whether a
  useful passage was discarded by reranking. It therefore falls back to a new
  search too often.
- Method 1 alone preserves outcomes and evidence but does not consistently
  assess court decisions, penalties, relief, operational steps, legal
  applicability, and whole-question coverage.
- A second general-model reviewer would add latency and cost before local JEV
  quality and repair rates are measured.

## Workflow Identity and Isolation

Add a new workflow selection and checkpoint policy:

- request field: `asv3_guardrails_v3`;
- workflow variant: `asv3_guardrails_v3`;
- workflow policy: `issue-legal-review-v1`;
- frontend selection: `experimental_guardrails_v3`; and
- button label: `Experimental Guardrails v3`.

The workflow requires:

- `atez_search_v3=true`;
- `asv3_research_profile="normal"`;
- `asv3_parallel_research=false`;
- one selected generation model; and
- neither the v1 nor v2 guardrails request flag.

All guardrails variants are mutually exclusive. A resumed run uses the variant
and policy stored in its checkpoint even if the current UI selection differs.
The v3 run has a 30-minute overall deadline and reserves time for final review,
repair, and recheck. No deadline, budget, or retry setting changes for another
variant.

## Data Model

### Requested outcomes and source-backed conditions

Reuse the existing `OutcomeMap` as the persistent Method 1 record. For every
substantive v3 request, the first useful coordinator action must record stable
requested outcomes covering all original questions and alternative scenarios.
The coordinator must not make a separate planning-model call.

Existing records remain authoritative:

- `RequestedOutcome` binds an outcome to original question IDs and exact
  decisive facts;
- `OutcomeCondition` binds a material rule or condition to original evidence
  witness spans; and
- `OutcomeResolution` records supported, conditional, or precisely unresolved
  treatment.

The v3 review packet derives separate evidence, answer, and resolution states
without changing the shared `OutcomeMap` checkpoint schema. This avoids a
migration or behavioral change for existing workflows:

- evidence state: unavailable, candidate excluded, delivered incomplete,
  adequate delivered, or conflicting;
- answer state: unanswered, omitted, partial, contradicted, conditional, or
  addressed; and
- resolution state: supported, conditional, unresolved disclosed, or open.

An absent outcome map is a review defect for a substantive v3 request, never an
implicit pass. Greetings, arithmetic, clarification responses, and requests
that do not require research remain exempt.

### Retrieval candidate audit

Add a v3-only candidate audit sidecar at the search/rerank boundary. The search
pipeline already knows the candidate set before final LLM delivery, so the
audit must be emitted there rather than reconstructed from selected results.

Each audit record contains bounded metadata, not unrestricted passage text:

- search run and query IDs;
- stable document/source and chunk identity;
- retrieval lane and mode;
- raw score, normalized score, and rerank position when available;
- selected, excluded, unmapped, hydration-failed, or delivered status;
- deterministic reason for the status;
- scope/version metadata needed for an authorized later read; and
- linked outcome IDs when the search action supplied them.

The evidence ledger continues to own original text, citation numbers, hashes,
and delivery receipts. Excluded candidates are not citable until they are
revalidated, hydrated through the authorized `CorpusBroker`, added to the
evidence ledger, and delivered to the repair model. The audit is exported in
the v3 checkpoint with strict count and byte bounds. Other variants neither
produce nor consume it.

## Frozen Review Packet

Create a versioned, immutable v3 review packet after the ASv3 candidate answer
is complete and before publication. It contains:

- the complete original request and scenario branches;
- the `OutcomeMap` view, including unassessed outcomes and source witnesses;
- cited originals and all originals required by retained outcome conditions;
- relevant delivery receipts and candidate-audit summaries;
- the candidate answer split into stable section/assertion IDs;
- citation, source, draft, prompt, checklist, and packet hashes; and
- remaining time and repair budget metadata.

Evidence allocation is issue-aware rather than a global first-40 prefix:

1. include every cited original;
2. include every original required by an outcome-condition witness;
3. allocate remaining evidence round-robin across outcomes; and
4. include compact candidate-audit records for unresolved outcomes.

Required evidence is never silently truncated. The packet builder has a strict
state-character budget chosen below the provider token limit. If the required
packet cannot fit, v3 records `review_packet_too_large` and retains the original
answer; it does not drop a requested outcome and call that answer reviewed.
The first implementation uses one JEV request for the initial review. Packet
partitioning is out of scope until measurements show it is necessary.

## Combined JEV Review

Use the existing TypeSafe `/v1/systemone` transport boundary but implement a
separate v3 typed adapter so v2 behavior and metrics remain unchanged. The
request uses independent questions because questions in the same batch cannot
depend on other answers.

The review has two layers in one request.

### Method 1 checks

For each requested outcome:

- inventory coverage;
- evidence adequacy;
- answer coverage;
- fact application;
- contradiction probability; and
- condition/exception preservation.

### Method 2 checks

Assess six versioned dimensions:

- D1 court decisions;
- D2 penalties and liability;
- D3 exemptions and relief;
- D4 operational steps;
- D5 legal basis and applicability; and
- D6 question coverage and uncertainty.

Applicability is evaluated before treatment in application logic. The paired
questions remain independent: treatment instructions say to assess treatment
assuming the dimension is relevant. Code then combines applicability and
treatment. `not_applicable`, `not_searched`, `unknown`, and
`review_unavailable` are distinct states.

D1 and D2 do not trigger research merely to fill a checklist row. D3 through
D6 bind to the affected requested outcomes. Duplicate Method 1 and Method 2
signals produce one normalized finding.

Prefer choice labels for categorical state:

- evidence: adequate, missing, incomplete, conflicting, uncertain;
- coverage: addressed, omitted, partial, uncertain;
- application: correct, missing_fact, misapplied, uncertain; and
- applicability: relevant, not_applicable, uncertain.

Use bounded probabilities only for contradiction, inventory omission, and the
final material-repair signal. Validate exact answer keys and types. Persist the
requested alias, actual returned model identity, raw typed answers, token
usage, duration, checklist version, and packet hash.

## Finding Normalization and Repair Routing

Convert JEV results into application-owned findings. Each material finding
contains issue IDs, dimensions, evidence references, affected answer units,
reason origin, action, success condition, draft hash, review question IDs, and
repair attempt number. JEV does not invent evidence spans or tool actions.

Deduplicate findings that describe the same defect. Route one bounded action
using this order:

1. **Patch from delivered evidence.** If sufficient original evidence was
   already delivered, patch only the affected answer units.
2. **Recover an excluded candidate.** Revalidate and hydrate the best matching
   authorized audited candidate, deliver it, then patch.
3. **Read a known original.** Use a source-scoped direct read when the finding
   identifies the governing source or provision.
4. **Run one focused search.** Only when the earlier routes cannot establish the
   missing issue, run one query bound to the outcome and exact evidence target.
5. **Disclose the narrow gap.** If evidence, time, or a decisive user fact is
   unavailable, preserve supported sections and add a precise conditional or
   unresolved statement.

One v3 answer may execute at most one recovery/search action and one Gemini
repair call. The Gemini model remains `gemini-3.8-flash`, uses low reasoning,
one provider attempt, no streaming, bounded output, and the remaining overall
deadline. Repair instructions treat all evidence as untrusted data and forbid
new citation numbers.

The repair is accepted only if:

- it changes the intended answer units and preserves unrelated supported text;
- every citation exists in the post-recovery evidence ledger;
- every cited passage was actually delivered to the repair call;
- original citations are not all removed without a finding that requires it;
- the final draft hash is new and internally consistent; and
- the targeted JEV recheck clears the repaired findings and affected dependent
  conclusions.

The recheck evaluates only changed findings and dependencies, not the complete
packet. It is still part of the same logical review stage and is limited to one
request. A failed, malformed, timed-out, or contradictory recheck rejects the
repair and retains the original answer.

## Failure and Publication Behavior

The workflow is experimental and must avoid turning a review-provider problem
into `Response was terminated prior to completion`.

- Missing TypeSafe credentials, review timeout, malformed JEV output, or an
  oversized packet preserves the original ASv3 answer and records a precise
  `review_unavailable` reason.
- Missing Gemini configuration or a failed repair preserves the original
  answer and records `repair_unavailable` or `repair_failed`.
- A recovered candidate that fails authorization, identity, version, or
  hydration validation is not delivered and cannot become a citation.
- Exhausted research time does not start a repair that cannot be rechecked.
- Existing deterministic citation and source-access validation always runs on
  the final selected answer.

Fail-open transport behavior is not recorded as a review pass. Checkpoint and
telemetry distinguish clean review, repair requested, recovery attempted,
repair applied, recheck passed, and every failure reason.

## Cost and Latency Bounds

The pass path adds one bounded JEV request and no Gemini or search call. Outcome
tracking is attached to existing coordinator decisions and adds no mandatory
planning call.

The repair path adds at most:

- one candidate recovery, direct read, or focused search;
- one Gemini patch; and
- one targeted JEV recheck.

The controller reserves review/repair time before optional retrieval and uses
the v3 30-minute deadline for all provider retries and tool work. It records
review, recovery, repair, recheck, and total durations separately, plus input
and output tokens and estimated/provider cost where available.

The document estimates of approximately `$0.00084` for a 20k-token JEV pass,
`+$0.026633` mean cost and `+18.4 seconds` mean latency at a 20% repair rate are
planning assumptions, not acceptance thresholds. New-search repairs have much
higher historical latency and therefore remain the last route.

## Frontend and Request Flow

Add one independent input selector beside the existing experimental options.
Selecting v3 deselects every other research mode through the existing single
workflow state. It is disabled during multi-model chat, project-scoped chat,
or any condition that already disables ASv3 Single Research.

New sends, resends, and regenerations propagate only the v3 request flag.
Checkpoint resume is authoritative and must not inherit whichever experimental
button happens to be selected later. Existing payloads remain byte-for-byte
equivalent when v3 is not selected.

## Checkpoint and Telemetry

The v3 checkpoint stores bounded, versioned records for:

- workflow variant and policy;
- outcome map;
- evidence ledger and delivery receipts;
- candidate audit summaries;
- frozen packet hash and checklist version;
- normalized findings and chosen action;
- review/recovery/repair/recheck token and timing metrics; and
- final status and failure reason.

Resume rejects incompatible v3 policies or altered request/scope hashes.
Existing checkpoints do not gain v3 fields.

## Testing

### Backend unit and integration boundaries

- v3 request validation, mutual exclusion, normal/non-parallel selection, and
  checkpoint-authoritative resume;
- no v3 behavior in every existing workflow variant;
- mandatory substantive outcome coverage with exempt non-research responses;
- candidate audit capture across retrieved, filtered, reranked, selected,
  unmapped, hydration-failed, and delivered states;
- audit bounds, authorization-preserving recovery, checkpoint round trip, and
  no citation before hydration/delivery;
- frozen packet completeness, issue-aware evidence allocation, stable hashes,
  and oversize fail-open behavior;
- exact typed JEV question/answer validation, independent applicability and
  treatment questions, timeout/malformed/missing-key behavior;
- duplicate finding consolidation and the four repair routes in priority order;
- one recovery/search, one Gemini repair, and one targeted recheck maximum;
- citation/delivery validation, unrelated-section preservation, rejected
  repair fallback, and final checkpoint metrics; and
- Case 02 regression: repair treatment cannot close the separate original-tax
  refund and procedure outcomes.

### Frontend

- independent v3 selector and mutual exclusivity;
- exact request payload for new send, resend, and regeneration;
- multi-model and project isolation; and
- checkpoint resume that does not relabel or reroute historical answers.

### Evaluation before broader use

Run the stored v34/v37 answers and deliberately damaged variants through v3.
Record issue recall, justified repair rate, false passes, false alarms,
unjustified `not_applicable`, candidate recovery rate, repair success,
post-repair regression, p50/p95 stage latency, tokens, and cost per accepted
correct answer. The explicit experimental button is the initial small-cohort
boundary; no existing workflow is promoted to v3 automatically.

## Commit Structure

Keep implementation reviewable with scoped commits:

1. design specification;
2. v3 frontend/backend workflow identity and routing;
3. v3-only candidate audit and checkpoint support;
4. frozen packet, typed combined JEV review, and finding normalization;
5. bounded recovery, Gemini repair, and targeted recheck;
6. integration tests, telemetry, configuration documentation, and final
   verification fixes.

Each commit must keep the repository buildable and must not alter another
workflow's expected test results.

## Non-goals

- No change to an existing button or workflow policy.
- No second general-model reviewer.
- No separate issue-generation model call.
- No unbounded search, repair, retry, or context partition loop.
- No database migration in the first implementation; v3 state remains within
  the existing ASv3 checkpoint boundary.
- No automatic promotion of v3 behavior to production workflows.
