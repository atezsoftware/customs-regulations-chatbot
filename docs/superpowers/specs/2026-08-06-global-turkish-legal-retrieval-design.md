# Global Turkish Legal Retrieval and OpenRouter Reranking

**Date:** 2026-08-06

## Context

The installation has not indexed a corpus yet. The first indexing run can therefore
start with contextual indexing and Turkish analysis; no document backfill or index
conversion workflow is required.

This repository is treated as the owned product, not as an upstream-compatible Onyx
fork. Existing Community and licensed Enterprise code may be changed or reused when
that produces the cleanest architecture; preserving compatibility with removed
upstream settings is not a constraint.

The corpus is Turkish legal and regulatory material. Retrieval must preserve formal
legal terminology, article and paragraph identifiers, dates, institution names, and
cross-references. A complete answer may depend on several chunks from one document,
so source diversity must never become a one-chunk-per-source rule.

The behavior in this design is global for user-facing retrieval. It applies to normal
Chat, the Search API, the end-user Search UI, and Deep Search. Administrative document
inventory searches are intentionally excluded because they are not answer-retrieval
paths.

## Goals

- Make contextual indexing the required default before the first document is indexed.
- Retrieve a broad, multi-query candidate set suited to Turkish legal language.
- Rerank the fused candidates once through an administrator-selected OpenRouter
  reranker.
- Preserve multiple complementary chunks from the same document while reducing true
  duplicates and gently improving source coverage.
- Use the same core retrieval and ranking behavior in normal and Deep Search paths.
- Improve citation completeness with a bounded support audit, without unbounded
  per-claim searches.
- Fail safely when the external reranker is unavailable and expose enough telemetry
  to diagnose that fallback.

## Non-goals

- Enforcing a fixed number of chunks or sources per answer.
- Enforcing one result per document.
- Reindexing an existing corpus; there is no existing indexed corpus in scope.
- Sending administrative inventory-search results through the answer reranker.
- Testing OpenRouter with a real administrator API key in automated tests.

## Global Behavior and Boundaries

All user-facing search paths use a shared post-fusion ranking service. The normal Chat
and Search API paths already meet in `SearchTool`; Deep Search constructs and reuses
the same tool, so the shared stage naturally applies to both ordinary and research
agent searches. The separate end-user Search UI pipeline invokes the same ranking
service at the equivalent point in its flow.

The shared stage runs only after access control and post-query document censorship.
This prevents inaccessible chunk text from being sent to OpenRouter. The retrieval
and ranking stages keep chunk identity as `(document_id, chunk_id)` throughout; they
never deduplicate on `document_id` alone. The downstream citation mappings retain
that identity, and recovery evidence is extended to carry the same canonical tuple.

Caller-specific context budgets remain valid. For example, Deep Search may request a
different number of final chunks than normal Chat. Those budgets are applied after
the common broad retrieval and reranking stages, so a small answer context does not
also become a small reranking candidate pool.

## First-Index Contextual Configuration

For supported, non-multitenant deployments:

1. Contextual indexing is enabled by default in configuration, database defaults, and
   the admin UI. An idempotent first-index bootstrap also updates an already-created
   `SearchSettings` row when the installation still has no indexed documents; changing
   a column default alone is not considered sufficient.
2. An administrator selects the contextualization LLM before starting indexing.
3. Starting indexing without a usable contextualization LLM is blocked with a clear
   validation error. The system must not silently index ordinary chunks instead.
4. Document summaries and chunk summaries remain enabled by default.
5. Every eligible multi-chunk passage written during the first indexing run includes
   its generated document and local chunk context in the searchable representation.
   Single-chunk documents and chunks that cannot safely reserve contextual tokens keep
   their document summary but do not generate a redundant or truncating local context.
6. Context generation failure is strict for every eligible connector and regulatory
   corpus path: the indexing task fails instead of silently writing ordinary chunks.

The existing multitenant-cloud restriction on contextual indexing remains intact. A
single effective-settings function enforces `false` for that deployment before both
ordinary connector and user-file indexing; merely changing the global environment
default must not bypass the restriction.

Because the corpus is empty, the bootstrap/settings update, analyzer configuration,
and strict first-index gate are sufficient. The work includes a normal schema
migration for new reranker settings, but no corpus migration or backfill job.

## Turkish Legal Retrieval

### Index analysis

The new Elasticsearch index uses Turkish text analysis rather than the current
English default. Existing exact title fields remain, `heading_path` gains an exact
subfield, and indexing extracts normalized provision identifiers, decision numbers,
and legally significant dates into dedicated keyword fields. Query construction adds
boosted exact clauses for those fields alongside analyzed body-text clauses. Legal
references such as `m. 5`, `Madde 5`, `Geçici Madde 2`, decision numbers, dates, and
institution names must not be discarded by stop-word processing.

### Query generation

Each `SearchTool` call produces at most five bounded retrieval lanes in total, not
five lanes per seed query:

1. the original query, unchanged;
2. one semantic reformulation in Turkish;
3. up to three Turkish legal/lexical variants when useful.

The expansion prompt keeps the source language, formal legal register, exact quoted
phrases, provision identifiers, dates, and named institutions. It may add common
legal forms and abbreviations, but it must not translate the query to English or
replace a precise legal term with colloquial wording.

The original query is always searched. The pipeline stops applying the hard-coded
English stop-word remover to Turkish hybrid-search keywords; it either uses the
original query or a conservative Turkish-aware form. Empty or near-duplicate query
variants are removed before retrieval.

Query expansion is enabled by default for the normal SearchTool and the separate
end-user Search UI/API pipeline. Administrative inventory search keeps its current
single-query behavior and remains outside this feature.

The canonical query passed to the reranker is the original query for that `SearchTool`
call, before reformulation. Semantic, model-written, keyword, and original variants
all count toward the single five-lane cap.

Deep Search continues to create focused research questions. Its current secondary
expansion bypass and the corresponding "without secondary term expansion" contract
are deliberately replaced: every focused `SearchTool` call may use the same bounded
five-lane Turkish expansion, broad candidate collection, and reranking behavior.
Existing limits on the number of focused Deep Search tool calls remain in place.

## Candidate Collection and Fusion

Each query lane retrieves keyword and semantic candidates. Exact duplicate chunks
are removed by `(document_id, chunk_id)`, then the lanes are fused with reciprocal
rank fusion. Four separate named budgets prevent early truncation:

- a per-lane retrieval budget;
- a fused rerank pool, defaulting to the best 100 accessible chunks;
- the caller's returned-result budget for Search API/UI;
- the caller's final LLM-context budget for Chat or Deep Search.

The current `num_hits` truncation moves after the fused rerank pool is built. The
SearchTool and separate end-user Search UI pipeline both pass these budgets
explicitly; a final context budget of 8, 12, or 25 never reduces the pool evaluated by
the reranker.

The current regulatory provision-family and referenced-provision expansion stays
after primary ranking. It can therefore add the exact neighboring or referenced text
needed to interpret a selected provision without forcing unrelated adjacent chunks
into every result.

## OpenRouter Reranker Administration

### Persistence

`SearchSettings` remains the canonical home for global retrieval behavior. A new
migration adds typed reranking fields for:

- enabled state;
- provider type, initially `openrouter`;
- API key encrypted at rest through the licensed AES encryption implementation;
- selected reranker model identifier;
- timestamps and normal ownership metadata used by other admin settings.

These are newly defined product fields with current validation and API semantics, even
where their purpose resembles columns removed by an older upstream migration. The
configuration does not reuse an ordinary chat LLM provider row. Startup/configuration
fails clearly if the encryption key required to store a reranker credential is not
available; plaintext or base64-only storage is not an accepted fallback.

### Admin API

Admin-only endpoints provide:

- `GET /admin/reranking/config` — returns configuration with only a masked key state;
- `PUT /admin/reranking/config` — creates or updates configuration; an omitted key
  retains the stored key;
- `DELETE /admin/reranking/config` — disables and removes the stored credential;
- `POST /admin/reranking/test` — accepts optional unsaved key/model overrides and sends
  a small fixed sample through the selected model without persisting those overrides;
- `POST /admin/reranking/openrouter-models` — accepts an optional, unsaved API key in
  the request body and lists models from
  `/api/v1/models?output_modalities=rerank`; the key never appears in a URL.

The OpenRouter base URL is fixed server-side. User-supplied arbitrary URLs are not
accepted. Keys are never returned, logged, or embedded in frontend state after save.
If the filtered catalog is unavailable or omits a valid reranker, the form permits a
manually entered model identifier that must pass the test endpoint before enablement.

### Admin UI

Index Settings gains a clearly named **Retrieval Optimization / Reranking** section.
The administrator can enter an OpenRouter API key, load reranking-capable models,
select or manually enter a model, test the configuration, enable it, and save it. The
UI explains that both the user's query and candidate document text are sent to the
selected external provider for ranking. Existing stale reranking fields in frontend
index-setting types are removed or reconciled with this single configuration schema.

Contextual indexing configuration remains separate: the administrator first selects
the contextualization LLM, and indexing cannot start until that mandatory selection
is valid. Reranking activates once its own valid OpenRouter configuration is saved;
the absence of a reranker key must not prevent contextual indexing.

## Shared Reranking and Soft Diversity

After reciprocal-rank fusion, a dedicated OpenRouter adapter sends one bounded request
to `/api/v1/rerank` containing the canonical Turkish query and accessible candidate
chunks. It does not reuse the dormant LiteLLM reranker abstraction, whose provider
enum, timeout, `top_n`, and index-preserving response shape do not meet this contract.

OpenRouter documents are strings (or documented text-bearing objects), so title,
canonical source, contextual summary, and chunk text are serialized into one labeled
text value rather than sent as undocumented metadata. In addition to the 100-chunk
count limit, centrally named per-document and total serialized-size/token limits bound
every request. Deterministic truncation removes repeated summary prose first and
preserves the chunk body, provision identifiers, decision numbers, headings, and
dates. Candidates that cannot fit are retained in the local tail rather than lost.

The server requests `top_n` equal to the number of candidates actually submitted.
Valid returned items are ordered by reranker score; any valid but omitted candidate is
appended in its fused order. Thus a successful partial provider response never shrinks
the caller's requested result cardinality.

The response is validated strictly:

- every returned index must be unique and within the submitted candidate range;
- every relevance score must be finite;
- duplicate indices or response items missing their required index/score fields are
  treated as provider failure; returning fewer items is valid only under the
  deterministic omitted-tail rule above;
- ties preserve a deterministic order based on the pre-rerank fused rank.

On a successful OpenRouter result, that ordering replaces the existing SearchTool LLM
relevance selector so a second model cannot undo the ranking. When OpenRouter is
disabled or fails, the existing selector remains the relevance fallback where it is
currently available. Soft diversity is applied after either path and is therefore the
last ordering operation before caller-specific truncation.

Selection uses reranker relevance as the primary signal and the fused rank as a small
stability signal. A deterministic greedy soft-diversity pass then applies:

- a strong penalty only to chunks that are textual near-duplicates of an already
  selected chunk;
- a small, decaying bonus to a competitive chunk from a source not yet represented;
- no hard source quota, no per-document cap, and no penalty merely because two chunks
  share a document.

For diversity, `source` means the canonical document origin: normalized external
source URL when available, otherwise `document_id`. It never means an entire connector
type or URL domain. The unseen-source bonus is considered only inside a narrow
relevance band, so it cannot promote a materially weaker document over complementary
chunks from the same regulation. Near-duplicate detection is based on normalized
chunk-text overlap and is independent of source identity.

Consequently, three complementary sections from one regulation can all outrank a weak
passage from another source. Diversity only breaks close relevance decisions and
reduces redundant copies. The algorithm's weights and thresholds are centrally named
constants and covered with regression tests, including the required same-document,
multi-chunk cases.

## Failure Handling and Observability

If reranking is disabled, times out, receives a rate-limit or server error, or returns
an invalid payload, retrieval fails open to the fused local ordering plus the same
local duplicate/diversity pass. The answer request continues; it does not discard the
candidate set.

The adapter uses a short bounded timeout and a tenant-scoped circuit breaker. Invalid
credentials, payment/model errors, and repeated transient provider failures enter a
cooldown so one Deep Search run cannot repeatedly issue paid or predictably failing
requests. Runtime configuration lookups and circuit state are tenant-scoped. Saving
or deleting a configuration invalidates them immediately, and deletion prevents any
previously decrypted key from being reused.

Every rerank attempt records structured, secret-free telemetry under the existing
rerank flow category:

- selected model and candidate/result counts;
- latency and outcome (`success`, `disabled`, `timeout`, `rate_limited`,
  `provider_error`, or `invalid_response`);
- whether fallback ordering was used.

No API key or document text is written to logs. Configuration validation errors are
shown explicitly in the admin UI instead of being silently treated as runtime
fallback.

## Citation Completeness

Answer prompts for Turkish legal material instruct the model to cite the exact chunks
that directly support each material claim, use formal Turkish, and retain article and
decision identifiers. Deep Search's final synthesis and correction prompts receive
the same explicit language requirement. The prompts do not require citing every
retrieved source or inventing source diversity.

A bounded post-draft support audit checks whether material claims lack direct support
or whether cited chunks fail to contain the asserted rule. It activates through the
existing focused-regulatory/legal retrieval mode, not a new language heuristic. For
normal Chat, the legal draft is buffered server-side and audited before any answer
tokens are emitted; published streaming text is never retracted or silently replaced.

Both normal Chat and Deep Search replace their iterative audit-recovery agent loop for
this mode with at most one direct `SearchTool` invocation. This cap covers actual
support-audit searches, not the ordinary pre-draft research calls. The reviewer emits
a stable claim identifier/span. Recoverable issues are prioritized deterministically:
unsupported legal rule, obligation, prohibition, exception, deadline, or conclusion;
then other material facts; then earliest occurrence in the draft. Both uncited claims
and cited-but-unsupported claims are eligible. The narrow current exact-substring and
"no related citation" recovery restrictions are replaced by this structured mapping.

Recovery evidence carries `(document_id, chunk_id)` and preserves both
`citation_mapping` and `citation_chunk_mapping`. After the one focused search, the
buffered answer/final report is regenerated, or the unsupported claim is qualified or
removed. There is no unbounded search for every sentence.

Search API and Search UI result lists do not generate prose, so they receive the
retrieval/reranking improvements but not the post-draft audit.

## End-to-End Data Flow

```text
Turkish legal question
  -> original + bounded Turkish legal query variants
  -> keyword and semantic retrieval per lane
  -> ACL/post-query censorship
  -> exact-chunk deduplication
  -> reciprocal-rank fusion (broad pool, default 100)
  -> bounded OpenRouter payload + one rerank call, or deterministic local fallback
  -> soft near-duplicate suppression + soft source bonus
  -> caller-specific final context budget
  -> provision/reference expansion where applicable
  -> buffered draft with exact chunk citations
  -> support audit + at most one recovery search when needed
  -> emit the final answer
```

## Security and Privacy

- The reranking APIs require the same administrator authorization used by other model
  and index settings.
- The API key is encrypted at rest with the repository's licensed AES implementation
  and is always masked on reads. Plaintext or encoding-only fallback is forbidden.
- Only already authorized, post-censorship chunks may leave the system.
- The admin UI discloses external query and candidate-text processing before
  enablement.
- Requests require OpenRouter zero-data-retention and deny-data-collection routing
  preferences. A selected route that cannot honor the configured privacy policy is
  rejected during configuration testing rather than silently downgraded.
- The OpenRouter origin is server-controlled to avoid SSRF.
- Provider errors and traces exclude credentials and raw chunk bodies.

## Testing Strategy

### Backend

- Unit-test OpenRouter request construction, response validation, stable ordering,
  payload truncation, omitted-tail behavior, timeouts, circuit breaking, rate limits,
  malformed responses, and fail-open behavior with mocked HTTP.
- Test exact-chunk deduplication and prove that several complementary chunks from one
  document survive reranking and final selection.
- Test that only true/near duplicates are penalized and that the source bonus never
  acts as a hard quota.
- Test Turkish query expansion, preservation of legal identifiers, and removal of the
  English stop-word dependency from the Turkish path.
- Test the five-lane cap per `SearchTool` call, Deep Search's deliberate expansion
  behavior, separate retrieval/rerank/context budgets, and absence of pre-rerank
  truncation.
- Integration-test admin authorization, API-key masking and retain-on-omission update
  behavior, tenant cache invalidation, model/manual-ID selection, privacy policy, and
  first-index contextual-model validation/bootstrap.
- Exercise the shared stage through normal Search/Chat, end-user Search UI processing,
  and the Deep Search `SearchTool` path.
- Test pre-stream buffering, the single actual support-gap `SearchTool` call, canonical
  evidence identity, cited-but-unsupported recovery, deterministic issue priority,
  and no-op behavior when citations already support the answer.

### Frontend

- Component/service tests cover loading, saving, masking, model loading, test status,
  invalid configuration errors, and the external-processing disclosure.
- A mocked Playwright flow configures the contextual model before first indexing and
  saves/tests an OpenRouter reranker, including an unsaved-key model lookup, without a
  real external credential.

### Verification Limits

Automated tests use a mock OpenRouter server/client. A real reranker request cannot be
claimed as verified until an administrator supplies an API key and selects a model.

## Acceptance Criteria

- The first indexing run cannot start without contextual indexing and a selected
  contextualization model on supported deployments.
- New eligible multi-chunk Turkish legal passages use contextual content and Turkish
  analysis from the outset; any contextual generation failure stops indexing.
- All user-facing retrieval surfaces rank a broad fused pool through the same service.
- Saving a valid OpenRouter reranker configuration enables one post-fusion rerank call
  per retrieval batch; disabling or failing it preserves usable local results.
- Multiple relevant chunks from the same document remain eligible and are observable
  in regression tests.
- Deep Search receives the same contextual, Turkish-aware, reranked retrieval behavior.
- Generated regulatory answers are audited before streaming, use exact chunk
  citations, and perform no more than one actual support-gap `SearchTool` call.
- Secrets and inaccessible document text are not exposed through APIs, logs, or
  reranker requests.
