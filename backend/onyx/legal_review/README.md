# Legal Review

Legal Review is an opt-in workflow that combines question-derived legal issues with
twelve code-owned research and review dimensions. Its implementation and prompts live
in this package and `onyx/prompts/legal_review`. Existing Legal Composite, Supersearch,
ASv3 and Deep Research engines and their prompts remain independent.

## Request and source boundary

Use the normal authenticated chat endpoint through the frontend proxy:

```http
POST /api/chat/send-chat-message
Content-Type: application/json
```

```json
{
  "message": "Explain the application conditions, alternative procedures and relevant deadlines for this situation.",
  "legal_review": true
}
```

The normal chat-session fields can also be supplied. The mode requires the default
assistant outside projects and one model. It rejects other workflow flags, ASv3
resume/research options, attachments, additional context and forced tools. A saved
session or request model override does not replace its fixed reasoning model.

Sources are confined to the captured, authorized PC Külliyatı corpus. Search and
canonical reads use the same ACL, date and document-set scope. Source discovery uses
the public search adapter; identified provisions and material references use the
public `CorpusBroker` capabilities. No fixed topic-to-article, instrument or source-lane
catalog is installed. The workflow operates on existing indexed content; it does not
ingest, relabel, reindex or change publication metadata.

## Runtime

1. **Preflight:** select the `vertex_ai` override and require the exact model name
   `gemini-3.8-flash` for research and drafting. Resolve an accessible configured
   `openai` provider with visible `gpt-6-luna` and an official OpenAI base URL for
   the native Decisions reviewer before research begins. Existing group, persona
   and provider cost limits also apply; provider rows and model visibility are not
   changed. All available simple conversation history is retained. Earlier user
   messages can supply facts;
   earlier assistant legal statements are not independent legal evidence.
2. **Planner — Gemini:** identify requested outcomes, supplied facts and missing
   facts. Return stable issue IDs and a required nonempty plan-wide `discovery_queries`
   list with query text and covered `issue_ids`. Related issues share searches; no
   per-issue query is required. Code
   attaches the twelve dimensions; issue identification does not assume unseen law.
3. **Discovery and canonical acquisition:** run bounded, parallel hybrid searches,
   rank candidates and hydrate authorized originals into the shared evidence ledger.
   Search hits, labels and rank scores provide navigation, rather than legal proof.
4. **Reading — Gemini:** extract source-backed requirements and application
   conditions. Complete canonical originals appear as ordered, numbered passages;
   the model selects `citation` and `span_number` instead of rewriting quotations,
   hashes or offsets. Code resolves each selector to the exact original passage and
   validates canonical identity and integrity. Requirements are global source
   findings without an issue owner, application or dimension field. Assessments
   alone bind findings to issues and dimensions; their reasons explain the specific
   application. One finding can serve several issues and dimensions. The first
   accepted reading must explicitly assess all twelve dimensions for every issue;
   newly added issues also require their complete first assessment. Later calls
   send an explicitly present `dimensions` array containing only changed rows.
   Host-created unresolved placeholders never count as an accepted assessment.
   The model
   returns sparse assessment updates, and code maintains the complete twelve-row
   matrix per issue. Unassessed rows remain unresolved and unchanged rows are
   retained. Corrections use explicit supersession; obsolete findings remain in
   audit history, and affected assessment links become unresolved until reassessed.
5. **Source-derived issues — Gemini:** the reader actively resolves existing issues.
   It can open a linked child when a question discovered in the originals could
   materially change the parent's outcome and cannot be handled adequately inside
   that issue. This judgment applies across all twelve dimensions. An exception,
   tax consequence, procedural condition or judicial effect can be a dependency;
   merely citing a provision never automatically creates a child or an article search.
   A child needs a stable ID, parent, trigger dimension, finding IDs associated with
   the parent, a material reason and explicit closure criteria. Code derives exact
   trigger passages and citations and snapshots the parent assessments and findings.
   Accepted triggers retain that provenance after supersession. Code rejects cycles;
   distinct material questions can share a source and dimension. The model reuses
   an existing dependency for the same unresolved question and can use existing originals,
   request a canonical read or choose a focused search. One early source return can
   execute its requested operations and reread the originals before the early review.
   Missing evidence and unexecuted operations remain gaps; a zero-result search never
   proves absence of law or continued legal validity.
6. **Early review — OpenAI Decisions:** one native Decisions HTTP request using
   `gpt-6-luna` checks all issue/dimension combinations
   together, requested-outcome coverage and decisive source conditions. The same
   batch also checks each active finding against its selected complete originals
   for entailment, applicability conditions, exceptions and temporal effects.
   Finding IDs bind these flags to a specific interpretation that can be corrected.
   It receives
   complete retained originals, actual conversation facts and compact research
   receipts with query, status and access/truncation limitations. If the early source
   return has not been used, flags can direct that one return. Incomplete review
   prevents drafting a publishable answer.
7. **Integrated draft — Gemini:** produce a `GeneratedDraft` of ordered Markdown
   blocks with claim bindings, passage selectors, conditions and unresolved issues.
   Code adds missing selected source markers as separate paragraphs within their
   containing blocks, preserving Markdown fences and tables. It compiles the actual
   rendered block text into one literal answer and derives claim excerpts from it.
   Foreign model-written markers are retained for rejection by validation. The model
   does not recopy an answer or `answer_excerpt` field. Issues serve as a private
   checklist; blocks do not force
   repeated per-issue sections. Code validates passage selectors, literal claim
   bindings and citation targets, then inserts required scoped validity disclosure
   before review. A limitation-only answer can have no claims only with explicit
   unresolved issue IDs; it still undergoes complete independent review.
8. **Draft review — OpenAI Decisions:** independently check all dimensions, the
   entire answer's coverage, every material legal assertion (including assertions absent from the
   planned issues), source conditions and consistency across issues. The same
   request includes finding-bound checks and claim-bound checks, so the repair can
   locate the particular rule or literal block and its selected originals. These
   predicates add no per-source calls or extra review rounds. Reviewer scores are
   defect probabilities; scores at or above `0.5` flag the bound question. A flag is
   a suspicion to investigate, not a legal finding.
9. **Optional repair:** permit one post-draft evidence-directed repair. The reader
   sees the actual draft and flags, can obtain missing originals, and the writer
   produces a complete replacement. One Decisions recheck follows. Remaining flags
   or incomplete review withhold publication; they do not start another repair.
10. **Publication:** revalidate cited evidence against current authorization and
    publication state, persist its checkpoint, and then emit the standard answer and
    citation packets. Answer text, citation mapping, source operations, supersession
    history, usage, gaps and progress are retained. `completed` is emitted after the
    publication gate and answer stream. Saved progress is identified as
    `asv3_workflow_variant="legal_review"` and has no ASv3 resume control.

The dimensions are legal basis/hierarchy, validity/timing, case law/rulings,
exceptions/exemptions, penalties/reductions, tax/financial consequences, alternative
routes, procedure/deadlines, evidence/documents, operational steps, missing facts,
and liability/conflicting sources. A dimension can be addressed, affirmatively
not applicable, or unresolved; lack of evidence does not prove non-applicability.

## Canonical passage and finding contract

`canonical_evidence_view` preserves every retained original, its citation, source and
chunk identities, full text hash, citable state and metadata. It replaces the flat
`text` field with ordered `passages` containing `span_number` and exact text. Joining
those passage texts reconstructs the original byte for byte, including Unicode,
whitespace and paragraph boundaries. Passage boundaries are transport selectors;
they do not define a legal provision or limit the context the reader must consider.
Select multiple passages when a condition, exception or operative rule crosses a
boundary. No source-specific LLM call or retrieval reduction is introduced.

`PassageReference` contains only strict positive integer `citation` and `span_number`
fields. The model cannot submit a quotation, hash, witness ID or character offset.
`resolve_passage` derives the exact quotation, offsets and hash-bound witness identity
from the current ledger original. It rejects missing or unknown selectors, invalid
source/chunk targets, changed hashes, external/derived/untrusted/truncated originals,
and whitespace-only support. Canonical integrity establishes source binding; it does
not establish a finding's legal truth, relevance, applicability or in-force status.

The model records each global source finding once, then links it through
`DimensionAssessment.requirement_ids`. Findings have no `issue_id`, `application` or
`dimension`; each assessment's `reason` explains the application to that issue and
dimension. Code merges sparse updates into the full matrix, rejects duplicate or
unknown issue/dimension/finding identities, and leaves missing cells unresolved. The
same finding can serve multiple issues and dimensions without duplicating its rule
or evidence. Supersession is atomic and invalidates obsolete active assessment links.
The independent reviewer evaluates each relationship's relevance and entailment;
a valid ID link alone is insufficient.

Source-derived child triggers must reference a finding associated with their parent.
Code preserves the adopted finding and parent-assessment snapshots, with canonical
passage provenance derived from the ledger. These historical trigger snapshots do
not make a superseded finding active evidence for a current conclusion. Issue
closure and validity limitations follow each issue's current assessment links, so
an unused global finding does not affect unrelated issues.

Writer claims bind to their containing block and select the same canonical passages.
The compiler joins blocks in order, adds missing selected markers in a separate
paragraph after each block, and derives literal claim excerpts. This preserves closed
Markdown fences and table rows; it does not infer legal support.
The independent reviewer still examines the entire published draft, including claimless
headings or connective blocks that might contain an unrecorded legal assertion,
qualifying conditions, and contradictions across blocks and issues.

## Calls, retrieval width and cost accounting

There is one early review and one first-draft review, with at most one recheck after
the sole post-draft repair. The shortest path is three Gemini generations (planner,
reader, writer) and two Decisions requests. A model-requested early source return
adds a reader generation, making that path four Gemini generations and two Decisions
requests. A flagged early review can add a diagnostic reader call. Post-draft repair
adds its diagnostic reader, an optional reader after acquisition, one replacement
writer and one Decisions request. These are bounded paths, not a fixed per-answer bill.
An invalid initial structured plan permits one separately admitted Flash schema
correction. It preserves the request and uses the same remaining deadline and call
budget; provider failures do not trigger this correction.

Planner, reader, writer and repair use Gemini 3.8 Flash with low reasoning effort.
The fixed override has temperature zero. The independent reviewer uses the native
`https://api.openai.com/v1/decisions` endpoint with pinned `gpt-6-luna` and named
defect predicates. Predicate names map back to code-owned checks; the provider does
not generate queries or workflow actions. Each invocation makes one physical
request without internal retries or partitions. Missing credentials, refusals,
invalid responses, deadline failures and oversized packets prevent review completion.
There is no TypeSafe, OpenRouter, chat-completion or alternate-model fallback.
Checkpoints retain only the `openai_decisions` route and provider name, never keys.

Search retains the broad retrieval contract: 256 hits per lane, 384 rerank candidates
(including regulatory candidates), 50 model chunks and source diversity. Automatic
scope/time detection and query expansion are disabled on this workflow's own search
forks. Guarded advisory JEV reranking is disabled. Canonical originals are not
silently shortened to fit a generation or review packet.

Embedding and ranking inference still use the deployment's configured providers.
The existing ranking configuration can use a cross-encoder or an external chat
completion ranker; the latter is a separate model call, potentially a model other
than Gemini. Search's selected secondary LLM calls use the public `ScopedSearchLLM`
and are counted in the shared generation budget and Gemini usage meter. Embedding
and external reranker calls retain their own provider traces and usage accounting;
they are **not** included in the Gemini/Decisions token totals or the 32-admission counter.
No total dollar-price or live latency guarantee follows from the stage counts.

## Default limits and cancellation

The owned Legal Review search fork uses an inclusive `0.82` threshold on the
global min-max normalized external reranker score. Selection retains the baseline
candidate coverage and adds the complete qualifying score band. Other search
workflow classes retain their existing `0.90` default. The search trace records the
actual normalized threshold and selected identities.

| Limit | Default |
| --- | ---: |
| Overall cooperative deadline | 240 seconds |
| Finalization reserve | 110 seconds |
| Planner/reader/writer/repair or Decisions call timeout | 45 seconds |
| Shared Gemini/secondary-LLM/Decisions admissions | 32 |
| Admissions reserved for finalization | 8 |
| Source operations | 96 |
| Discovery queries across all issues and rounds | 24 |
| Parallel source operations | 4 |
| Initial reader rounds | 1 |
| Early source returns | 1 |
| Post-draft repairs | 1 |
| Input/output token admission thresholds | 2,000,000 / 128,000 |
| Gemini context ceiling | 192,000 tokens, or the smaller provider limit |
| Maximum generated output per Gemini call | 16,384 tokens |
| Evidence budget | 2,000,000 bytes |
| Serialized Decisions request/response cap | 256,000 bytes each |

Actual provider usage is recorded after each response, including a failed Decisions
validation with reported usage. Token thresholds are admission checks against usage
already observed; they are not an exact advance estimate of the next response's
bill. There is one provider attempt and one compatibility attempt for selected LLM
calls. Secondary search generations use the remaining research deadline; external
embedding/reranker providers retain their own timeout behavior. The reviewer provider's
token limits also apply: the byte cap alone does not guarantee a packet fits those
limits. Provider rejection produces an explicit incomplete review.

There is no numeric issue-count limit. Materiality, reuse of existing issues and
global time, call, search, tool and context budgets control growth. Every explicit
requested outcome must remain represented; the planner must not compress outcomes to
fit an artificial issue cap. Identical initial queries and identical source operations
within a reading batch are executed once with their issue IDs combined. A decisive
question that cannot be resolved within the remaining budget stays open and disclosed
instead of being forcibly closed.

Stop checks propagate through research, selected provider calls, Decisions response
handling and publication. Pending pooled operations are cancelled when possible;
an already running provider/read may finish after cancellation. Its result cannot
bypass the stop/deadline and publication fences. Unexecuted requested source
operations are preserved as gaps rather than silently considered complete.

## Legal validity and result status

Code derives each issue's `open`, `partial` or `closed` status from its complete
assessment matrix, supported resolution, pending operations and validity limits.
The model cannot declare closure directly. An open or partial material child prevents
its parent being closed. A newly discovered unresolved dependency is preserved in the
writer/reviewer state, result and checkpoint, and code inserts a scoped research
limitation into the literal draft before final review when needed. Validity and
supported resolution use the issue's linked global findings, not finding ownership.

Recorded version windows, read dates and a chunk lifecycle value such as `active`
do not prove legal in-force or annulment status. A recorded window is checked only
when an explicit captured `as_of_date` agrees with the original read's date. Otherwise
the requirement's temporal status is unknown. A legal status requires a separately
verified canonical status field; the model cannot assert it into existence.

The current shared evidence metadata whitelist does not transport verified legal
status fields. This workflow does not change that shared contract or fabricate a
sidecar. Ordinary corpus evidence therefore retains `legal_status="unknown"` even
after a targeted validity search. Any affected result is `partial`, and code adds a
scoped, user-visible validity limitation to the literal draft before its final Decisions
review. A clean reviewer score cannot promote unknown status to in-force status.

`verified` requires passed complete reviews and no remaining evidence, validity or
pending-operation gaps. `partial` requires passed complete reviews with accurately
disclosed remaining limits. `unavailable` has no publishable answer; `cancelled`
records a stopped run. These are workflow publication states, not a certification
that the corpus contains every potentially relevant legal source.

## Validation

Provider-free tests cover review failure, complete numbered canonical source views,
strict passage selectors, source identity/hash/trust checks, Unicode and whitespace
preservation, Markdown-safe host citation rendering, compiled literal claim bindings,
shared global findings across issues and dimensions, sparse assessment updates,
dimension closure, supersession, model-chosen dependencies across dimensions,
historical canonical child triggers, dependent parent closure, shared searches without
losing outcomes, cancellation, both citation display modes, persisted answer/citation
state, publication revocation and missing
Decisions credentials and provider group/persona access. Run them with:

```bash
uv run --no-sync pytest -q backend/tests/unit/onyx/legal_review \
  backend/tests/unit/onyx/chat/test_legal_review_workflow_routing.py
```

These tests establish local control flow and contract behavior. Deployment identity,
real provider availability, legal completeness, quality, cost and wall time require
separate live validation.
