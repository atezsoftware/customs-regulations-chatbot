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
   `openai` provider with visible `gpt-6-luna` for native Decisions and `gpt-6.1-sol`
   for independent diagnosis/publication examination, using the official OpenAI base
   URL before research begins. Existing group, persona
   and provider cost limits also apply; provider rows and model visibility are not
   changed. All available simple conversation history is retained. Earlier user
   messages can supply facts;
   earlier assistant legal statements are not independent legal evidence.
2. **Planner — Gemini:** identify requested outcomes, supplied facts and missing
   facts. Return stable issue IDs and a required nonempty plan-wide `discovery_queries`
   list with query text and covered `issue_ids`. Related issues share searches; no
   per-issue query is required. Code
   attaches the twelve dimensions; issue identification does not assume unseen law.
3. **Discovery and canonical acquisition:** run parallel hybrid searches with bounded concurrency,
   rank candidates and hydrate authorized originals into the shared evidence ledger.
   Search hits, labels and rank scores provide navigation, rather than legal proof.
   Each operation preserves the model's concrete evidence question as its reranking
   context. The full parent case remains available for legal analysis, without replacing
   the relevance target of an independent source search. Shared queries retain all their
   bound questions; source and article identities receive no case-specific boosts.
4. **Reading — Gemini:** first account for newly received sources in parallel batches,
   preserving every original and source identity. The source pass uses medium reasoning
   to distinguish an operative rule or disposition from a quotation, party submission,
   reasoning or procedural background. It reports a concrete established effect or a
   material missing effect. A quoted rule cannot establish the containing decision's
   outcome. The model selects canonical continuations for material unread effects;
   the host executes those reads together before synthesis. It does not rediscover the
   same source or create an issue per document. Completed identical reads are not
   repeated, and no-new-evidence operations leave their missing effect open. An expanded
   source or changed research question invalidates the corresponding assessment cache.
   This inventory is recorded in checkpoints and generation traces; independent review
   receives the originals rather than adopting these preliminary interpretations.
   Then extract source-backed requirements and application
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
   request a canonical read or choose a focused search. Each distinct material gap receives
   one discovery search; an empty or failed result consumes that attempt. An initial broad
   issue search does not consume the focused search for a specific gap exposed by its results.
   A genuinely different source-derived dependency gets its own attempt, while a paraphrase
   of the same unresolved question does not. Canonical reads can complete returned sources.
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
   receipts with query, status and access/truncation limitations. An independent batched OpenAI examiner diagnoses flagged checks using code-owned
   response slots. Each research diagnosis contains its concrete source questions. Code
   assigns task identities and compiles the shared inventory; the model never has to keep
   forward task references aligned with a separate list. Identical questions and query
   strings share work without dropping their bound diagnoses or supporting passages.
   Dimension-specific slots retain their assessed dimension and cannot alias a diagnosis
   in another dimension or issue. Their explicit finding scope includes all findings used
   by that issue, across its dimensions, rather than inheriting one narrow assessment.
   A second batched OpenAI Responses call combines the accepted questions into shared
   searches. Questions about the same controlling norm can share one query for conditions,
   amounts, exceptions, temporal changes and judicial effects. The planner returns fixed
   gap keys, explicit shared-group references and a coverage reason for each gap; code
   resolves their dependency graph and rejects missing keys, unknown references and cycles.
   It retains every diagnosis and derives the dimensions covered by each shared search.
   Query sharing never closes an assessment or proves that all its evidence was found.
   Subsequent reuse is allowed in the dimensions explicitly covered by the prior search;
   an operative-only search cannot consume a separate judicial question. Automatic identity
   matching also includes subject and question. Independent tasks run together, with no
   per-source or per-flag model call and no arbitrary query-count target.
   Provider schemas encode fresh work (a required nonempty query) and reuse (an eligible
   attempted need ID with no new query) as separate alternatives. Both fields being null
   is invalid in the schema sent to the provider, not merely in a post-response validator.
   Only code-owned specific gap IDs are eligible for reuse in its response schema; broad
   initial issue searches cannot suppress newly identified research. Its material source
   questions include discovery queries, which the host executes in one batch with bounded
   parallelism before handing the results to the reader. The reader does not
   decide again whether to execute an independently identified source task. It returns
   actual findings, changed assessments and further source actions, rather than a second
   diagnosis or a self-declared completion inventory. A known original may retain its
   citation number; novelty of an ID is not evidence of completed research. Source
   operations and the reader's claims remain subject to independent answer review.
   Evidence reading answers a code-owned keyed inventory of concrete research diagnoses;
   the host restores their check and issue identities. Correction-only findings cannot
   accidentally enter that inventory. Each shared investigation provides an index of all
   its returned originals, preserving potentially limiting or contrary leads without
   selecting a particular source family or supplying a legal outcome. Canonical texts
   remain intact. A search receipt is not a resolved legal question.
   Native Decisions refusals retain the valid results for other checks and carry the
   unanswered check into the one repair as unassessed. Transport or malformed-response
   failures still stop the workflow. Publication requires a completed final review;
   a refusal never counts as a passing score.
   When a completed final review rejects the repaired draft, the chat shows an
   explicit non-answer notice instead of misclassifying that outcome as a model
   provider error. The checkpoint remains `unavailable` with no published legal
   answer and records `non_publication_reason: review_rejected`. The notice carries
   no draft text or citations. Incomplete reviews, provider/transport failures and
   revoked source access remain failures; cancellation still prevents publication.
7. **Integrated draft — Gemini:** produce a `GeneratedDraft` of ordered Markdown
   blocks with claim bindings, passage selectors, conditions and unresolved issues.
   The writer receives finding-to-source bindings, recorded validity and unresolved gaps,
   while private rule paraphrases and affirmative assessment reasons stay in the research
   ledger. Repair receives exact confirmed error targets and prior source bindings instead
   of copying the old answer as a template. All canonical originals remain intact.
   Within the same call, each block selects its supporting passages and records a concise
   source-to-case application before generating its prose. That private record states the
   operative source conditions, their match to supplied facts and any remaining uncertainty.
   It is retained in generation traces, not published or used as proof by the independent
   reviewer; the reviewer still receives canonical originals and the literal answer.
   Code adds missing selected source markers as separate paragraphs within their
   containing blocks, preserving Markdown fences and tables. It compiles the actual
   rendered block text into one literal answer and derives claim excerpts from it.
   Foreign model-written markers are retained for rejection by validation. The model
   does not recopy an answer or `answer_excerpt` field. Issues serve as a private
   checklist; blocks do not force
   repeated per-issue sections. Code validates passage selectors, literal claim
   bindings and citation targets. Code binds unresolved issue IDs from the complete
   closure ledger without adding blanket legal conclusions or rewriting the prose.
   The writer distinguishes missing facts, unverified metadata and established adverse
   effects, and qualifies only the affected conclusions. The full literal answer and
   its actual disclosures remain subject to independent review. A limitation-only answer can have no claims only with explicit
   unresolved issue IDs; it still undergoes complete independent review.
8. **Draft review — OpenAI Decisions:** independently check all dimensions, the
   entire answer's coverage, every material legal assertion (including assertions absent from the
   planned issues), source conditions and consistency across issues. The same
   request includes finding-bound checks and claim-bound checks, so the repair can
   locate the particular rule or literal block and its selected originals. These
   predicates add no per-source calls or extra review rounds. Reviewer scores are
   defect probabilities; scores at or above `0.5` flag the bound question. A flag is
   a suspicion to investigate, not a legal finding. Research review assesses private
   findings and dimension reasons. Draft review assesses the literal rendered answer;
   its packet retains all originals and finding-to-source/issue bindings but excludes
   private interpretations that could be mistaken for assertions in the answer.
   Open issue status alone does not establish an answer defect: the reviewer checks
   whether the corresponding conclusion preserves the material qualification.
9. **Optional repair:** permit one post-draft evidence-directed repair. A separate
   publication examiner first distinguishes actual answer defects from rebutted flags
   and correctly disclosed limitations. It assesses the literal answer, not ideal research
   completeness. It selects code-owned answer passages; the host derives exact quotations
   instead of asking the model to retype them. The answer, inventory and source selectors
   are validated. If no material defect remains, no rewrite or new research is triggered.
   Only confirmed defect checks enter the independent research diagnosis.
   Each check carries its matched publication finding, literal answer passages and requested
   change into diagnosis; a broad predicate does not replace the identified defect. The
   query planner receives only prior attempts referenced by the accepted gaps. With at most
   one fresh query, no query-sharing call is needed; existing attempts remain unchanged.
   The reader then sees
   the actual draft, diagnoses, source receipts and originals. Code binds every flag
   to the examiner's diagnosis and executes its discovery queries before the reader.
   The reader updates findings and their assessment uses and can request more source
   operations. The writer receives these evidence updates and the independent change
   plan, then produces a complete replacement. The reader does not redundantly
   reclassify every flag or attest that its own correction has passed review.
   One Decisions recheck follows. Remaining probabilistic flags receive one independent
   batched examination against the exact replacement answer and complete originals.
   Its inventory, selected literal answer text and all source passages are validated.
   Publication requires every residual flag to be substantively rebutted or shown to be
   an accurately disclosed limitation. A concealed source gap or a false positive conclusion
   still blocks publication, as do any remaining material correction, invalid inventory or
   incomplete native review. This examination does not execute searches or start another repair.
   Raw scores and flags are preserved alongside the final adjudication for audit.
   The shared review batch also includes one source-use predicate per distinct cited
   original. These are review checks, not issues or separate LLM calls. They prevent a
   broadly supported answer section from hiding misuse of a narrower cited provision.
   All original evidence remains available to the reviewer; source-use checks do not
   replace whole-answer, contrary-source, or twelve-dimension review.
   An inline numeric citation also selects a source. The compiler preserves the literal
   prose and binds inline selections absent from structured claims to the complete
   canonical original, within the containing block's issue scope (or the same source's
   existing issue bindings for a claimless summary). It does not infer legal entailment.
   Unknown, noncanonical or unscoped references fail validation; new bindings undergo
   the same claim, source-use and whole-answer review. No binding-repair LLM call is added.
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

Planner, reader, writer and repair use Gemini 3.8 Flash with medium reasoning effort.
Their model transport shares repeated source metadata and heading prefixes in a
document registry. Citation numbers, chunk identities, complete passage text and all
temporal/closure metadata remain recoverable without truncation. Canonical hash checks
remain in the host ledger; digest strings are not repeated in the model-facing view.
Raw acquisition receipts are replaced by query/status and discovery-limitation
diagnostics, and writer stages omit tools. Post-draft repair receives the current
review rather than duplicate obsolete early-review predicates. The full prompt,
structured schema and provider response format are included in context admission.
Vertex timeout exceptions follow the workflow's controlled failure/checkpoint path;
they do not escape as an unhandled provider stack trace. Call deadlines remain bounded.
The fixed override has temperature zero. The independent reviewer uses the native
`https://api.openai.com/v1/decisions` endpoint with pinned `gpt-6-luna` and named
defect predicates. Predicate names map back to code-owned checks; the provider does
not generate queries or workflow actions. Each invocation makes one physical
request without internal retries or partitions. The separate Responses examiner uses
one streamed batch for flagged checks. Its output allowance scales with the check
inventory within the model's capacity. Terminal status and usage are captured before
structured parsing; truncated output is never repaired into an accepted diagnosis.
Missing credentials, refusals,
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
and are counted in the shared execution counters and Gemini usage meter. Embedding
and external reranker calls retain their own provider traces and usage accounting;
they are **not** included in the Gemini/Decisions token totals.
No total dollar-price or live latency guarantee follows from the stage counts.

## Default limits and cancellation

The owned Legal Review search fork uses an inclusive `0.82` threshold on the
global min-max normalized external reranker score. Selection retains the baseline
candidate coverage and adds the complete qualifying score band. Other search
workflow classes retain their existing `0.90` default. The search trace records the
actual normalized threshold and selected identities.

| Limit | Default |
| --- | ---: |
| Overall cooperative deadline | None by default; explicit cancellation remains active |
| Finalization reserve when an explicit deadline is configured | 80 seconds |
| Gemini reader, examiner and native Decisions idle-network timeout | 45 seconds; no cumulative call deadline |
| Aggregate model admissions, source operations and reader rounds | No quota |
| Discovery searches per distinct material gap | 1; initial broad discovery is separate |
| Parallel source operations | 4 |
| Post-draft repairs | 1 |
| Aggregate input/output token quota | None by default |
| Gemini context ceiling | Configured provider input capacity; an explicit policy cap may lower it |
| Decisions input ceiling | 922,000 tokens (GPT-6 Luna context minus full output reserve); an explicit policy cap may lower it |
| Maximum generated output per Gemini call | 65,536 tokens (provider capacity) |
| Maximum generated output per OpenAI diagnosis/planning call | 128,000 tokens (provider capacity) |
| Evidence capacity | 2,000,000 bytes |
| Serialized Decisions request/response capacity | 4,000,000 / 256,000 bytes |

Actual provider usage and execution counts are recorded. Technical context and packet
capacities still apply; originals are never silently clipped to fit. Native Decisions
requests also undergo model-token admission. Reader streams enforce idle-network timeouts,
explicit cancellation, and any configured absolute phase deadline, without imposing a total
reading duration on the default unlimited run. Each provider call has one physical attempt.
Embedding/reranker providers retain their own timeout behavior.

The workflow does not impose a shared 192,000-token context ceiling on different
providers. Gemini uses its configured input capacity. Native Decisions uses the
[documented GPT-6 Luna capacity](https://developers.openai.com/api/docs/models/gpt-6-luna),
with output headroom. Compact JSON changes only transport whitespace; all original
passages and provenance remain present. Larger-than-provider requests still fail admission.

There is no numeric issue-count or aggregate search-count limit. Each specific gap receives
one search, shared across all checks it can resolve; independently useful searches run with
four-way concurrency. Matching issue IDs or dimensions alone does not establish duplicate
questions. Materiality, reuse of existing gaps and supported resolution guide research.
Every explicit requested outcome remains represented.
Identical operations within one batch execute once with their issue IDs combined. Repeating
identical operations with unchanged evidence is detected as no progress. Missing user facts
remain explicit and conditional. After its single attempt, a missing authority remains a
precisely disclosed gap rather than triggering another wording of the same search.

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
writer/reviewer state, result and checkpoint. Code binds unresolved issue IDs without
appending a blanket disclaimer; the writer must qualify the affected conclusions and
the independent reviewer checks those actual qualifications. Validity and
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
