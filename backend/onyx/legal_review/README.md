# Legal Review

Legal Review is an opt-in workflow (`legal_review: true`) through the normal authenticated
frontend `/api/chat/send-chat-message` endpoint. It runs independently of ASv3, Legal
Composite, Supersearch and Deep Research. The captured authorized PC Külliyatı corpus,
ACLs, dates, canonical text hashes and publication fences apply throughout.

## Workflow

1. **Plan — Gemini 3.8 Flash, medium reasoning.** Identify coherent legal decisions,
   supplied facts, missing facts, materiality and closure criteria. Issues are neither
   copies of numbered subquestions nor twelve checklist categories. Several requests can
   share an issue; one request can expose distinct decisions. A plan-wide set of focused
   queries binds all covered issue IDs. There is no numeric issue-count limit.
2. **Discovery — parallel hybrid search, reranking, canonical acquisition.** Independent
   searches run with concurrency four. The owned search fork uses inclusive `0.82`
   global min-max normalized reranker scores. Baseline candidate coverage remains: 256
   hits per lane, 384 rerank candidates, 50 model chunks plus the qualifying band and
   source diversity. Embedding and ranking use deployment-configured providers. No fixed
   topic-to-source catalogue or benchmark-specific source boost is installed.
3. **Read — Gemini source accounting and synthesis.** Newly received originals are
   accounted for in parallel batches, not one LLM call per source. Dispositions and exact
   canonical passage references record used, rejected and unresolved evidence. Synthesis
   records source rules and assesses all twelve dimensions per issue; subsequent rounds
   update changed assessments. A relied-on provision is expanded to its article context.
   Completed source batches survive a later batch reaching the phase deadline.
4. **Develop issues from evidence.** A material new condition, exception, operative
   effect or cross-source dependency can create a child issue in any dimension. It needs
   a parent, original-backed trigger, material reason and closure criteria. The host
   preserves those bindings and does not close a parent with an unresolved material child.
   Incidental citations and every source do not automatically become issues. Initial
   discovery can have multiple queries. A specific later gap gets one focused search and
   at most one justified improved retry; identical and renamed repeats are excluded.
   Independent queries are combined and executed in parallel.
5. **Conditional research review.** Native OpenAI Decisions (`gpt-6-luna`) screens
   material unresolved research where needed. An independent `gpt-6.1-sol` examiner
   separates actual source gaps from corrections and rebutted suspicions. Confirmed
   research tasks are dispatched by the host, not offered back to the reader to skip.
6. **Draft — Gemini.** Produce one integrated answer with block-level claim/source
   bindings. Issues remain a research checklist rather than mandatory answer sections.
   Exact cited passage selectors are checked by code; this does not prove entailment.
7. **Review — Decisions and independent examiner.** Screen the entire answer, including
   claims outside identified issues. The examiner confirms or rebuts suspicions against
   whole originals. Larger check inventories run in up to four parallel batches; every
   batch sees the entire answer and source context. Equivalent findings share diagnoses.
   Existing-source corrections and missing-source research remain distinct.
8. **Correct or research.** If existing originals suffice, the independent examiner
   edits the affected answer blocks directly. No research or separate Gemini rewrite is
   needed for that path. Material source gaps can use one post-draft research/repair
   round within the available time, followed by review. The final editor may correct
   remaining wording and qualify unresolved conclusions once; it cannot start searches.
   Its response must account for every supplied finding, preserve unaffected blocks and
   retain valid claims, source selectors and known issue identities.
9. **Publish.** Publish the available answer when the time/repair boundary is reached,
   even when review findings remain. `publication_mode` distinguishes reviewed,
   editor-adjusted, limit-reached and review-incomplete answers. The latter three are
   explicitly partial, never verified. Unresolved confirmed findings remain visible.
   If the first writer did not finish, return established research findings labelled as
   incomplete application to the facts, or honestly state that no outcome was established.
   User cancellation and revoked/inconsistent source authorization still block delivery.

## Time and source assumptions

Default total generation/research time is 600 seconds. Initial research reserves the
last 300 seconds for writing, review and corrections. Drafting retains 120 seconds for
publication work; examination retains the last 60 seconds for the independent editor.
Each source/model phase enforces its absolute deadline as well as a 45-second idle
network timeout. Provider failures do not gain unlimited retries. A separate bounded
30-second delivery context permits authorization revalidation and streaming after the
model deadline; it preserves cancellation and does not admit more model/tool research.
This is an approximate ten-minute response target, not a transport latency guarantee.

There is no aggregate token quota, model-call quota, issue-count limit or total search
count limit. Physical concurrency, the overall deadline, per-gap retries, provider
context/output capacities and evidence/request byte capacities remain enforced. Source
text is never silently truncated to fit a model. Source-accounting provider schemas
select only existing citation/span pairs. Gemini and review usage counters exclude
embedding and external reranker costs; those have their own inference traces.

The default corpus policy tells every analysis stage that supplied chunks are current
versions of the user's maintained corpus. Missing validity metadata alone creates no
new issue or disclaimer. This does not fabricate a database in-force status. Explicit
amendments, annulments, event dates, quotations of superseded law and conflicts between
current sources still govern. Retrieving a statutory name and administrative explanation
alone does not settle a material scope or authority interaction.

## Publication and audit

`verified` means complete review accepted the answer and the workflow has no remaining
material gaps. It is not a certification of exhaustive legal research. `partial` can mean
reviewed limitations, examiner-edited text without another independent review, or an
available answer whose review/time budget ended. `unavailable` is retained for technical
failures with no usable result; `cancelled` is a user-stopped run. Merely reducing warning
counts is not a quality metric.

Execution graphs retain the actual provider generations, search/reranker/canonical
operations, review inputs and results. Additional explicit steps are:

- `legal_review.issue_plan`: issues, facts, closure criteria and bound discovery queries.
- `legal_review.issue_update`: adopted issues, new child IDs, source triggers, closure
  status and pending source operations after an accepted reading.
- `legal_review.review_diagnoses` and `legal_review.final_adjudication`: research versus
  correction decisions and their supporting passages.
- `legal_review.editor` / generation flow `legal_review_editor`: actual editor input,
  replacements, before/after text, resolved/unresolved check IDs, or editor failure.
- `legal_review.publication_decision`: publication mode, reason, remaining time and
  whether repair/editor were used. A partial publication does not mark research closed.

Checkpoints also retain canonical source journeys from retrieval receipts and reading
assessment through requirements to final draft claims, plus candidate-selection audit.
Credentials are never included. No ingestion, relabelling, reindexing or source metadata
mutation belongs to this workflow.

## Validation

Run the owned provider-free tests and protected workflow routing checks using the root
uv environment. These cover dependencies, duplicate/retry admission, canonical selectors,
partial completed batches, targeted edits, whole-answer review, deadline delivery,
cancellation, authorization revocation, publication labels and graph events. Real local
provider runs and deployed frontend API/graph checks separately establish live behavior;
offline passing tests alone do not establish legal completeness or live latency.
