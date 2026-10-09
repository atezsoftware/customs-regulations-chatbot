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
   `gemini-3.8-flash`. Resolve a real JEV credential before research begins: prefer
   `TYPESAFE_API_KEY`, otherwise use an existing authorized provider with the official
   OpenRouter base URL and a configured key. Existing provider cost limits also apply.
   This does not require registering JEV as a chat model. All available
   simple conversation history is retained. Earlier user messages can supply facts;
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
   conditions. Each support must quote a nonempty exact substring of the canonical,
   hash-bound original. Every issue needs exactly one assessment for every dimension.
   An addressed dimension must reference a requirement for the same issue and
   dimension. Corrections use explicit supersession; obsolete interpretations remain
   in audit history and leave the current drafting state.
5. **Source-derived issues — Gemini:** the reader actively resolves existing issues.
   It can open a linked child when a question discovered in the originals could
   materially change the parent's outcome and cannot be handled adequately inside
   that issue. This judgment applies across all twelve dimensions. An exception,
   tax consequence, procedural condition or judicial effect can be a dependency;
   merely citing a provision never automatically creates a child or an article search.
   A child needs a stable ID, parent, dimension, exact canonical trigger citations,
   parent-backed requirement IDs, a material reason and explicit closure criteria.
   Code validates those bindings, rejects cycles and deduplicates the same
   parent/canonical-trigger/dimension dependency. The model can use existing originals,
   request a canonical read or choose a focused search. One early source return can
   execute its requested operations and reread the originals before the early JEV review.
   Missing evidence and unexecuted operations remain gaps; a zero-result search never
   proves absence of law or continued legal validity.
6. **Early review — JEV:** one HTTP request checks all issue/dimension combinations
   together, requested-outcome coverage and decisive source conditions. It receives
   complete retained originals, actual conversation facts and compact research
   receipts with query, status and access/truncation limitations. If the early source
   return has not been used, flags can direct that one return. Incomplete review
   prevents drafting a publishable answer.
7. **Integrated draft — Gemini:** produce one answer with a literal claim inventory,
   citations, conditions and unresolved issues. Issues serve as a private checklist;
   they do not force repeated per-issue answer sections. Code validates source quotes,
   literal answer excerpts and citation targets and inserts any required scoped
   validity disclosure before review.
8. **Draft review — JEV:** independently check all dimensions, the entire answer's
   coverage, every material legal assertion (including assertions absent from the
   planned issues), source conditions and consistency across issues. JEV scores are
   defect probabilities; scores at or above `0.5` flag the bound question. A flag is
   a suspicion to investigate, not a legal finding.
9. **Optional repair:** permit one post-draft evidence-directed repair. The reader
   sees the actual draft and flags, can obtain missing originals, and the writer
   produces a complete replacement. One final JEV recheck follows. Remaining flags
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

## Calls, retrieval width and cost accounting

The shortest path is three Gemini generations (planner, reader, writer) and two JEV
requests. A model-requested early source return adds a reader generation, making that
path four Gemini generations and two JEV
requests. A flagged early review can add a diagnostic reader call. Post-draft repair
adds its diagnostic reader, an optional reader after acquisition, one replacement
writer and one JEV request. These are bounded paths, not a fixed per-answer bill.
An invalid initial structured plan permits one separately admitted Flash schema
correction. It preserves the request and uses the same remaining deadline and call
budget; provider failures do not trigger this correction.

Planner, reader, writer and repair use Gemini 3.8 Flash with low reasoning effort.
The fixed override has temperature zero. JEV uses either the real TypeSafe SystemOne
endpoint with `jev-latest`, or the official OpenRouter SystemOne endpoint with
`typesafe/jev-1.13`. Both routes make one physical request per invocation, without internal retries or
partitions. Missing credentials, invalid responses, deadline failures and oversized
packets never become a successful review.
There is no GPT/native reviewer substitute and no fallback to another provider after a
review failure. Checkpoints retain only JEV route/provider-name provenance, never keys.

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
they are **not** included in the Gemini/JEV token totals or the 32-admission counter.
No total dollar-price or live latency guarantee follows from the stage counts.

## Default limits and cancellation

| Limit | Default |
| --- | ---: |
| Overall cooperative deadline | 240 seconds |
| Finalization reserve | 110 seconds |
| Planner/reader/writer/repair or JEV call timeout | 45 seconds |
| Shared Gemini/secondary-LLM/JEV admissions | 32 |
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
| Serialized JEV request/response cap | 256,000 bytes each |

Actual provider usage is recorded after each response, including a failed JEV
validation with reported usage. Token thresholds are admission checks against usage
already observed; they are not an exact advance estimate of the next response's
bill. There is one provider attempt and one compatibility attempt for selected LLM
calls. Secondary search generations use the remaining research deadline; external
embedding/reranker providers retain their own timeout behavior. The smaller JEV
provider token limits also apply: the byte cap alone does not guarantee a packet fits
those limits. Provider rejection produces an explicit incomplete review.

There is no numeric issue-count limit. Materiality, reuse of existing issues and
global time, call, search, tool and context budgets control growth. Every explicit
requested outcome must remain represented; the planner must not compress outcomes to
fit an artificial issue cap. Identical initial queries and identical source operations
within a reading batch are executed once with their issue IDs combined. A decisive question that cannot be resolved within the remaining
budget stays open and disclosed instead of being forcibly closed.

Stop checks propagate through research, selected provider calls, JEV response
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
limitation into the literal draft before final review when needed.

Recorded version windows, read dates and a chunk lifecycle value such as `active`
do not prove legal in-force or annulment status. A recorded window is checked only
when an explicit captured `as_of_date` agrees with the original read's date. Otherwise
the requirement's temporal status is unknown. A legal status requires a separately
verified canonical status field; the model cannot assert it into existence.

The current shared evidence metadata whitelist does not transport verified legal
status fields. This workflow does not change that shared contract or fabricate a
sidecar. Ordinary corpus evidence therefore retains `legal_status="unknown"` even
after a targeted validity search. Any affected result is `partial`, and code adds a
scoped, user-visible validity limitation to the literal draft before its final JEV
review. A clean JEV score cannot promote unknown status to in-force status.

`verified` requires passed complete reviews and no remaining evidence, validity or
pending-operation gaps. `partial` requires passed complete reviews with accurately
disclosed remaining limits. `unavailable` has no publishable answer; `cancelled`
records a stopped run. These are workflow publication states, not a certification
that the corpus contains every potentially relevant legal source.

## Validation

Provider-free tests cover review failure, literal source support, dimension closure,
supersession, model-chosen dependencies across dimensions, canonical child triggers,
dependent parent closure, shared searches without losing outcomes, cancellation, both citation
display modes, persisted answer/citation state, publication revocation and missing
JEV credentials. Run them with:

```bash
uv run --no-sync pytest -q backend/tests/unit/onyx/legal_review \
  backend/tests/unit/onyx/chat/test_legal_review_workflow_routing.py
```

These tests establish local control flow and contract behavior. Deployment identity,
real provider availability, legal completeness, quality, cost and wall time require
separate live validation.
