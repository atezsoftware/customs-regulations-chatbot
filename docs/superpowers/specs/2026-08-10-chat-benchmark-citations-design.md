# Chat, Benchmark, Citation, and Research Workflow Design

## Goal

Make per-chat model selection authoritative, expose the active OpenRouter catalog,
make regulatory benchmark runs reliable and usable, render exact legal chunk
citations, enforce an agent's Document Set retrieval boundary, and document the
Deep Research workflow from the active code path.

## Decisions

### Chat model selection

The chat-session model endpoint persists both the UI selection and the structured
`LLMOverride` consumed by chat execution. Exact provider instance names remain
preferred; a provider-type fallback is accepted only when it resolves to one
unique nameless provider. Invalid or ambiguous selections fail explicitly instead
of falling back to the administrator default.

### OpenRouter catalog

OpenRouter auto mode treats the current provider discovery response as its visible
catalog. Models present in the response become visible; stale historical rows are
not re-exposed merely because they exist in the database. Non-OpenRouter provider
visibility behavior remains unchanged. A forward migration reveals existing
OpenRouter rows at deployment so users do not wait for the next sync.

### Benchmark reliability and interface

Persona selection returns the single highest-priority accessible persona rather
than requiring that only one accessible persona exist. An unexpected benchmark
worker failure records a run-level error and terminalizes unfinished items so the
UI cannot remain indefinitely pending. The admin interface uses the existing Opal
design language, separates question management from run operations, presents a
clear run builder and progress/status hierarchy, and allows a failed or pending
run to be retried without recreating it.

### Exact citations

Citation identity remains `(document_id, chunk_ind)`. Regulatory labels are
derived deterministically from the indexed heading path/semantic identifier,
preferring the document title and nearest article heading (for example,
`Gümrük Kanunu · 46. Madde`). This avoids an extra model call and prevents a model
from inventing legal attribution. Citation clicks use the authenticated exact
chunk endpoint and render only that chunk; they never fall back to another chunk
from the same document.

### Agent Document Set boundary

A persona with Document Sets must execute through the internal SearchTool. Normal
editor writes persist that association, while runtime construction repairs legacy
records that lack it. Request filters may narrow the persona's Document Sets but
may not replace or escape that boundary. Deep Research fails clearly if a scoped
agent still has no usable SearchTool instead of producing an ungrounded generic
report.

### Workflow documentation

The repository contains a versioned Markdown document with Mermaid architecture,
sequence, and control-flow diagrams generated from `process_message`, `dr_loop`,
and `research_agent`. It identifies document-set scoping, concurrency and timeout
limits, evidence review, citation publication, and persistence.

## Error Handling

- Model/provider ambiguity returns an invalid-input error.
- Benchmark task crashes retain diagnostic text and do not leave pending items.
- Exact citation chunk mismatches fail closed and show an actionable UI error.
- Scoped research without SearchTool returns a configuration error.

## Compatibility and Scope

The implementation does not alter provider credentials, model pricing, normal
full-document preview behavior, benchmark historical data, or visibility rules
for other providers. Existing user changes outside the files named in the
implementation plan are preserved and excluded from delivery.

## Verification

Focused backend and frontend regression tests cover every behavior above. Live
verification uses the active Compose topology without changing credentials or
privilege roles, then the exact selected diff is committed and pushed to
`test/v1`. The deployment workflow and service health/logs are checked after the
push; failures are diagnosed from the actual job or runtime logs.
