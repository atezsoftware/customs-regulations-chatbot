# Repeated Regulatory Amendments Design

## Problem

An amendment can target a chunk that was created by an earlier approved
amendment. The current flow can retrieve a stale Elasticsearch identity, carry
mutable heading text into the replacement, lose the original chunk type, and
accept an LLM draft that does not contain the explicit replacement text. A
whole-paragraph replacement can also leave prior descendant chunks active,
which creates contradictory active evidence.

Batch 21 demonstrates the structural corruption: its active Article 20
paragraphs contain the new value while their heading paths retain the old
value. The replacement for paragraph 3 contains omission markers while older
child clauses remain active.

## Design

### Canonical repeated-version retrieval

PostgreSQL remains authoritative. Search-result chunk IDs are resolved through
their `superseded_by_chunk_id` lineage to the current active descendant, within
the same file. Search rank and source identity are retained while duplicate
active descendants are collapsed. The existing source-scoped structural lane
remains available when the instrument identity is sufficiently distinctive;
generic source labels are not broadened across unrelated instruments.

### Deterministic draft integrity

The LLM continues to segment amendments, formulate search queries, select
candidates, draft non-literal edits, and resolve dates. For an explicit
"aşağıdaki şekilde değiştirilmiştir" replacement, the quoted replacement body
is authoritative. A proposal is emitted only when the drafted text contains
that complete body after whitespace and lightweight markup normalization.

Replacement bodies containing omission markers such as `...` or `…` are not
approvable when they replace a provision that owns descendant chunks. They are
returned as an attention item rather than silently publishing a partial
subtree.

### Structural consistency

For an existing target, a missing drafted `chunk_type` inherits the old type.
The terminal structural heading is rebuilt from the amended text using the
same marker and 90-character label convention as the regulatory chunker.
Stable document, section, and article ancestors are preserved. The normalized
heading path is written both to the chunk column and to the metadata copy.

### Subtree safety

This change does not guess missing legal text. A complete replacement that
contains child clauses may proceed only when the proposal can represent the
same atomic structure safely; otherwise analysis returns an attention item.
This prevents a new parent chunk from coexisting with contradictory old child
chunks. A future multi-row change-set can extend the proposal schema, but is
not required to make incomplete replacements safe.

### Projection and publication

Approval continues to project only the bounded affected neighborhood. The
projection write remains part of the approval worker, so a bulk indexing
failure prevents approval from becoming terminal. Existing Gemini embedding
configuration and the Elasticsearch 1024-dimension contract remain unchanged.

## Data repair

After deployment, Batch 21 Article 20 paragraph 1 can be repaired
deterministically without creating a new legal version: restore its inherited
chunk type, rebuild both heading-path copies from the active text, and reproject
only the affected neighborhood. Paragraph 3 must not be guessed from its
abbreviated `...` source; it requires the complete legal replacement text
before its subtree can be rebuilt.

## Acceptance criteria

- A stale search hit for version N resolves to active version N+1.
- A subsequent 7-to-8 batch shows `Before=7` and `After=8`.
- Draft text, terminal heading, metadata heading, and chunk type agree.
- An explicit replacement whose draft omits the quoted new body is rejected.
- An incomplete parent replacement with active descendants is not approvable.
- Only the affected structural neighborhood is embedded and indexed.
- DEV verification uses the real admin UI approval action and confirms the
  terminal proposal, PostgreSQL lineage, Elasticsearch identity, and search
  answer.
