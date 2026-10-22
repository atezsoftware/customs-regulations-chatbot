# Metadata-only regulatory source notices

ASv3 Tek (`standard`, normal, nonparallel) and Exp Cropped can consult a small
document-set-scoped catalogue of instruments whose names and repeal annotations
are known while their original texts have not been uploaded.

These are `regulatory_source_notice` records. They are distinct from `UserFile`,
canonical `RegulatoryChunk`, original evidence and Elasticsearch documents. No
empty original, ingestion-complete state, embedding or publication receipt is
fabricated. Existing source files and indexes do not need rewriting.

## Identity, scope and provenance

Each notice has a UUID, a typed `identity_key`, its original catalogue label,
optional aliases, `legistation_metadata`, workbook provenance and a revision.
Automatic admission requires a reviewed unambiguous identity. A title alone,
number alone, or an identity shared by different authorities is insufficient.
Conflicting, existing and weakly identified spreadsheet records are held outside
the append-only manifest. Duplicate observations retain all original row locators.

Queries use the ordinary tenant and PC Külliyatı access boundary. Narrower scopes
that a metadata-only entry cannot satisfy fail closed for notice enrichment,
while ordinary real-source results remain unchanged. Model delivery refreshes
current access and revisions, including after checkpoint restoration.

## Research behavior

The source resolver can return a fenced target ID, `notice:<UUID>`, and its
catalogue annotation. Trying to read a metadata-only target returns an
`unavailable` tool outcome with `source_known: true`,
`content_available: false`, `original_acquired: false`, and the precise status
annotation. It does not return original evidence or a numbered legal citation.

Newly acquired genuine canonical chunks are also checked for explicit typed
first-level references without waiting for a model `follow_reference` action.
The lookup is deterministic and batched; it makes no embedding, reranking or
additional model call. Notice descriptions are never recursively scanned.

Only encountered notices reach model context. Their small `source_notices`
catalogue survives history compaction and is supplied to researchers, writers,
repairs and verification. Source notices cannot satisfy an original-reading
requirement. The system preserves uncertainty about an unavailable provision,
repeal date or historical applicability. A repeal annotation about a target also
does not establish repeal of the referring provision.

Known-empty exact source reads and repeated focused searches return the terminal
metadata-only result. Broad research, a search for the repealing instrument and
independently identified sources remain available. Other workflow variants keep
their existing behavior.

Before final publication, access and the last delivered notice revisions are
revalidated. Revoked or changed annotations cannot be published from a stale
checkpoint or draft.

## Later original-content availability

A verified, normally uploaded and published `UserFile` can be explicitly linked
to the existing notice through `user_file_id`, `content_state=materialized`, and
an increased revision. This preserves notice identity and provenance. The new
real file retains its own ordinary source and chunk IDs; no fake chunk needs
replacement. Materialized notice reads redirect to that file through ordinary
current source permissions and publication checks. A failed upload leaves the
metadata-only record intact.

The catalogue does not change canonical validity windows. An unknown effective
repeal date stays unknown. Original-content linking requires verified instrument
identity and successful ordinary publication; filename equality is insufficient.

## Schema and import operations

The migration creates only the new table and its own indexes. An online upgrade
also validates an identically pre-created registry without recreating it; schema
drift is rejected. This permits a bounded DEV table bootstrap without running
unrelated migrations or advancing the existing Alembic version.

`onyx.db.source_notice_import` separates a read-only manifest from append-only
application. A reviewed manifest binds the workbook hash, current file inventory
and a bounded canonical identity-head audit. Apply revalidates these bindings,
uses a transaction-scoped importer lock and writes only previously absent notice
identities. Conflicts are not overwritten. Import safety checks reject changes to
existing tables, upserts, mutations hidden in SELECT, DDL and multi-statement SQL.

Operational helpers, credentials, workbook rows and acceptance captures are local
task artifacts; they are not part of the release. Preservation acceptance compares
the pre-existing files, canonical rows, publication records, memberships and SQL
index definitions. No Elasticsearch request is required by this feature.
