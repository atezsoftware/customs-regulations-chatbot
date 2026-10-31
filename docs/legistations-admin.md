# Legistations

The admin screen at `/admin/legistations` browses the tenant's uploaded files across owners, document sets and processing statuses. Its filename-derived folder tree is loaded independently from the current file page. File and chunk lists are paginated; original filenames and stable IDs are retained.

All endpoints below accept the normal session or an admin API key in `Authorization: Bearer <key>`. They require `FULL_ADMIN_PANEL_ACCESS`; ordinary user keys do not gain catalogue-wide access. Call them through the frontend's `/api` proxy.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/regulatory/catalog/files?q=&folder=&recursive=true&offset=0&limit=25` | Filename search and file pages; omit `folder` for all files |
| GET | `/api/regulatory/catalog/folders?parent=&q=&offset=0&limit=100` | Immediate folder children across the catalogue |
| GET | `/api/regulatory/catalog/files/{file_id}` | File metadata and canonical chunk count |
| POST | `/api/regulatory/catalog/files/{file_id}/index` | Queue initial indexing for one reviewed CHUNKED file |
| POST | `/api/regulatory/catalog/search` | Direct BM25, semantic or hybrid chunk retrieval |
| GET | `/api/regulatory/files/{file_id}/chunks/page?offset=0&limit=25` | All canonical chunks of a selected file, including historical rows |
| PATCH | `/api/regulatory/chunks/{chunk_id}` | Edit a canonical chunk and publish its updated index representations |
| GET | `/api/regulatory/chunks/{chunk_id}/revisions` | Existing canonical revision history |

Search accepts `{"query":"gümrük vergisi","mode":"bm25","offset":0,"limit":25}`. Modes are `bm25`, `semantic` and `hybrid`. It does not construct an LLM, expand the query, rerank with a model or generate an answer. Semantic and hybrid modes call the configured embedding model; BM25 does not. Search pages cover a bounded ranked candidate set, with `candidate_limit` and `total_is_exact:false` in the response. The returned count is not an exact count of every matching chunk in the corpus. Existing temporal and publication protections apply to indexed retrieval.

Each search result includes the canonical chunk and its file identity. “Dosyaya git” opens that file's full chunk list in the same screen. Files without canonical regulatory chunks remain visible in the catalogue with a zero count; this screen does not fabricate editable chunks from generic connector documents.

Editors submit the chunk's `updated_at` as `expected_updated_at`. A revision mismatch returns `409 CONFLICT` before preparing the correction; the editor retains its draft. Existing API clients may omit this optional precondition. Derived aggregate chunks remain read-only because their source chunks determine their contents. Files awaiting new-document review retain their dedicated review workflow.

For a file that has only reached `CHUNKED`, edits save its reviewed canonical text without publishing it. Use the file's “İndeksle” action to queue its first publication. The screen shows queued/running/failed progress and refreshes its status; saving alone does not claim that this unpublished file has been indexed.

Edits reuse publication ownership, canonical revision history and fenced Elasticsearch writes. A single logical chunk edit can refresh dependent representations within its file. The operation does not delete or recreate the corpus index, re-ingest unrelated files or replace stable canonical chunk IDs.
