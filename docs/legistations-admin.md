# Legistations

The admin screen at `/admin/legistations` browses the tenant's uploaded files across owners, document sets and processing statuses. Its filename-derived folder tree is loaded independently from the current file page. File and chunk lists are paginated; original filenames and stable IDs are retained.

All endpoints below accept the normal session or an admin API key in `Authorization: Bearer <key>`. They require `FULL_ADMIN_PANEL_ACCESS`; ordinary user keys do not gain catalogue-wide access. Call them through the frontend's `/api` proxy.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/regulatory/catalog/files?q=&folder=&recursive=true&offset=0&limit=25` | Filename search and file pages; omit `folder` for all files |
| GET | `/api/regulatory/catalog/folders?parent=&q=&offset=0&limit=100` | Immediate folder children across the catalogue |
| GET | `/api/regulatory/catalog/files/{file_id}` | File metadata and canonical chunk count |
| POST | `/api/regulatory/catalog/files/{file_id}/index` | Queue initial indexing for one reviewed CHUNKED file |
| POST | `/api/regulatory/catalog/search` | Hybrid retrieval, reranking and file diversity |
| POST | `/api/regulatory/catalog/simple-search` | The same search with a factual Nemotron overview |
| GET | `/api/regulatory/catalog/files/{file_id}/chunks/{chunk_id}/location?limit=25` | Locate a canonical chunk in the full file's paginated ordering |
| GET | `/api/regulatory/files/{file_id}/chunks/page?offset=0&limit=25` | All canonical chunks of a selected file, including historical rows |
| PATCH | `/api/regulatory/chunks/{chunk_id}` | Edit a canonical chunk and publish its updated index representations |
| GET | `/api/regulatory/chunks/{chunk_id}/revisions` | Existing canonical revision history |

Search accepts `{"query":"gümrük vergisi"}`. Content search always uses hybrid retrieval with the configured embedding model, admitting at most 100 candidates. Valid, published canonical candidates are scored by the configured reranker before selecting at most 20 results, with a hard cap of eight chunks from any file. A smaller result set is returned if there are insufficient distinct-file candidates; the cap is never relaxed to fill the list. Reranker failure is an explicit search error, rather than an unranked fallback. The optional `mode` can only be `hybrid`; `offset` and `limit` paginate the final set and cannot increase its 20-result bound. Responses expose `candidate_limit:100`, `result_limit:20`, `per_file_limit:8` and `total_is_exact:false`. The returned count is not an exact count of every matching chunk in the corpus. Existing tenant, temporal and publication protections apply to indexed retrieval.

Each search result includes the canonical chunk and its file identity. The catalog groups results by stable file ID, initially showing only each file's name and matched-chunk count. Expanding a group previews its first three results. “Daha fazla göster” and “Dosyaya git” open that file's full chunk list, locating the selected match across pages and highlighting all returned matches for that file in yellow. Files without canonical regulatory chunks remain visible in the catalogue with a zero count; this screen does not fabricate editable chunks from generic connector documents.

The admin page `/admin/simple-search` sends `{"query":"gümrük vergisi"}` to `/api/regulatory/catalog/simple-search`. It uses the same final search set once, then passes the complete canonical text and source identities to `nvidia/nemotron-3-ultra-550b-a55b` (Nemotron 3 Ultra 550B A55B). The model must be visible on an accessible configured LLM provider; it is never replaced by the default chat model. NVIDIA identifies Ultra as its most capable Nemotron model: [official model overview](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/).

The overview uses one Nemotron call to select one to three short, verbatim source excerpts as strict JSON. The server validates the JSON shape, unique integer references and each quotation's continuous match against that exact referenced chunk, normalizing whitespace only. It renders the validated quotations with fixed Turkish attribution and a maximum of 120 words. It never invents linking prose, repairs a quotation, remaps a reference or makes a second model call to validate or retry a rejected selection. The prompt requires preserving relevant conditions, scope and exceptions and prohibits legal interpretation or advice. Treat the query and source content as untrusted data. Results remain explicit chunk cards with file-navigation buttons. The response contains `search`, `summary`, `summary_model` and `summary_error`: missing model configuration, malformed or unsupported excerpts, truncation and generation errors preserve the search results and show an error instead of an unsupported overview. An empty result set does not call the model. Publication visibility is rechecked before delivery after model generation.

Simple search also returns `search_query` and `query_rewritten`. Clear document lookups and short keyword searches go straight to retrieval. A descriptive request can make one bounded call to an accessible, configured inexpensive model to form one search query. There are no research loops or extra retrieval rounds; failed query preparation uses the original query. The final Nemotron excerpt selection still receives the original user request and the complete final chunk set. A changed search expression is shown in the UI.

Editors submit the chunk's `updated_at` as `expected_updated_at`. A revision mismatch returns `409 CONFLICT` before preparing the correction; the editor retains its draft. Existing API clients may omit this optional precondition. Derived aggregate chunks remain read-only because their source chunks determine their contents. Files awaiting new-document review retain their dedicated review workflow.

For a file that has only reached `CHUNKED`, edits save its reviewed canonical text without publishing it. Use the file's “İndeksle” action to queue its first publication. The screen shows queued/running/failed progress and refreshes its status; saving alone does not claim that this unpublished file has been indexed.

Edits reuse publication ownership, canonical revision history and fenced Elasticsearch writes. A single logical chunk edit can refresh dependent representations within its file. The operation does not delete or recreate the corpus index, re-ingest unrelated files or replace stable canonical chunk IDs.
