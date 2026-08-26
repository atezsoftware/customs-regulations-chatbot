# Amendment Source Inputs Design

## Context

The admin `Updates` page currently accepts only pasted text. It sends a
`document_set_id` and `raw_text` to the synchronous amendment-analysis API,
which segments the text, finds candidate regulatory chunks, drafts proposals,
and waits for an administrator to approve or reject each proposal.

Administrators also need to start the same workflow from:

- a Resmî Gazete document URL such as
  `https://www.resmigazete.gov.tr/eskiler/2026/08/20260826-2.htm`; or
- a PDF selected from their computer.

Both sources must be normalized into reviewable text before the existing
analysis pipeline runs. Source extraction must never make an amendment to a
document set by itself.

## Goals

- Add Text, URL, and PDF input modes to the Updates page.
- Fetch public HTTP(S) HTML or PDF URLs safely on the backend.
- Extract the main readable text from Resmî Gazete-style HTML documents.
- Extract text from uploaded PDFs with the repository's existing PDF parser.
- Show extracted text in the existing editable text area before analysis.
- Reuse the existing amendment-analysis request after extraction.
- Reject empty, binary-looking, oversized, or otherwise unusable source text.
- Preserve document-set scoping even if the matching LLM returns an unexpected
  chunk identifier.

## Non-goals

- General-purpose web crawling or recursive link following.
- Indexing the submitted update source as a new document-set file.
- OCR or vision processing for image-only PDFs in the first version.
- Persisting source URLs, filenames, hashes, or extraction metadata in new
  database columns.
- Background-job conversion of the existing synchronous amendment-analysis
  endpoint.

## User Experience

The Updates page keeps the Document Set selector and adds an input-mode control
with three options:

1. **Text** — the existing paste-and-analyze behavior.
2. **URL** — a URL field and an `Extract` action.
3. **PDF** — a PDF file picker and an `Extract` action.

For URL and PDF modes, successful extraction places the normalized result in
the same editable amendment-text area used by Text mode. The source control
shows the selected URL or filename, while the administrator may correct
extraction artifacts before selecting `Analyze`.

Changing the source URL or selected PDF marks the previous extraction stale and
prevents it from being analyzed as if it belonged to the new source. Switching
to Text mode retains normal editable-text behavior.

Extraction failures appear as actionable messages. Important cases include:

- unreachable URL or HTTP failure;
- URL resolving to a blocked private, loopback, link-local, or metadata host;
- unsupported response type;
- invalid or oversized PDF;
- encrypted PDF without an available password;
- PDF containing no useful embedded text; and
- extracted text exceeding the analysis input limit.

An image-only PDF is rejected with guidance to upload a text-searchable PDF or
paste OCR output. It is never silently sent to the amendment LLM as empty or
near-empty text.

## Architecture

### Source extraction service

A focused backend module under `onyx/regulatory/amendments` owns normalization:

- `extract_amendment_html(content: bytes, content_type: str | None) -> str`
- `extract_amendment_pdf(content: bytes, file_name: str) -> str`
- `fetch_and_extract_amendment_url(url: str) -> AmendmentSourceExtraction`

`AmendmentSourceExtraction` contains the normalized `text`, detected
`source_type` (`html` or `pdf`), and a display name. These values are returned
to the frontend but are not persisted in this version.

The service has no database or amendment-writing responsibilities. Its output
is only input to the existing preview and analysis path.

### HTML extraction

HTML bytes are decoded with the repository's `decode_html_bytes`, then parsed
with Beautiful Soup. Scripts, styles, templates, navigation, and other obvious
non-content elements are removed. Main-content candidates are preferred in
this order:

1. `main` or `article` content;
2. the document body when no semantic container exists.

Text is emitted with preserved block boundaries, normalized whitespace, and no
page chrome. Extraction is site-agnostic enough to handle legacy Resmî Gazete
HTML while remaining limited to a single supplied page.

### PDF extraction

PDF input is validated using both MIME/filename hints and the `%PDF-` file
signature. Text extraction reuses `extract_file_text` with a `.pdf` extension,
which uses the configured Unstructured service when available and otherwise
falls back to the repository's isolated PDFium/pypdf implementation.

The first version does not extract embedded images or invoke a vision model.
A minimum useful-text check rejects scanned PDFs whose extracted content is
empty or effectively unusable.

### URL fetching and security

Only `http` and `https` URLs are accepted. The backend uses `ssrf_safe_get`,
which validates DNS targets and each redirect hop. Private/internal,
loopback, link-local, metadata, credential-bearing, and unsupported-scheme
URLs remain blocked.

The response is streamed with separate connection/read timeouts and an
application-level byte limit. The server does not trust `Content-Length` by
itself. Extraction type is selected from the final URL, response content type,
and initial byte signature. HTML and PDF are the only accepted types.

Requests use the repository's normal browser-like HTTP headers so public
government sites that reject default Python clients have the best chance of
responding. A remote site's timeout or outage is surfaced as an extraction
error; it is not retried indefinitely.

### API

Two admin-only endpoints keep request contracts explicit:

- `POST /regulatory/amendments/sources/url`
  - JSON: `{ "url": "https://..." }`
  - returns the source extraction snapshot.
- `POST /regulatory/amendments/sources/pdf`
  - multipart form with one `file` field;
  - returns the source extraction snapshot.

Both require `FULL_ADMIN_PANEL_ACCESS`. They do not require a document-set ID
because extraction cannot read or mutate document-set data. Analysis retains
its existing editable-document-set authorization.

Pydantic request/response models define the JSON contract. Content errors are
returned as invalid-input errors; outbound network failures use a concise,
non-sensitive message and do not echo response bodies.

### Limits

Limits are named application constants so deployments can tune them without
changing the workflow:

- maximum downloaded/uploaded source bytes;
- maximum extracted characters accepted for preview/analysis; and
- minimum useful extracted characters for a PDF.

The same text-length validation is also applied to directly pasted text in the
analysis request, preventing source mode from bypassing the LLM input bound.

## Amendment Matching Safety

Candidate search is already scoped to the selected document set, but the
matching LLM's returned `old_chunk_id` is currently looked up in the database
without first proving that it belongs to the provided candidate list.

The pipeline will build the allowed candidate-ID set and reject any non-null ID
outside it. An unexpected ID becomes an unmatched instruction; it must not be
reinterpreted as a new article. A null match is accepted as a new provision
only when the segmented instruction explicitly represents an addition.

Approval also verifies that the proposal's target/new user-file ID belongs to
the batch's document set before writing or projecting a regulatory chunk. This
is a defense-in-depth authorization invariant rather than a prompt instruction.

## Frontend Data Flow

The frontend regulatory service gains typed URL/PDF extraction functions. The
page maintains:

- selected mode;
- URL or PDF selection;
- extraction loading/error state;
- extracted source identity; and
- editable `rawText`.

URL/PDF extraction is a separate action from analysis. `Analyze` remains
disabled until non-whitespace preview text exists and the currently selected
source has been successfully extracted. Existing proposal review, approval,
rejection, and batch history behavior stays unchanged.

## Testing

### Backend unit tests

- HTML extraction keeps amendment content and removes script/style/navigation
  content.
- Legacy Turkish encodings decode without mojibake.
- PDF extraction accepts a valid text PDF and rejects empty output.
- URL extraction accepts HTML and PDF detected by MIME, URL, or signature.
- Streaming download stops at the configured byte limit.
- Unsupported content types and HTTP failures return safe errors.
- SSRF validation is invoked and redirect safety is inherited from
  `ssrf_safe_get`.
- Direct analysis rejects oversized text.
- A matcher result outside the candidate list becomes unmatched and never
  creates a proposal.
- Approval rejects a proposal whose target file is outside the batch document
  set.

### Frontend tests

- URL extraction sends the documented JSON request.
- PDF extraction sends multipart data without manually setting Content-Type.
- Extracted text appears in the editable preview.
- Analyze is disabled while extraction is stale or in progress.
- Switching modes does not accidentally analyze a prior URL/PDF extraction.
- Existing pasted-text analysis remains functional.

### Verification

- Run focused backend and frontend tests first.
- Run backend formatting/type checks for touched Python files.
- Run frontend formatting, TypeScript, and focused Jest checks for touched
  TypeScript files.
- Manually verify Text, URL, and PDF modes against the local admin page when an
  in-app browser is available.

## Operational Behavior

The existing analysis endpoint remains synchronous. Source extraction is also
synchronous but bounded by byte and timeout limits. No worker restart is needed
because this design does not add Celery work.

The external Resmî Gazete service can still be unavailable or rate-limit the
application. Such failures are reported clearly and leave the editable text
and all document-set data unchanged.
