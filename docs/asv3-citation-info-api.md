# ASv3 citation titles and exact source text

ASv3 Tek (`atez_search_v3: true`, `asv3_research_profile: "normal"`, nonparallel)
and Exp Cropped (`asv3_exp_cropped: true`) provide optional presentation fields
on each `citation_info` record:

| Field | Meaning |
| --- | --- |
| `citation_number` | The number printed in this answer, for example `[10]`. |
| `document_id` | The source file identity; several citations can share it. |
| `chunk_ind` | The saved chunk ordinal, including negative native locators. |
| `chunk_id` | The exact canonical chunk identity, or `null` for native-file evidence. |
| `display_title` | The nonblank `legistation_metadata.official_title`, with the existing source-name fallback. |
| `text` | The complete retained original supporting this citation, including its whitespace. |
| `preview_url` | The existing authenticated exact-source endpoint, retained as a fallback. |

Existing `semantic_identifier`, source identities, citation numbers and preview
URLs retain their previous meanings. In particular, `semantic_identifier` can
still be a stored filename; clients should prefer `display_title` for the label.

## Non-streaming response

The following shortened example illustrates the shape. IDs, title and passages
are illustrative. Other existing response fields are omitted.

```json
{
  "message_id": 6200,
  "answer": "İlk koşul [10], istisna ise [11].",
  "citation_info": [
    {
      "type": "citation_info",
      "citation_number": 10,
      "document_id": "7d839929-453a-4696-9d6f-e2b4ec995416",
      "chunk_ind": 7,
      "chunk_id": "a572ced3-13dc-44f9-8742-8d92e85a7913",
      "semantic_identifier": "Tır ve Transit Mevzuatı/stored_name.md",
      "display_title": "9366 Sayılı Karar — Tam Resmî Başlık",
      "text": "Birinci kaynak parçasının özgün metni.\n",
      "source_type": "file",
      "preview_url": "/api/asv3/citation/6200/10"
    },
    {
      "type": "citation_info",
      "citation_number": 11,
      "document_id": "7d839929-453a-4696-9d6f-e2b4ec995416",
      "chunk_ind": 8,
      "chunk_id": "b572ced3-13dc-44f9-8742-8d92e85a7913",
      "semantic_identifier": "Tır ve Transit Mevzuatı/stored_name.md",
      "display_title": "9366 Sayılı Karar — Tam Resmî Başlık",
      "text": "İkinci kaynak parçasının özgün metni.\n",
      "source_type": "file",
      "preview_url": "/api/asv3/citation/6200/11"
    }
  ]
}
```

`[10]` opens only the first record's `text`; `[11]` opens only the second.
The shared `document_id` does not combine these records.

## Streaming and history

Streaming uses the same object inside the existing packet envelope:

```json
{
  "placement": {"turn_index": 4},
  "obj": {
    "type": "citation_info",
    "citation_number": 10,
    "document_id": "7d839929-453a-4696-9d6f-e2b4ec995416",
    "chunk_ind": 7,
    "chunk_id": "a572ced3-13dc-44f9-8742-8d92e85a7913",
    "semantic_identifier": "Tır ve Transit Mevzuatı/stored_name.md",
    "display_title": "9366 Sayılı Karar — Tam Resmî Başlık",
    "text": "Birinci kaynak parçasının özgün metni.\n",
    "source_type": "file",
    "preview_url": "/api/asv3/citation/6200/10"
  }
}
```

On `GET /api/chat/get-chat-session/{session_id}`, each message now also has
`messages[].citation_info`. The legacy `messages[].citations` map is unchanged:

```json
{
  "message_id": 6200,
  "citations": {
    "10": "7d839929-453a-4696-9d6f-e2b4ec995416",
    "11": "7d839929-453a-4696-9d6f-e2b4ec995416"
  },
  "citation_info": [
    {
      "type": "citation_info",
      "citation_number": 10,
      "document_id": "7d839929-453a-4696-9d6f-e2b4ec995416",
      "chunk_ind": 7,
      "chunk_id": "a572ced3-13dc-44f9-8742-8d92e85a7913",
      "display_title": "9366 Sayılı Karar — Tam Resmî Başlık",
      "text": "Birinci kaynak parçasının özgün metni.\n",
      "preview_url": "/api/asv3/citation/6200/10"
    }
  ]
}
```

Keep citations keyed by `(message_id, citation_number)`, not by file ID, title,
source-panel order, or response array index. Keep separate numbers even when
they happen to target the same chunk. The history array and replay packets use
the same enriched citation objects.

## External frontend changes

1. Retain the three optional fields `display_title`, `chunk_id` and `text` from
   incoming citation packets and history JSON.
2. Find the record by `citation_number` when the user clicks `[10]`.
3. Prefer `display_title` and render `text` directly. A string, including an empty
   string, is supplied content; test for `null`/absence instead of truthiness.
4. If `text` is absent, preserve the existing authenticated preview fallback.
   Resolve a relative `preview_url` against the application's configured backend
   or API proxy and forward its normal authentication.

```typescript
const target = message.citation_info?.find(
  (item) => item.citation_number === clickedNumber
);

const title = target?.display_title ?? target?.semantic_identifier ?? "Kaynak";
if (target && typeof target.text === "string") {
  showSource({ title, text: target.text });
} else if (target?.preview_url) {
  const original = await authenticatedApiGet(target.preview_url);
  showSource({ title, text: original.content });
} else {
  showLegacyCitation(message.message_id, clickedNumber);
}
```

`showSource`, `authenticatedApiGet`, and `showLegacyCitation` stand for the
client's existing rendering and request functions. Render titles as text,
not HTML. The legacy handler must not guess a chunk by collecting every
`context_docs` row whose `document_id` matches.

## Compatibility and access

- No client migration is required to continue reading the old fields; the new
  fields and history array are additive.
- Existing owned ASv3 Tek/Exp Cropped history is enriched from its saved citation
  ledger and current file titles without rewriting old messages or sources.
- History text is added only after the owning user and current source access,
  publication, exact chunk/hash and saved ordinal are verified. Shared or
  anonymous viewers retain their previous response without new inline text.
- Missing legacy checkpoints, stale source identities, unavailable sources or a
  verification deadline leave the optional fields unavailable; they never cause
  a same-file chunk to be substituted.
- The preview endpoint remains available and applies its existing access checks.
- Other workflows keep their existing citation behavior. Source text and titles
  are presentation data; they are not added to LLM prompts or the next request's
  research context.

## Admin Query History

`GET /api/admin/chat-session-history/{session_id}` exposes the same additive
`messages[].citation_info` contract. The existing `documents` list remains the
retrieval inventory; it is not the list or numbering of cited passages.

Query History citation records also provide optional presentation fields:

| Field | Meaning |
| --- | --- |
| `link` | The recorded original source URL, when available. |
| `reference` | The recorded heading or passage location, separate from the official file title. |
| `resolution_status` | `retained`, `current`, `external`, or `unavailable`. |

`retained` identifies a saved original; opening it still checks current access,
publication and exact identity. `current` identifies an indexed chunk without a
verified retained passage; clients must distinguish it from the historical text.
`external` opens the recorded HTTP(S) source link. `unavailable` keeps the citation
visible without substituting another chunk. Missing saved mappings appear in
`messages[].unresolved_citations`, a list of citation numbers, rather than gaining
invented document or chunk identities.

The admin citation `preview_url` has this shape:

```text
/api/admin/chat-session-history/{session_id}/citation/{message_id}/{citation_number}
```

It requires `FULL_ADMIN_PANEL_ACCESS`, enabled Query History, the exact
session/message/citation binding and current source access. Its response retains
the existing `ChunkInfo` shape (`content`, `num_tokens`). It does not relax the
owner checks on `/api/asv3/citation/...` or add inline text to shared/anonymous
history. Query History may load passage text on demand; supported normal owned
ASv3 responses continue to provide inline `text` as before.

In every view, keep the number and official file title, and show `reference`
separately. Two citations may deliberately share an official title or chunk.
Selection/copy must preserve their original citation numbers and links. An empty
supplied `text` remains supplied content; it must not trigger replacement fetches.

## Metadata-only source notices

The separate `regulatory_source_notice` catalogue does not contain original
passages. A `notice:<UUID>` or name-only `catalog:` observation must not become
`citation_info`, a numbered legal citation or a canonical preview target. Its
repeal annotation does not supply a missing article or an unknown repeal date.
Only an ordinarily acquired and validated real source can supply a citation.
See [Metadata-only regulatory source notices](regulatory-source-notices.md).

## Reading citations in an external API client

`POST /api/chat/send-chat-message` retains each full `citation_info` object in
both the streamed packets and the `stream: false` JSON response. Each record now
also carries the recorded `reference` (for example, `Gümrük Kanunu > MADDE 241`)
and source `link` when available. These location fields do not require a research
profile change. The existing eligibility and access checks for inline `text` and
official `display_title` remain unchanged.

For `GET /api/chat/get-chat-session/{session_id}`, every resolved saved citation
also appears in `messages[].citation_info`, including workflows without inline
text enrichment. The JSON records and replay packets use the same exact saved
document/chunk targets and first-use order. If enrichment is unavailable, basic
targets remain visible without gaining private source text. Content-stripped and
publication-unavailable responses continue to omit citation details.

External clients must render one entry per `(message_id, citation_number)`.
`display_title` identifies the file; `reference` identifies the recorded passage
location. Several entries may have the same title or `document_id`. Neither
`top_documents` nor `context_docs` is the numbered citation list: they include
retrieved documents and must not be used to infer citation numbers or select a
same-file replacement. For instance, `[97]` can point to article 241 while
`[250]` points to article 242 of the same file. Citation numbers are not article
numbers.

Use the record's supplied `text`, including an empty string, or its authenticated
`preview_url`. A missing `reference` means no usable location was recorded; it
does not permit guessing an article from the citation number or another chunk.
These changes do not broaden the separately restricted external-share projection
at `/api/chat/shared/{share_token}/messages`.
