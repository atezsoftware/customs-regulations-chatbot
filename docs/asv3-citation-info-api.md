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
