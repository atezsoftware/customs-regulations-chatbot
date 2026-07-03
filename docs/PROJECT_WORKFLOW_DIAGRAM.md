# Project workflow diagram

This diagram explains the project as two connected flows:

- Indexing flow: uploaded files are parsed, chunked, embedded, and stored in Postgres with pgvector.
- Chat flow: a user question first searches linked directories, sends the highest scoring chunks to the agent, and then iterates with additional search or document fetches when evidence is not enough.

```mermaid
flowchart LR
    subgraph userLayer ["User experience"]
        user(["User"])
        frontend["React frontend"]
        dashboard["Dashboard and admin views"]
    end

    subgraph backendLayer ["Backend API"]
        backend["LoopBack API"]
        auth["Auth, users, sessions"]
        chatBridge["Chat bridge"]
        indexBridge["Index bridge"]
        fileStorage[("Uploaded file storage")]
    end

    subgraph indexingLayer ["Indexing pipeline"]
        coreIndexer["Core indexer"]
        parser["Docling parser"]
        chunker["Regulatory chunker"]
        metadata["Metadata extraction"]
        embedder["Embedding generation"]
    end

    subgraph retrievalLayer ["Chat retrieval workflow"]
        question[/User question/]
        linkedDirs["Linked directory scope"]
        presearch["Semantic presearch"]
        topChunks{{"Top 4 scored chunks"}}
        agent["Core API agent"]
        enough{"Enough evidence?"}
        reference{"Document reference?"}
        fetchDoc["Fetch referenced document"]
        newSearch["Run another search"]
        answer[/Answer with citations/]
    end

    subgraph dataLayer ["Shared data"]
        appData[("Users, sessions, messages, usage")]
        coreData[("Documents, chunks, embeddings")]
        pgvector[("Postgres and pgvector")]
    end

    subgraph externalLayer ["External services"]
        gemini["Gemini LLM"]
        googleEmbed["Google embeddings"]
    end

    user -->|"Uses"| frontend
    frontend -->|"HTTPS API"| backend
    frontend -->|"Shows metrics"| dashboard
    backend -->|"Reads and writes"| auth
    backend -->|"Stores uploads"| fileStorage
    backend -->|"Persists app data"| appData

    backend -->|"Index requests"| indexBridge
    indexBridge -->|"Calls /api/index"| coreIndexer
    fileStorage -->|"Files"| coreIndexer
    coreIndexer -->|"Parses"| parser
    parser -->|"Markdown text"| chunker
    chunker -->|"Structured chunks"| metadata
    metadata -->|"Chunk records"| coreData
    metadata -->|"Optional extraction"| gemini
    embedder -->|"Writes vectors"| coreData
    coreIndexer -->|"Embeds chunks"| embedder
    embedder -->|"Embedding API"| googleEmbed
    coreData -->|"Stored in"| pgvector
    appData -->|"Stored in"| pgvector

    frontend -->|"Chat message"| question
    question -->|"POST message"| backend
    backend -->|"Streams via SSE"| frontend
    backend -->|"WebSocket explore"| chatBridge
    chatBridge -->|"Starts agent"| agent
    chatBridge -->|"Search /api/search"| presearch
    presearch -->|"Limit 4 per directory"| topChunks
    linkedDirs -->|"Restricts corpus"| presearch
    topChunks -->|"Evidence context"| agent
    agent -->|"LLM reasoning"| gemini
    agent --> enough
    enough -->|"Yes"| answer
    enough -->|"No"| reference
    reference -->|"Yes"| fetchDoc
    reference -->|"No"| newSearch
    fetchDoc -->|"get_document"| agent
    newSearch -->|"semantic_search"| presearch
    answer -->|"Sources and usage"| backend
    answer -->|"Rendered response"| frontend
    presearch -->|"Reads vectors and chunks"| coreData
    fetchDoc -->|"Reads full indexed document"| coreData
```

## Notes

- The backend scopes every chat to explicitly linked directories before search.
- The first retrieval pass sends the highest scoring chunk hits into the agent context.
- If the agent cannot answer from that context, it can call another indexed search.
- If the answer path mentions a specific document reference, the agent can fetch that document through `get_document`.
- Final answers are stored with messages, sources, research steps, and LLM usage.
