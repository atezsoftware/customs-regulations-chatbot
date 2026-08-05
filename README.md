# Atez Customs Assistant

Atez Customs Assistant is a self-hosted AI assistant for customs and trade knowledge. It answers
questions strictly from the documents you connect to it — regulations, tariff schedules, internal
procedures, correspondence — with inline citations back to the source.

Everything runs on infrastructure you control. User accounts, chat history, documents, and search
indexes all live in your own PostgreSQL, Elasticsearch, and object storage. No usage data is sent
anywhere outside the deployment.

---

## Features

- **Grounded answers:** Every factual claim is traceable to a retrieved document, an attached file,
  or something the user said. When the sources don't cover a question, the assistant says so
  instead of guessing. See [Grounding](#grounding) below.
- **Agentic RAG:** Hybrid keyword + vector retrieval driven by an agent loop.
- **Deep Research:** Multi-step research flow for questions that need several rounds of retrieval.
- **Custom Agents:** Agents with their own instructions, knowledge scope, and actions.
- **Connectors:** 50+ indexing connectors out of the box, plus MCP.
- **Actions & MCP:** Let agents call external systems, with flexible auth.
- **Code Execution:** Run code in a sandbox to analyze data or render charts.
- **Artifacts:** Generate documents, graphics, and other downloadable output.
- **Web Search:** Optional, via Serper, Google PSE, Brave, SearXNG, Firecrawl, or Exa.

Works with any major LLM provider, self-hosted (Ollama, LiteLLM, vLLM) or hosted (Anthropic,
OpenAI, Gemini).

---

## Grounding

The assistant is configured to avoid hallucination rather than to sound confident:

- A grounding block is appended to **every** system prompt — the default one, an admin-edited one,
  and any custom agent's own prompt — so it cannot be removed by editing prompt text in the admin
  UI. It lives in `GROUNDING_GUIDANCE` in
  [`backend/onyx/prompts/chat_prompts.py`](backend/onyx/prompts/chat_prompts.py).
- Identifiers, figures, dates, and regulation numbers must be reproduced verbatim from the source.
- Conflicting sources are surfaced as a conflict rather than silently merged.
- Missing information is reported as missing, with a note on what is needed.
- Default sampling temperature is **0**, on both the backend (`GEN_AI_TEMPERATURE`) and the chat UI.

---

## Deployment

Two deployment modes are supported:

**Lite** — a lightweight chat UI. Under 1GB of memory, no vector index. Good for evaluating the
chat and agent functionality.

**Standard** — the full stack, recommended for real use:

- Vector + keyword index for RAG
- Background workers for connector syncing and indexing
- Model inference servers for embedding and reranking
- Redis cache and MinIO blob storage

Compose files live in [`deployment/docker_compose/`](deployment/docker_compose/). Kubernetes and
Helm manifests are under [`deployment/`](deployment/).

### Data residency

- User accounts and sessions: local PostgreSQL, via `fastapi-users` (`AUTH_TYPE=basic` by default).
- Documents and search indexes: local Elasticsearch and MinIO.
- Anonymous telemetry: **off** by default. Set `DISABLE_TELEMETRY=false` to opt back in.
- Product analytics (PostHog): off unless `POSTHOG_API_KEY` is set.

---

## Development

See [`AGENTS.md`](AGENTS.md) for the repository layout and per-area standards
([`backend/AGENTS.md`](backend/AGENTS.md), [`web/AGENTS.md`](web/AGENTS.md),
[`mobile/AGENTS.md`](mobile/AGENTS.md)), and [`CONTRIBUTING.md`](CONTRIBUTING.md) for engineering
best practices.

## Licensing

Built on [Onyx](https://github.com/onyx-dot-app/onyx). The core is MIT licensed; the `ee/`
directories carry the upstream Enterprise Edition license. See [`LICENSE`](LICENSE).
