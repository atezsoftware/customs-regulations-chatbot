# Regulatory platform deployment

Production uses the parser-free `runtime-lite` deployment. Start with
[`REGULATORY_PRODUCTION_RUNBOOK.md`](REGULATORY_PRODUCTION_RUNBOOK.md); it is the authoritative
contract for image publication, infrastructure selection, preflight, migration, rollout, backup,
and verification.

Do not use generic installer scripts or build application images on the production host. Publish
the release from one clean revision on an approved build runner:

```bash
deployment/docker_compose/publish-regulatory-images.sh \
  registry.example.com/team \
  "$GIT_COMMIT_SHA" \
  cloud
```

Cloud is the production mode. It publishes immutable backend-lite, web, and one-shot importer
artifacts. Only the backend-lite and web digests go to the production application environment; the
importer digest stays on the authorized import workstation. No CUDA/model-server image is built or
promoted. API and background use the same backend-lite digest.

The production overlay sets `DISABLE_MODEL_SERVER=true` for API and background and keeps inference
and indexing model services inactive. Generate the allowlisted production handoff with
`build-regulatory-prod-bundle.sh`, then use the documented wrappers:

```bash
./regulatory-prod-lite-preflight.sh --help
./regulatory-prod-lite-deploy.sh --help
```

For a new, empty installation, deploy first and then select **OpenRouter** under
**Admin > Search Settings** before opening user traffic. The Admin enters the API key, fetches the
available embedding models, and selects one from the list. The application owns the endpoint and
detects the vector dimension automatically. Configure the chat model separately under
**Admin > Language Models**.

This initial activation is allowed without an indexer only while PostgreSQL has no documents,
non-default connectors, or completed user files. A populated environment requires reindexing from
the separate authorized import/indexing environment before the embedding model can change.

Parser-backed and bulk source indexing run separately from an authorized workstation or controlled
import runner. Production-lite may run only the documented Markdown durable-indexing path through its
dedicated worker and minimal Beat. Never place the importer image, importer environment, bulk source
files, CUDA, Docling, generic indexing workers, or the generic Beat on the production application host.

The production host requires explicit TLS, secret-store, database migration, backup,
infrastructure, and immutable-image inputs. Do not substitute mutable tags or ad hoc Compose command
chains for the documented release process.
