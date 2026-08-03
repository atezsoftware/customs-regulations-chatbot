# Regulatory platform deployment

Production uses the parser-free `runtime-lite` deployment. Start with
[`REGULATORY_PRODUCTION_RUNBOOK.md`](REGULATORY_PRODUCTION_RUNBOOK.md); it is the authoritative
contract for image publication, infrastructure selection, preflight, migration, rollout, backup,
and verification.

Do not use generic installer scripts or build application images on the production host. Publish
the backend-lite, web, model-server, and importer images from one clean revision on an approved
build runner:

```bash
deployment/docker_compose/publish-regulatory-images.sh \
  registry.example.com/team \
  "$GIT_COMMIT_SHA"
```

Production receives digest-pinned backend-lite, web, and model-server images. It does not receive
the importer image, source documents, parser dependencies, or generic indexing workers. Generate
the allowlisted production handoff with `build-regulatory-prod-bundle.sh`, then run the documented
preflight and deployment wrappers:

```bash
./regulatory-prod-lite-preflight.sh --help
./regulatory-prod-lite-deploy.sh --help
```

Source-document parsing and indexing run separately from an authorized workstation or controlled
import runner. Use `regulatory-import-run.sh` with the matching importer digest and follow the
network, credential, manifest, and tenant requirements in the production runbook. Never place the
importer image, importer environment, or source files on the production application host.

The production host requires explicit TLS, secret-store, database migration, backup, infrastructure,
and immutable-image inputs. Do not substitute mutable tags or ad hoc Compose command chains for the
documented release process.
