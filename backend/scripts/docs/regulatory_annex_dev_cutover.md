# DEV annex first cutover

This belongs to the existing backend-lite GitHub workflow. It does not deploy TEST
or production. Use the reviewed final commit on `develop`; image tag and workflow
checkout must be the same full SHA. The normal push still builds and deploys the
API before background (API startup applies Alembic). Both DEV deploy and manual
verification runs share a non-cancelling concurrency group.

## First deployment prerequisites

If actual physical identifiers are only reachable inside DEV, the fixed manual
`annex-inventory` action on `develop`, `environment=dev`, `image_tag=<exact SHA>`
uses the already-built image to emit only DEV physical names/UUIDs and the ES
cluster UUID, then deletes its temporary probe. No writer deployment, server task
inventory or activation occurs. An image built by the normal push is required;
a first push lacking scope evidence will safely stop after the image build,
before writer quiescence. Inventory does not prove cluster dedication.

Root/release owner must supply `ANNEX_DEV_ES_SCOPE_JSON` as a GitHub repository
variable before the first push. This is non-secret, reviewed operational evidence,
not an environment-name heuristic. Required fields:

```json
{
  "mode": "dev-dedicated",
  "cluster_uuid": "actual verified DEV-exclusive Elasticsearch cluster UUID",
  "indices": ["actual_sorted_DEV_PRESENT_and_optional_FUTURE_names"],
  "release_sha": "exact reviewed 40-character commit",
  "expires_at": 0,
  "evidence_ref": "reference to authoritative DEV exclusivity and controller inventory"
}
```

Set `expires_at` to a future Unix timestamp covering the bounded deployment. The
probe verifies the actual cluster UUID and current DEV database search settings;
these matches alone **do not establish cluster exclusivity**. The evidence must
establish that independently. Only this dedicated mode permits operational
cluster task/pending-metadata inventory. It never fetches document contents.

For a genuinely shared cluster, use `mode=shared-scoped`, plus
`producer_inventory_complete=true`, `index_lifecycle_idle=true`, and `task_ids`
containing only authoritatively identified DEV task IDs. Each identified task must
return `completed=true`; missing/unknown responses refuse. An empty list is valid
only with authoritative complete scoped producer/lifecycle evidence. Do not
invent an empty manifest or label a shared cluster dedicated to make deployment
pass. Unknown scope refuses before old writers are stopped. No shared/global task
listing is performed in shared mode.

The supported first-cutover namespace has exactly the known API/background/web
Deployments, one stable API/background replica, no HPA, Job, CronJob, DaemonSet,
StatefulSet, unknown pod or externally owned writer Deployment. Independently
confirm no external GitOps/operator or manual deployment can recreate an old
image during the serialized cutover. Unexpected topology requires review; the
script does not delete or disable unknown controllers.

## First-cutover order

1. Create one temporary, exact-new-SHA probe Pod from the background template,
   retaining Vault configuration but replacing the command with bounded sleep.
   It has no Service labels, owner controller, Supervisor or application writer.
2. In the probe, source `/vault/secrets/config` without tracing. A short read-only
   transaction verifies `customs-regulations-dev` and resolves exact PRESENT /
   FUTURE physical names. Inspect physical UUIDs; refuse aliases, data streams,
   ILM-managed indices, non-DEV names, existing write blocks or unsupported scope.
3. Persist the fixed `regulatory-annex-dev-cutover` ConfigMap. Scale API to zero
   and wait for normal deletion. No force deletion is permitted.
4. Stop both old Beat processes. For every known old worker, verify Supervisor
   PID against an exact Celery destination, cancel and acknowledge all its
   consumers, and wait for active/reserved/scheduled tasks and queues to be empty.
   Stop only drained workers, then scale background to zero. Never purge, revoke,
   or discard queue contents. Delayed tasks that cannot drain cause a refusal.
5. Recheck writer pods/controllers are zero, HPA absent and old nodes reachable.
   Drain relevant server-side write/metadata work using the authorized ES scope.
   Add the exact-index write block and require full index/shard acknowledgement.
   Recheck physical UUIDs and old writer absence. Partial responses retain state
   and any applied blocks; they never authorize deployment.
6. Remove the acknowledged blocks **while there are no old writers**, immediately
   before installing only the reviewed compatible API/background SHA. This order
   prevents queued ordinary tasks from executing against a blocked index when
   the new background starts. DEV rollback and forced deletion remain disabled.
7. Render runner-local Helm `app.environment.parameters` entries by name:
   `REGULATORY_ANNEX_WORKER_ENABLED=true`, `REGULATORY_ANNEX_ENVIRONMENT=dev`,
   `REGULATORY_ANNEX_UPDATES_ENABLED=false`. No separate database-identity override;
   runtime derives it from PG host/port/database. The worker switch defaults false
   in runtime-lite and remains true after protection, even if creation is toggled
   off, so approved jobs and recovery remain serviced. Background retains 4 GiB limit.
8. Require exact-SHA Ready API/background pods, no restarts, and the real annex
   readiness module (Supervisor PID, exact three queues, four handlers,
   concurrency one, database scope). Persist `released` and delete the probe.

Known subsequent compatible releases use the retained `released` SHA to validate
existing writer images and then follow ordinary compatible rolling deployment.
They do not repeat the first-cutover barrier or require fresh ES-scope evidence.
An unrecognized writer image or unfinished cutover refuses automatic deployment.

## Fixed runner invocations

The workflow sets `env_x=dev`, `GITHUB_REF=refs/heads/develop`, `GITHUB_SHA`, and
`IMAGE_TAG` (the latter two must match). In the existing runner checkout:

```sh
python3 backend/scripts/regulatory_annex_dev_cutover.py prepare
python3 backend/scripts/regulatory_annex_dev_cutover.py values
# Existing API then background Helm deployment and normal worker health checks.
python3 backend/scripts/regulatory_annex_dev_cutover.py release
```

For final release verification, dispatch this **same backend workflow** on
`develop`, `environment=dev`, `image_tag=<exact current SHA>`, with action
`annex-verify` or `annex-activate`. These actions do not build another image or
accept shell/Python code as input. They require both backend and web's latest
matching successful `develop`/`push` workflow runs, exact-SHA Ready API/background/
web pods, annex readiness, and public-frontend `/api/health`.

Task6 must provide the repository-owned fixed module:

```sh
python -m onyx.regulatory.amendments.annexes.dev_acceptance preflight
python -m onyx.regulatory.amendments.annexes.dev_acceptance canary
```

The runner sources Vault in the exact background container before these calls.
`preflight` must prove the bounded parser/provider/fixture requirements while
creation is false. `annex-verify` stops after preflight. `annex-activate` first
passes preflight, Helm-upgrades the **same SHA** with creation true, rechecks
readiness/frontend, then calls `canary`. Task6 owns fixed fixtures, authenticated
frontend-routed job/approval/chat acceptance, bounded timeout, idempotence and
allowlisted output. Missing module or failed probe refuses activation. No ignored
local helper is copied or assumed present in the image.

## Failure and retained state

After first-cutover mutation, failures retain the ConfigMap and the probe (the
probe automatically expires after two hours; blocks/state do not). New protected
writes are never followed by automatic restoration of an unfenced image. Failed
Helm/readiness calls attempt to quiesce writers through the same drain path;
if a compatible worker is unhealthy, it remains on the reviewed binary and the
workflow reports failure. No automatic unsafe Helm rollback or forced pod deletion
runs in DEV. Non-DEV deployment/rollback behavior is unchanged.

A failed activation attempts to disable creation by upgrading the same reviewed
image with the flag false. It does not reverse protected data. If that recovery
also fails, retain the compatible image and investigate; do not restore old code.

An unfinished first-cutover state deliberately requires root's reviewed recovery.
Inspect only this DEV ConfigMap and known DEV resources; retain acknowledged index
blocks until the writer/server-work proof can be completed. Do not delete the
ConfigMap to bypass its guard, force-delete old pods, purge queues or roll back to
an old image. There is no automatic drain/reclaim service or arbitrary remote-exec
recovery interface. Root must collect actual DEV topology/drain/task/UUID evidence;
hermetic tests are not deployment acceptance.
