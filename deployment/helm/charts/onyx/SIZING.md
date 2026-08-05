# Sizing the Onyx chart

The chart's default resources are calibrated for a mid-size deployment
(roughly 200–1,000 users, up to ~2M documents) with everything at one
replica. This guide covers what to change as your deployment grows or
shrinks. Copy the snippets you need into **your own values file** — sizing
depends on which components you run in-cluster versus externally, so there is
no one-size preset. Elasticsearch is not a subchart: size its ECK
`Elasticsearch` resource separately from these Onyx values.

Rough tiers, matching the `size` input of the terraform module in
`deployment/terraform/modules/aws` (see its README):

| Tier | Users | Documents | Node pairing (terraform) |
|---|---|---|---|
| small | up to ~200 | < ~500k | 1× 8 vCPU / 32 GiB (m7i.2xlarge)¹ |
| medium | ~200–1,000 | ~0.5–2M | m7i.4xlarge ×1–5 |
| large | 1,000+ | multi-million | m7i.4xlarge ×2–8 + dedicated index node |

¹ Only with Postgres, Redis, and object storage external (RDS / ElastiCache
/ S3). Keeping those in-cluster too needs a second node.

## Principles (learned from production fleets)

- **Requests are for scheduling, limits are for bursts.** Steady-state usage
  is a fraction of the defaults' requests; what breaks deployments is almost
  always a *limit* (CPU throttling, OOM), not a request.
- **Scale the api-server out, not up.** Use the HPA with a CPU target only —
  idle api-server RSS (~1.1Gi) sits near the memory request, so a memory
  target pins the HPA at max.
- **Size Elasticsearch in ECK, not this chart.** Vector HNSW data and Lucene's
  page cache put pressure on memory outside the Java heap. Adjust the ECK
  node-set resources, JVM options, storage, and placement as the index grows.
- **A pegged indexing model server is not just slow.** An embedder stuck at
  its CPU limit for hours during a large re-index can starve docprocessing
  heartbeats and trip the indexing stall watchdog. Raise the indexing limit
  (and add a replica) before a planned bulk re-index.

## Small — trim for a single node

```yaml
webserver:
  replicaCount: 2  # keeps rolling deploys seamless

# Embedding traffic is light and bursty at this scale.
inferenceCapability:
  resources:
    requests: {cpu: 500m, memory: 3Gi}
    limits: {cpu: 3000m, memory: 10Gi}
indexCapability:
  resources:
    requests: {cpu: 500m, memory: 3Gi}
    limits: {cpu: 3000m, memory: 6Gi}

# Coordination singletons measure single-digit millicores in production.
celery_beat:
  resources:
    requests: {cpu: 250m, memory: 512Mi}
    limits: {cpu: 1000m, memory: 1Gi}
celery_worker_monitoring:
  resources:
    requests: {cpu: 250m, memory: 512Mi}
    limits: {cpu: 1000m, memory: 4Gi}
celery_worker_primary:
  resources:
    requests: {cpu: 250m, memory: 2Gi}
    limits: {cpu: 1000m, memory: 4Gi}

# Craft build-loop worker; set back to 1 if you use Craft/sandbox features.
celery_worker_scheduled_tasks:
  replicaCount: 0

```

With an external data plane this renders ~6.9 vCPU / ~22 GiB of requests —
it fits one 8 vCPU / 32 GiB node.

## Medium — the shape production deployments converge on

```yaml
webserver:
  replicaCount: 3

api:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 6
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: null  # see Principles

# 2 replicas so concurrent connector backfills don't starve each other and
# one bad fetch doesn't take out all ingestion.
celery_worker_docfetching:
  replicaCount: 2

```

## Large — org-wide, sustained re-indexes

```yaml
webserver:
  replicaCount: 3

api:
  # Generous CPU limit: bursty agent work (deep research, code interpreter)
  # exhausting the CFS quota stalls /health and causes liveness kills.
  resources:
    requests: {cpu: 1000m, memory: 2Gi}
    limits: {cpu: 4000m, memory: 8Gi}
  autoscaling:
    enabled: true
    minReplicas: 4
    maxReplicas: 12
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: null

# Two embedders with high CPU limits — see Principles on pegged embedders.
indexCapability:
  replicaCount: 2
  resources:
    requests: {cpu: 4000m, memory: 3Gi}
    limits: {cpu: 12000m, memory: 6Gi}

celery_worker_docfetching:
  replicaCount: 2
celery_worker_docprocessing:
  replicaCount: 2
  resources:
    requests: {cpu: 500m, memory: 2Gi}
    limits: {cpu: 1000m, memory: 18Gi}

# Metadata-sync throughput: one light worker drains ~800 tasks/min, which
# means multi-day backlogs after resyncs of multi-million-doc document sets.
celery_worker_light:
  replicaCount: 3

celery_worker_user_file_processing:
  replicaCount: 2
  resources:
    requests: {cpu: 500m, memory: 512Mi}
    limits: {cpu: 2000m, memory: 4Gi}

```

## Elasticsearch sizing

Configure CPU/memory requests, JVM heap, PVC size, replicas, and node placement
on the existing ECK `Elasticsearch` CR. The Onyx chart only needs the HTTP
service name, credentials Secret, and CA Secret. Set `elasticsearch.enabled`
to false only when search is intentionally disabled for an Onyx deployment.
