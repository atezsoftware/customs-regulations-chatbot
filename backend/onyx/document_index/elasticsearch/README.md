# Elasticsearch retrieval

Onyx uses Elasticsearch for both lexical and vector retrieval. Vectors are
stored in indexed `dense_vector` fields and searched with HNSW kNN queries.

## Hybrid search

The default hybrid query uses Elasticsearch's native `linear` retriever. Its
lexical and kNN child retrievers are normalized independently with `minmax`,
multiplied by the configured Onyx weights, and merged over a shared candidate
window. This feature requires an Elasticsearch license that includes the linear
retriever; Onyx deliberately does not disable it or silently fall back to a
lower-quality query. The default content search keeps the existing 50/50
lexical/vector weighting; the alternate search also keeps the existing
title-vector boost.

When `HYBRID_SEARCH_NORMALIZATION_METHOD=2` is selected, Onyx applies z-score
normalization to the same result windows and combines them with the same
weights. This fusion stays application-side because Elasticsearch's linear
retriever does not provide a z-score normalizer.

Filters are applied to every child retriever. Highlighting and the selected
source fields are retained when result sets are fused.

## Ranking refinements

Embedding score distributions vary by model and query, so time decay and other
business boosts are applied after the initial Elasticsearch candidate retrieval.
This avoids applying a fixed boost to incomparable raw lexical and vector score
ranges. The Elasticsearch candidate window is intentionally larger than the
number of results returned to leave room for these refinements and reranking.

## Cluster assumptions

The client supports self-managed Elasticsearch over HTTP or HTTPS with basic
authentication. Kubernetes deployments are expected to connect to an existing
Elasticsearch service, normally one reconciled by ECK. The Helm defaults use
the standard service, password Secret, and public HTTP certificate Secret names
for an ECK resource named `elasticsearch`; all of them are configurable.
