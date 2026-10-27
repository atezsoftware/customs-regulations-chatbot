"""Run unchanged frozen intake embedding requests with bounded concurrency."""

import json
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import Context, copy_context
from dataclasses import dataclass

from onyx.regulatory.indexing_jobs.embedding import _validate_response_vectors

MAX_CONCURRENT_REQUESTS = 2
MAX_REQUEST_BYTES = 256 * 1024
_REQUEST_ENVELOPE_RESERVE = 1024


@dataclass(frozen=True)
class FrozenEmbeddingGroup:
    key: str
    texts: tuple[str, ...]


def encode_intake_embedding_groups(
    groups: Sequence[FrozenEmbeddingGroup],
    *,
    encode: Callable[[list[str]], Sequence[object]],
    expected_dimension: int,
    check_owner: Callable[[], None],
) -> dict[str, list[list[float]]]:
    """Keep each complete provider request intact and return only after all succeed."""
    if len({group.key for group in groups}) != len(groups):
        raise ValueError("Intake embedding groups must have unique frozen identities")
    for group in groups:
        if not group.key or not group.texts:
            raise ValueError("Intake embedding groups require their complete inputs")
        for text in group.texts:
            if not isinstance(text, str) or not text:
                raise ValueError("Intake embedding inputs cannot be empty")
        if (
            len(json.dumps(group.texts, ensure_ascii=False).encode())
            > MAX_REQUEST_BYTES - _REQUEST_ENVELOPE_RESERVE
        ):
            raise ValueError(
                "A frozen intake embedding request exceeds the request byte budget"
            )

    def encode_group(group: FrozenEmbeddingGroup) -> list[list[float]]:
        return _validate_response_vectors(
            encode(list(group.texts)),
            expected_count=len(group.texts),
            expected_dimension=expected_dimension,
        )

    def encode_in_context(
        group: FrozenEmbeddingGroup, context: Context
    ) -> list[list[float]]:
        return context.run(encode_group, group)

    vectors: dict[str, list[list[float]]] = {}
    futures: dict[Future[list[list[float]]], FrozenEmbeddingGroup] = {}
    pending = iter(groups)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        try:
            while True:
                while len(futures) < MAX_CONCURRENT_REQUESTS:
                    # Observe already acknowledged failures before dispatching more.
                    if any(future.done() for future in futures):
                        break
                    group = next(pending, None)
                    if group is None:
                        break
                    check_owner()
                    future = executor.submit(encode_in_context, group, copy_context())
                    futures[future] = group
                    check_owner()
                if not futures:
                    break
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                check_owner()
                for future in completed:
                    group = futures[future]
                    vectors[group.key] = future.result()
                check_owner()
                for future in completed:
                    del futures[future]
        except BaseException:
            for future in futures:
                future.cancel()
            # The executor drains in-flight acknowledgements before propagating.
            raise
    return {group.key: vectors[group.key] for group in groups}
