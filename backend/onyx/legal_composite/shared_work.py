"""Coalesce identical work inside one authorized Legal Composite request."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from threading import BoundedSemaphore, Lock

from onyx.asv3.models import EvidenceItem, RunContext
from onyx.context.search.models import SearchDoc

CenterKey = tuple[str, int]
HydratedCenters = dict[CenterKey, list[EvidenceItem]]


class SharedCanonicalCenters:
    """Callers reauthorize and check prepared revision before subscribing.

    Only exact immutable retrieved originals are shared. Publication checks still
    run for each subscriber and all cited text is freshly verified before output.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._slots = BoundedSemaphore(8)
        self._reads: dict[str, Future[list[EvidenceItem]]] = {}

    @staticmethod
    def _key(doc: SearchDoc, revision: int | None) -> str:
        return json.dumps(
            [doc.document_id, doc.chunk_ind, doc.metadata, revision],
            ensure_ascii=False,
            sort_keys=True,
        )

    def read(
        self,
        docs: list[SearchDoc],
        context: RunContext,
        execute: Callable[[list[SearchDoc], RunContext], HydratedCenters],
        revisions: dict[str, int | None],
    ) -> HydratedCenters:
        owned: list[SearchDoc] = []
        futures: dict[CenterKey, Future[list[EvidenceItem]]] = {}
        with self._lock:
            for doc in docs:
                key = self._key(doc, revisions.get(doc.document_id))
                future = self._reads.get(key)
                if future is None:
                    future = Future()
                    self._reads[key] = future
                    owned.append(doc)
                futures[(doc.document_id, doc.chunk_ind)] = future
        try:
            groups: dict[str, list[SearchDoc]] = {}
            for doc in owned:
                groups.setdefault(doc.document_id, []).append(doc)
            for originals in groups.values():
                while not self._slots.acquire(timeout=0.05):
                    context.check_active()
                try:
                    context.check_active()
                    result = execute(originals, context)
                    for doc in originals:
                        pair = (doc.document_id, doc.chunk_ind)
                        futures[pair].set_result(
                            [
                                item.model_copy(deep=True)
                                for item in result.get(pair, [])
                            ]
                        )
                finally:
                    self._slots.release()
        except BaseException as error:
            with self._lock:
                for doc in owned:
                    future = futures[(doc.document_id, doc.chunk_ind)]
                    if not future.done():
                        future.set_exception(error)
                    self._reads.pop(
                        self._key(doc, revisions.get(doc.document_id)), None
                    )
            raise
        retained: HydratedCenters = {}
        for pair, future in futures.items():
            while True:
                context.check_active()
                try:
                    retained[pair] = [
                        item.model_copy(deep=True)
                        for item in future.result(timeout=0.05)
                    ]
                    break
                except FutureTimeout:
                    if future.done():
                        raise
        return retained
