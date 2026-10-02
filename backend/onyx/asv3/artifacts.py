"""Keep large artifact payloads once, outside model views and durable checkpoints."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from itertools import islice

from pydantic import JsonValue

from onyx.asv3.models import Artifact, RunContext

_BINARY_KEYS = {"base64", "bytes", "image_data", "binary", "data_uri", "png"}


def compact_json(value: JsonValue, *, max_chars: int = 12000) -> JsonValue:
    remaining = max_chars

    def visit(item: JsonValue, depth: int = 0) -> JsonValue:
        nonlocal remaining
        if remaining <= 0 or depth > 12:
            return {"truncated": True}
        if isinstance(item, str):
            if item.startswith("data:image/") or item.startswith("data:application/"):
                return {"artifact_payload_omitted": True}
            limit = min(remaining, 4000)
            remaining -= min(len(item), limit)
            return item[:limit] if len(item) <= limit else item[:limit] + "…"
        if isinstance(item, list):
            output: list[JsonValue] = []
            for child in item[:128]:
                if remaining <= 0:
                    break
                remaining -= 2
                output.append(visit(child, depth + 1))
            if len(output) < len(item):
                output.append({"truncated": True})
            return output
        if isinstance(item, dict):
            result: dict[str, JsonValue] = {}
            for key, child in islice(item.items(), 128):
                if key.lower() in _BINARY_KEYS:
                    result[key + "_omitted"] = True
                    continue
                if remaining <= 0:
                    result["truncated"] = True
                    break
                remaining -= len(key) + 4
                result[key] = visit(child, depth + 1)
            return result
        remaining -= 8
        return item

    return visit(value)


def artifact_reference(artifact: Artifact) -> Artifact:
    metadata = compact_json(artifact.metadata, max_chars=2000)
    assert isinstance(metadata, dict)
    return Artifact(
        artifact_id=artifact.artifact_id,
        name=artifact.name,
        media_type=artifact.media_type,
        source_ids=list(artifact.source_ids),
        metadata=metadata,
    )


class ArtifactStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, Artifact] = {}
        self._resident_sizes: dict[str, int] = {}

    def add(self, items: Iterable[Artifact], context: RunContext) -> list[Artifact]:
        references: list[Artifact] = []
        with self._lock:
            context.check_active()
            for artifact in items:
                old = self._items.get(artifact.artifact_id)
                if old is not None and (
                    old.media_type != artifact.media_type
                    or old.source_ids != artifact.source_ids
                ):
                    raise ValueError("Artifact identity changed")
                if old is None or artifact.artifact_id not in self._resident_sizes:
                    size = sum(
                        len(value)
                        for value in artifact.metadata.values()
                        if isinstance(value, str)
                    )
                    limit = context.budget.limits["artifact_bytes"]
                    if size > limit:
                        references.append(artifact_reference(artifact))
                        self._items[artifact.artifact_id] = artifact_reference(artifact)
                        continue
                    # Evict derived payloads only; authorized source/page locators remain
                    # addressable and can be reopened without discarding original evidence.
                    while context.budget.snapshot()["artifact_bytes"] + size > limit:
                        victim = next(iter(self._resident_sizes))
                        resident = self._resident_sizes.pop(victim)
                        self._items[victim] = artifact_reference(self._items[victim])
                        context.budget.release("artifact_bytes", resident)
                    context.budget.consume("artifact_bytes", size)
                    self._resident_sizes[artifact.artifact_id] = size
                    self._items[artifact.artifact_id] = artifact.model_copy(deep=False)
                references.append(artifact_reference(artifact))
        return references

    def get(self, artifact_id: str) -> Artifact | None:
        with self._lock:
            artifact = self._items.get(artifact_id)
            return artifact.model_copy(deep=False) if artifact else None

    def recent_images(self, max_images: int = 2) -> list[Artifact]:
        with self._lock:
            return [
                item.model_copy(deep=False)
                for item in reversed(list(self._items.values()))
                if item.media_type.startswith("image/")
                and isinstance(item.metadata.get("base64"), str)
            ][: min(2, max_images)]
