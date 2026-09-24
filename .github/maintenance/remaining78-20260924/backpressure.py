"""Bound maintenance requests and retry only explicit transient ES rejection."""

import json
import time
from collections.abc import Callable
from typing import Any

from elasticsearch import ApiError


class GuardedClient:
    def __init__(
        self, client: Any, check_owner: Callable[[], object] = lambda: None
    ) -> None:
        self.client = client
        self.check_owner = check_owner

    def retry(self, call: Callable[..., Any], **kwargs: Any) -> Any:
        for attempt in range(7):
            self.check_owner()
            try:
                return call(**kwargs)
            except ApiError as error:
                if error.status_code not in (429, 502, 503, 504) or attempt == 6:
                    raise
                time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.client, name)
        if name == "indices":
            return GuardedClient(target, self.check_owner)
        if callable(target):
            return lambda **kwargs: self.retry(target, **kwargs)
        return target

    def options(self, **kwargs: Any) -> "GuardedClient":
        return GuardedClient(self.client.options(**kwargs), self.check_owner)

    def bulk(self, *, operations: list[Any], **kwargs: Any) -> dict[str, Any]:
        if len(operations) % 2:
            raise ValueError("maintenance bulk requires paired actions and payloads")
        results: list[Any] = []
        packet: list[Any] = []
        size = 0

        def send() -> None:
            if not packet:
                return
            pending = list(packet)
            slots: list[Any] = [None] * (len(packet) // 2)
            positions = list(range(len(slots)))
            for attempt in range(7):
                response = self.retry(self.client.bulk, operations=pending, **kwargs)
                items = response.get("items", [])
                if len(items) != len(positions):
                    raise ValueError("incomplete bounded bulk response")
                retry_ops: list[Any] = []
                retry_positions = []
                for n, (position, item) in enumerate(
                    zip(positions, items, strict=True)
                ):
                    outcome = next(iter(item.values()))
                    if outcome.get("status") in (429, 502, 503, 504) and attempt < 6:
                        retry_ops.extend(pending[n * 2 : n * 2 + 2])
                        retry_positions.append(position)
                    else:
                        slots[position] = item
                if not retry_ops:
                    results.extend(slots)
                    return
                pending, positions = retry_ops, retry_positions
                time.sleep(min(2**attempt, 30))
            raise AssertionError("unreachable")

        for offset in range(0, len(operations), 2):
            pair = operations[offset : offset + 2]
            pair_size = len(json.dumps(pair).encode())
            if packet and (len(packet) >= 32 or size + pair_size > 512_000):
                send()
                packet = []
                size = 0
            packet.extend(pair)
            size += pair_size
        send()
        return {
            "items": results,
            "errors": any(
                next(iter(item.values())).get("status", 500) >= 300 for item in results
            ),
        }
