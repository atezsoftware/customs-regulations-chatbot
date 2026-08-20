from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest

from onyx.regulatory.indexing_jobs.models import OpenRouterBatchConfig
from onyx.regulatory.indexing_jobs.openrouter_batch import (
    HttpxOpenRouterBatchGateway,
    OpenRouterBatchJobStatus,
    OpenRouterEmbeddingBatchRequest,
    parse_openrouter_embedding_results,
)


@pytest.fixture
def fake_openrouter_batch_server() -> Iterator[tuple[str, dict[str, object]]]:
    state: dict[str, object] = {"cancelled": False, "post_count": 0}

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: object) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/beta/batches":
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                state["payload"] = payload
                post_count = state["post_count"]
                assert isinstance(post_count, int)
                state["post_count"] = post_count + 1
                self._json(202, {"id": "batch-local", "status": "validating"})
                return
            if self.path == "/api/beta/batches/batch-local/cancel":
                state["cancelled"] = True
                self._json(200, {"id": "batch-local", "status": "cancelled"})
                return
            self._json(404, {"error": "not found"})

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/beta/batches/batch-local":
                self._json(
                    200,
                    {
                        "id": "batch-local",
                        "status": "completed",
                        "results": [
                            {
                                "custom_id": "group-1",
                                "response": {
                                    "status_code": 200,
                                    "body": {
                                        "model": "openai/text-embedding-3-large",
                                        "data": [
                                            {
                                                "index": 1,
                                                "embedding": [0.4, 0.5, 0.6],
                                            },
                                            {
                                                "index": 0,
                                                "embedding": [0.1, 0.2, 0.3],
                                            },
                                        ],
                                    },
                                },
                            }
                        ],
                    },
                )
                return
            self._json(404, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            del format, args
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = str(server.server_address[0])
        port = int(server.server_address[1])
        yield f"http://{host}:{port}/api/beta/batches", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_http_batch_submit_poll_and_cancel_with_official_contract(
    fake_openrouter_batch_server: tuple[str, dict[str, object]],
) -> None:
    url, server_state = fake_openrouter_batch_server
    gateway = HttpxOpenRouterBatchGateway(
        config=OpenRouterBatchConfig(
            api_url=url,
            model_name="openai/text-embedding-3-large",
            effective_dimension=3,
            request_input_size=2,
            max_requests=10,
            max_inputs=100,
            max_bytes=1_000_000,
            completion_horizon_seconds=60,
        ),
        api_key_provider=lambda: "local-test-key",
    )
    requests = [
        OpenRouterEmbeddingBatchRequest(
            custom_id="group-1", inputs=["context\nchunk-1", "context\nchunk-2"]
        )
    ]

    submitted = gateway.submit(requests, submission_key="submission-local")
    assert submitted.status is OpenRouterBatchJobStatus.PENDING
    payload = cast(dict[str, Any], server_state["payload"])
    assert payload["endpoint"] == "/v1/embeddings"
    assert payload["model"] == "openai/text-embedding-3-large"
    batch_requests = cast(list[dict[str, Any]], payload["requests"])
    body = cast(dict[str, Any], batch_requests[0]["body"])
    assert body["model"] == "openai/text-embedding-3-large"
    assert "metadata" not in payload
    completed = gateway.get(submitted.remote_batch_id)
    assert completed.status is OpenRouterBatchJobStatus.SUCCEEDED
    parsed = parse_openrouter_embedding_results(
        completed.results or [],
        expected_custom_ids={"group-1"},
        expected_model="openai/text-embedding-3-large",
        expected_dimension=3,
    )
    assert parsed["group-1"].vectors == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    gateway.cancel(submitted.remote_batch_id)
    assert server_state["cancelled"] is True
    assert server_state["post_count"] == 1
