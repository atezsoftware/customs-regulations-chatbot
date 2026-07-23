# OpenRouter Request Compatibility Design

## Goal

Prevent chat requests from being rejected when a selectable OpenRouter model does not support reasoning controls or when the selected route cannot satisfy strict structured output.

## Design

The OpenRouter adapter will stop attaching a `reasoning` object to agent calls. The agent needs strict JSON-schema output for planning, but it does not require provider reasoning controls to function. Structured planning requests will include OpenRouter provider preferences requiring the response-format capability, so routing only considers compatible providers.

For non-success responses, the adapter will extract a bounded provider error message from the JSON error payload and expose it with the HTTP status. It will never include request headers, request payloads, or API keys.

## Verification

Unit tests will prove that a structured call made with a thinking level omits `reasoning`, requires response-format support, and preserves a provider rejection reason without exposing request data.
