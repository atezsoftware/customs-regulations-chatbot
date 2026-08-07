# OpenRouter Catalog and Chat Model Selection Design

## Goal

Expose the complete OpenRouter chat-model catalog, including Anthropic models,
and guarantee that the model selected in chat is the model used by normal chat,
deep research, and session-backed background work.

## Current Problems

OpenRouter discovery already retrieves the full provider catalog, but auto mode
only marks models from the static recommendation list as visible. Existing
OpenRouter rows are therefore hidden even though they are available.

Provider instances may have no configured display name. The web client serializes
such a selection with an empty provider name, and the backend only resolves exact
provider-instance names. The override is then omitted or unresolved and model
selection falls back to the administrator default.

The model-selection endpoint updates `current_alternate_model`, which is a UI
field, but does not update the structured `llm_override` used by backend chat
execution. Subsequent work can therefore lose the user's selection.

## Design

### Provider resolution

The frontend will use the provider type as the stable selector when an instance
has no display name. The backend will continue to prefer exact instance-name
matches and will fall back to the unique nameless provider of that type. Named
provider behavior remains unchanged.

### Session model persistence

The existing alternate-model value will be parsed into its provider name,
provider type, and model name. The endpoint will store a corresponding
`LLMOverride` on the chat session in the same transaction. Malformed values will
be rejected as invalid input instead of silently falling back to another model.

All chat modes will keep using the existing override precedence: request override,
then session override, then persona/provider default. Deep research already uses
the resolved model passed by the chat pipeline, so no separate model-selection
path will be introduced.

### OpenRouter visibility

For an OpenRouter provider in auto mode, every non-embedding model returned by
OpenRouter discovery will be visible. Synchronization will make both new and
previously known catalog entries visible and will not re-hide them during the
periodic recommended-model sync. Other providers retain their current
recommendation-based auto-mode behavior.

A forward data migration will mark existing OpenRouter model configurations
visible so deployment has deterministic immediate behavior without waiting for a
scheduled refresh. Future catalog refreshes preserve the invariant.

## Error Handling

An override that names no model or does not match the expected serialized format
will return the existing invalid-input error type. Provider resolution will only
fall back by provider type for a nameless provider; it will not arbitrarily choose
between named instances.

## Tests

Backend unit tests will cover nameless-provider resolution, structured session
override persistence, malformed alternate-model rejection, and OpenRouter catalog
visibility during discovery and auto-mode synchronization. Frontend tests will
cover the provider-type fallback in generated model options. The migration will
be checked with the repository's migration validation and the focused backend and
frontend suites will run before commit and push.

## Scope

This change does not alter OpenRouter credentials, account-level routing policy,
model pricing, or the visibility rules for non-OpenRouter providers.
