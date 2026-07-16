# Chat Resume Authentication and Session Timestamps Design

## Goal

Keep an interrupted chat resumable after an access-token expiry and make chat sessions easier to identify in the sidebar.

## Design

The frontend's authenticated request helper owns the existing, single-flight refresh-token flow. The SSE transport will call the same helper when its initial request returns 401, then retry the original stream request exactly once with the refreshed bearer token. The retry preserves `resumeRunId`, so the backend continues the retained core run rather than creating a new one. A second 401, or an unsuccessful refresh, surfaces the normal request failure and clears invalid credentials through the existing refresh path.

`ChatSidebar` renders a compact locale-aware date and time beneath each session title when `createdAt` is valid. It uses no new API field or database migration and preserves the current title truncation and selection behavior.

## Verification

Build the frontend after the type-safe changes. Manually verify that an expired access token with a valid refresh token causes one retry carrying the same `resumeRunId`, and that both new and historic sessions with a `createdAt` display their local date and time.
