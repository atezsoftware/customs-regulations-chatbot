# Chat Resume Authentication and Session Timestamps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry SSE chat streams after one successful token refresh and show each sidebar session's creation date and time.

**Architecture:** Reuse the existing refresh-token single-flight state from the API transport for both JSON and SSE requests. Render `ChatSession.createdAt` locally in `ChatSidebar`; no API or persistence changes are needed.

**Tech Stack:** React 19, TypeScript, Vite, Tailwind CSS.

## Global Constraints

- Retry an SSE request at most once after a 401.
- Preserve the exact stream URL, including `resumeRunId`, on retry.
- Do not add dependencies or change backend behavior.

---

### Task 1: Reusable authenticated stream request

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/sse.ts`

**Interfaces:**
- Produces: `fetchWithAuthRetry(input: RequestInfo | URL, init?: RequestInit): Promise<Response>`.
- Consumes: `tokenStore` and the existing refresh-token endpoint.

- [ ] **Step 1: Define the failing behavior**

Specify that a 401 stream request with a valid refresh token retries once with the new bearer token and retains its `resumeRunId`.

- [ ] **Step 2: Implement the minimal shared request helper**

Move the existing token attachment and single refresh/retry behavior into `fetchWithAuthRetry`, with an internal retry guard; make `apiFetch` consume it.

- [ ] **Step 3: Route SSE through the helper**

Replace the direct fetch in `streamMessageEvents` with `fetchWithAuthRetry(url, {signal})`.

- [ ] **Step 4: Verify**

Run `npm run build` from `frontend`; expected exit code: 0.

### Task 2: Sidebar session timestamp

**Files:**
- Modify: `frontend/src/components/ChatSidebar.tsx`

**Interfaces:**
- Consumes: optional `ChatSession.createdAt`.
- Produces: a local, compact date-time subtitle for valid timestamps.

- [ ] **Step 1: Define the failing behavior**

Specify that valid `createdAt` values render below the title and missing or invalid values render no subtitle.

- [ ] **Step 2: Implement formatting and markup**

Create a local formatter using `Intl.DateTimeFormat` and render the formatted value beneath the title without changing title selection behavior.

- [ ] **Step 3: Verify**

Run `npm run build` from `frontend`; expected exit code: 0.
