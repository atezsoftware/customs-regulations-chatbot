# backend

TypeScript API backend (LoopBack 4), backed by Postgres (see `../db`).

## Setup

```bash
npm install
cp .env.example .env   # fill in DB_*, JWT_SECRET, STORAGE_ROOT
npm run dev             # NODE_ENV=local, watch mode
npm run build && npm start   # production
```

Requires `../db` to be running and migrated first (`../scripts/run.sh --env dev --apps db,backend` does both).

## Module layout

Each domain owns its own models/repositories/services/controllers under `src/modules/<name>/`. Modules depend on each other in one direction only (`directories`/`chat` may read `auth`'s `UserRepository` to resolve the current user; nothing flows the other way) — no module reaches into another's internals.

- `modules/auth` — accounts, login, tokens.
- `modules/directories` — a user's file clusters ("directories") and the files inside them.
- `modules/chat` — chat sessions and which directories each one is linked to.
- `common/auth/current-user.ts` — the **only** place that resolves "who is the current user" for every controller; this is also where the dev auth-bypass lives (see below). Don't re-implement auth checks elsewhere.
- `common/env.ts` — `isLocalEnv()`, the single place that checks `NODE_ENV === 'local'`. Gates both the auth bypass and file uploads. Deliberately distinct from `NODE_ENV=development`, which names a real deployed cluster (the `dev` EKS environment) and must behave like `test`/`production`.

## Auth, including the dev bypass

`common/auth/current-user.ts`'s `getCurrentUser()` is called by every controller that needs the logged-in user.

- `NODE_ENV=local` → every endpoint (except the `/auth/*` ones, which still work normally if you want to test them) treats the request as a fixed local dev user, auto-created on first use. No login needed. Directory management — creating a directory, uploading files (`POST /directories/{id}/files`), generating chunks and indexing (`POST /directories/{id}/chunks` and `/index`) — is also only enabled in this environment, since `core-indexer` isn't wired into any deployed environment yet; the frontend hides the whole Directories page via `uploadsEnabled` on `/auth/me`.
- Any other `NODE_ENV` (`development`, `test`, `production`) → a valid `Authorization: Bearer <accessToken>` is required everywhere; missing/invalid tokens get a 401. The directory-management endpoints above return 404. Listing/viewing directories and their index status still works everywhere (needed by chat sessions to link an already-indexed directory).

### `/auth/*`

- `POST /auth/register` — email + password (strength-validated) + optional full name → user + access/refresh token pair.
- `POST /auth/login` — bcrypt-compared against a dummy hash for unknown emails (no email-enumeration timing signal); locks the account 15 minutes after 5 consecutive failed attempts.
- `POST /auth/refresh` — rotates the refresh token; reusing a revoked one is rejected.
- `POST /auth/logout` — revokes a refresh token.
- `GET /auth/me` — current user.
- `POST /auth/change-password` — requires current password; revokes all of the user's outstanding refresh tokens on success.

Passwords: bcrypt, 12 rounds. Access tokens: short-lived JWT (`ACCESS_TOKEN_TTL`, default 15m). Refresh tokens: random 48-byte value, stored only as a SHA-256 hash (`REFRESH_TOKEN_TTL_DAYS`, default 30d). Global middleware: `helmet` + `express-rate-limit` (300 req / 15 min / IP).

### `/directories/*` — a user's file clusters

- `POST /directories` `{name}` — create.
- `GET /directories` — list the current user's directories.
- `GET /directories/{id}` — directory + its files.
- `PATCH /directories/{id}` `{name}` — rename.
- `DELETE /directories/{id}` — deletes the directory, its file records, and the files on disk.
- `POST /directories/{id}/files` — multipart upload, field name `files` (repeatable → multiple files per request, any extension).
- `PATCH /directories/{id}/files/{fileId}` `{name}` — rename a file.
- `DELETE /directories/{id}/files/{fileId}` — delete a file (DB row + on-disk file).

Uploaded files are stored under `STORAGE_ROOT/<directoryId>/<uuid><ext>` on disk; only the metadata (original name, mime type, size) lives in Postgres.

### `/chat-sessions/*`

- `POST /chat-sessions` `{title?}` — create (shown in the sidebar).
- `GET /chat-sessions` — list the current user's sessions.
- `PATCH /chat-sessions/{id}` `{title}` — rename.
- `DELETE /chat-sessions/{id}` — delete.
- `GET /chat-sessions/{id}/directories` — directories currently linked to this session.
- `PUT /chat-sessions/{id}/directories` `{directoryIds: string[]}` — **replaces** the full set of linked directories (every id must belong to the current user, or 400). This is the one place that decides what a session can see.
- `GET /chat-sessions/{id}/files` — every file from the session's linked directories, and *only* those — directories not linked to the session never appear here, even if they belong to the same user. Verified end-to-end (link to dir A → only A's files; expand to A+B → both show; delete dir B → cascades out of both the link table and this endpoint).
- `POST /chat-sessions/{id}/messages` `{content}` — creates a user message plus a pending assistant message.
- `GET /chat-sessions/{id}/messages/{messageId}/stream` — SSE stream for the assistant message. Backend builds a temporary symlink view under `STORAGE_ROOT/_sessions/<sessionId>/`, connects to internal `core` over WebSocket, persists research steps/sources/usage, and streams typed events to the browser.
- `GET /chat-sessions/{id}/messages` — persisted history with nested research steps, sources, and usage.
- `POST /chat-sessions/{id}/messages/{messageId}/cancel` — cancels an active stream best-effort.

## Not built yet

- Admin interface (per-user activity/audit view).
- Forgot-password-via-email flow.
