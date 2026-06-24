# frontend

React + TypeScript + Vite, styled with Tailwind CSS v4, routed with `react-router-dom`. Talks to `../backend`.

## Setup

```bash
npm install
cp .env.example .env   # VITE_API_URL, defaults to http://localhost:3000
npm run dev
```

## Structure

- `pages/` — `LoginPage`, `RegisterPage`, `ChatPage`, `DirectoriesPage`.
- `components/` — `AppShell` (nav rail + logout), `ChatSidebar`, `LinkedDirectoriesPanel`, `AuthLayout`, `ProtectedRoute`, and `ui/` (small primitives: `Button`, `TextField`, `Modal`, `ConfirmModal`, `NamePromptModal`, `Spinner`).
- `context/AuthContext.tsx` — current user + login/register/logout, backed by `lib/tokenStore.ts`.
- `lib/api.ts` — fetch wrapper: attaches the bearer access token, transparently refreshes it once on a 401 and retries, throws `ApiError` otherwise.
- `lib/endpoints.ts` — one function per backend endpoint, grouped as `authApi` / `directoriesApi` / `chatSessionsApi` — mirrors `backend`'s module layout 1:1.

## What's wired up

- Register / login / logout, with automatic access-token refresh.
- Directories: create, rename, delete, multi-file upload (any extension), per-file rename/delete.
- Chat sessions: create (sidebar "New chat"), list, and per-session directory linking — the linked-directories panel calls `PUT /chat-sessions/{id}/directories`, and the "visible to this chat" list calls `GET /chat-sessions/{id}/files` to make the isolation guarantee visible in the UI (a session only ever shows files from directories explicitly linked to it).

Verified against a real backend + Postgres with a scripted browser run (register → create two directories with files → create a chat → link only one directory → confirm only that directory's file appears).

## Not built yet

- The actual chat message exchange (composer is present but intentionally inert — see the note in `ChatPage`) — needs `core`'s agent wired in behind a backend endpoint first.
- Admin interface.
