import {tokenStore} from './tokenStore';

export const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:3000';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

let refreshPromise: Promise<boolean> | null = null;

async function refreshAccessToken(): Promise<boolean> {
  const refreshToken = tokenStore.getRefresh();
  if (!refreshToken) return false;
  const res = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({refreshToken}),
  });
  if (!res.ok) {
    tokenStore.clear();
    return false;
  }
  tokenStore.set(await res.json());
  return true;
}

function withAccessToken(init: RequestInit = {}): RequestInit {
  const headers = new Headers(init.headers);
  const access = tokenStore.getAccess();
  if (access) headers.set('Authorization', `Bearer ${access}`);
  return {...init, headers};
}

/**
 * Sends an authenticated request and retries it once after refreshing an
 * expired access token. Keeping this at the transport layer lets both JSON
 * API calls and long-lived SSE streams recover without losing their URL or
 * request-specific options.
 */
export async function fetchWithAuthRetry(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  let res = await fetch(input, withAccessToken(init));
  if (res.status !== 401) return res;

  refreshPromise ??= refreshAccessToken().finally(() => {
    refreshPromise = null;
  });
  if (!await refreshPromise) return res;

  res = await fetch(input, withAccessToken(init));
  return res;
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  isForm?: boolean;
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const {method = 'GET', body, isForm} = options;
  const headers: Record<string, string> = {};

  let fetchBody: BodyInit | undefined;
  if (body !== undefined) {
    if (isForm) {
      fetchBody = body as FormData;
    } else {
      headers['Content-Type'] = 'application/json';
      fetchBody = JSON.stringify(body);
    }
  }

  const res = await fetchWithAuthRetry(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: fetchBody,
  });

  if (!res.ok) {
    let message = res.statusText;
    try {
      const data = await res.json();
      message = data?.error?.message ?? message;
    } catch {
      // response had no JSON body — fall back to statusText
    }
    throw new ApiError(message, res.status);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}
