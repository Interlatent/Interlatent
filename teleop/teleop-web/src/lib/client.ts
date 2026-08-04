/**
 * Minimal typed API client for the standalone teleop webapp.
 *
 * Zero dependencies — plain fetch. Auth is an Interlatent API key
 * ('ilat_…') sent as `x-api-key` on every request; the key and the API
 * base URL live in localStorage (set from the app's Settings panel):
 *
 *   interlatent.apiBase  — default https://interlatent.com
 *   interlatent.apiKey   — required for every call
 *
 * The response types below are copied from Interlatent-Main
 * site/src/lib/api.ts @ f7e4bfb6 (2026-07-30) — TeleopIkHints and
 * TeleopTokenOut verbatim (the copied VRTeleopOverlay imports
 * TeleopTokenOut from here); InferenceSessionOut and TeleopRecordingOut
 * are deliberately loose subsets (only the fields this UI reads, plus an
 * index signature) so backend additions never break the app.
 */

// ---------------------------------------------------------------------------
// Settings (localStorage)
// ---------------------------------------------------------------------------

export const API_BASE_STORAGE_KEY = 'interlatent.apiBase';
export const API_KEY_STORAGE_KEY = 'interlatent.apiKey';
export const DEFAULT_API_BASE = 'https://interlatent.com';

export function getApiBase(): string {
  const stored = localStorage.getItem(API_BASE_STORAGE_KEY);
  const base = (stored ?? DEFAULT_API_BASE).trim() || DEFAULT_API_BASE;
  return base.replace(/\/+$/, '');
}

export function getApiKey(): string {
  return (localStorage.getItem(API_KEY_STORAGE_KEY) ?? '').trim();
}

export function saveSettings(apiBase: string, apiKey: string): void {
  const base = apiBase.trim().replace(/\/+$/, '');
  localStorage.setItem(API_BASE_STORAGE_KEY, base || DEFAULT_API_BASE);
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey.trim());
}

// ---------------------------------------------------------------------------
// Fetch core
// ---------------------------------------------------------------------------

async function request<T>(method: 'GET' | 'POST', path: string): Promise<T> {
  const headers: Record<string, string> = { 'x-api-key': getApiKey() };
  const res = await fetch(`${getApiBase()}${path}`, {
    method,
    headers,
    cache: 'no-store',
  });
  if (!res.ok) throw await toError(res);
  return res.json() as Promise<T>;
}

/** Non-2xx → Error with the most useful message the body offers
 *  (FastAPI `{detail: "..."}` / `{detail: {message}}` / plain text). */
async function toError(res: Response): Promise<Error> {
  let message = '';
  let raw = '';
  try {
    raw = await res.text();
  } catch {
    /* body unreadable */
  }
  const trimmed = raw.trim();
  if (trimmed) {
    try {
      const body = JSON.parse(trimmed) as Record<string, unknown>;
      if (typeof body.detail === 'string') {
        message = body.detail;
      } else if (body.detail && typeof body.detail === 'object') {
        const d = body.detail as { message?: string };
        if (d.message) message = d.message;
      } else if (typeof body.message === 'string') {
        message = body.message;
      } else if (typeof body.error === 'string') {
        message = body.error;
      }
    } catch {
      if (!trimmed.startsWith('<')) {
        message = trimmed.length > 300 ? `${trimmed.slice(0, 300)}…` : trimmed;
      }
    }
  }
  if (!message) {
    if (res.status === 401 || res.status === 403) {
      message = `API ${res.status}: unauthorized — check your API key in Settings.`;
    } else if (res.statusText) {
      message = `API ${res.status}: ${res.statusText}`;
    } else {
      message = `API ${res.status}: request failed`;
    }
  } else if (res.status === 401) {
    message = `${message} (401 — check your API key in Settings)`;
  }
  return new Error(message);
}

// ---------------------------------------------------------------------------
// Types — copied from site/src/lib/api.ts @ f7e4bfb6 (2026-07-30)
// ---------------------------------------------------------------------------

export interface TeleopIkHints {
  // 3x3 rotation taking WebXR-world vectors into the arm-base frame, plus
  // clutch-mapper gains/limits — from the robot bundle's ik_config.
  webxr_to_base_R?: number[][];
  scale_translation?: number;
  scale_rotation?: number;
  pos_reach_limit?: number;
  rot_reach_limit?: number;
  // Present instead of the flat fields above when the robot bundle is
  // bimanual (two ik_config chains) — one hint set per arm. Single-arm
  // bundles never set this.
  chains?: { left: TeleopIkHints; right: TeleopIkHints };
}

export interface TeleopTokenOut {
  token: string;
  expires_at: string;
  // Static robot teleop schema (joint_names/min/max, max_velocity, rest_pose).
  // Consumed by the VR producer; null until a node has reported one.
  robot_schema?: Record<string, unknown> | null;
  // VR clutch-mapper hints from the curated robot bundle; null when no
  // bundle exists for this robot kind (overlay falls back to defaults).
  // On the QUIC path the overlay builds its mappers from the node-served
  // kinematic spec instead, so this is informational only.
  ik_hints?: TeleopIkHints | null;
  // Control transport. Teleop is QUIC-only: in-browser IK → WebTransport
  // datagrams to the QUIC relay (the WS relay path was removed).
  transport?: 'quic';
  // WebTransport/HTTP3 URL of the QUIC relay.
  webtransport_url?: string | null;
}

/** Loose subset of the dashboard's InferenceSessionOut — only what this
 *  UI reads. Statuses: provisioning | active | stopping | stopped |
 *  provision_failed. */
export interface InferenceSessionOut {
  id: string;
  status: string;
  environment_id: string | null;
  task?: string;
  policy_uri?: string;
  created_at?: string;
  [key: string]: unknown;
}

/** Loose subset of the dashboard's TeleopRecordingOut — only what this
 *  UI reads. Statuses: provisioning | active | stopping | stopped | failed. */
export interface TeleopRecordingOut {
  id: string;
  status: string;
  environment_id: string;
  task?: string;
  robot_kind?: string | null;
  created_at?: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Calls
// ---------------------------------------------------------------------------

export function listSessions(): Promise<InferenceSessionOut[]> {
  return request<InferenceSessionOut[]>('GET', '/api/v1/inference/sessions');
}

export function listTeleopRecordings(): Promise<TeleopRecordingOut[]> {
  return request<TeleopRecordingOut[]>('GET', '/api/v1/teleop-recordings');
}

export function mintSessionTeleopToken(sessionId: string): Promise<TeleopTokenOut> {
  return request<TeleopTokenOut>(
    'POST',
    `/api/v1/inference/sessions/${encodeURIComponent(sessionId)}/teleop-token?role=browser`,
  );
}

export function mintRecordingTeleopToken(recordingId: string): Promise<TeleopTokenOut> {
  return request<TeleopTokenOut>(
    'POST',
    `/api/v1/teleop-recordings/${encodeURIComponent(recordingId)}/teleop-token?role=browser`,
  );
}

// ---------------------------------------------------------------------------
// Compat shim for the copied VRTeleopOverlay
// ---------------------------------------------------------------------------

/**
 * Drop-in replacement for the dashboard's react-query `useTeleopToken()`
 * mutation hook (site/src/lib/api.ts) with the one call shape the overlay
 * uses: `mint.mutate({ sessionId }, { onSuccess, onError })`. Plain fetch,
 * no react-query. (The app shell always passes an explicit `mintToken`
 * prop, so this default path only runs if the overlay is mounted bare.)
 */
export function useTeleopToken() {
  return {
    mutate(
      { sessionId }: { sessionId: string },
      opts?: {
        onSuccess?: (tok: TeleopTokenOut) => void;
        onError?: (err: Error) => void;
      },
    ): void {
      mintSessionTeleopToken(sessionId).then(
        (tok) => opts?.onSuccess?.(tok),
        (err) => opts?.onError?.(err instanceof Error ? err : new Error(String(err))),
      );
    },
  };
}
