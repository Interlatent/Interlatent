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

export const API_BASE_STORAGE_KEY = 'interlatent.coordinator';
const LEGACY_API_BASE_KEY = 'interlatent.apiBase';
export const API_KEY_STORAGE_KEY = 'interlatent.apiKey';
/** Where the hosted dashboard serves the coordinator protocol.
 *  Offered as a suggestion in the settings panel; never a default —
 *  this app talks to whatever coordinator you point it at. */
export const HOSTED_COORDINATOR = 'https://interlatent.com';

/** The configured coordinator, or '' when none has been set.
 *
 *  There is no default. This app is a client of whatever coordinator you run —
 *  a hosted dashboard or `interlatent up` on your own LAN — and silently
 *  defaulting to one of them is how a self-hosted deployment ends up quietly
 *  talking to somebody else's control plane. `hasCoordinator()` lets the UI
 *  ask for one instead of failing at the first request. */
export function getApiBase(): string {
  const stored =
    localStorage.getItem(API_BASE_STORAGE_KEY) ??
    // Written by a build that predates the rename; read once so an existing
    // operator does not have to re-enter their address.
    localStorage.getItem(LEGACY_API_BASE_KEY);
  const base = (stored ?? '').trim();
  if (!base) return '';
  return resolveApiBase(base.replace(/\/+$/, ''));
}

/** False until the operator has told us where their coordinator is. */
export function hasCoordinator(): boolean {
  return getApiBase() !== '' || isDevProxy();
}

function isDevProxy(): boolean {
  const stored =
    localStorage.getItem(API_BASE_STORAGE_KEY) ??
    localStorage.getItem(LEGACY_API_BASE_KEY);
  return Boolean(import.meta.env.DEV && stored && stored.trim());
}

/**
 * Under `vite dev` we return the empty string instead of the hosted base — a
 * same-origin URL that the dev server's /api proxy (vite.config.ts) forwards to
 * the same backend, server-side, where CORS does not apply. A base pointing
 * anywhere else (self-hosted backend) is left alone.
 *
 * The hosted deployment's teleop CORS policy now defaults to allowing any
 * origin on the paths this app calls, so a direct dev-origin call would very
 * likely work too — but the proxy keeps dev working against a deployment that
 * has narrowed `INTERLATENT_TELEOP_CORS_ORIGINS` to an explicit list, which
 * would otherwise fail the preflight with `400 Disallowed CORS origin`.
 */
function resolveApiBase(base: string): string {
  return import.meta.env.DEV && base === HOSTED_COORDINATOR ? '' : base;
}

export function getApiKey(): string {
  return (localStorage.getItem(API_KEY_STORAGE_KEY) ?? '').trim();
}

export function saveSettings(apiBase: string, apiKey: string): void {
  const base = apiBase.trim().replace(/\/+$/, '');
  localStorage.setItem(API_BASE_STORAGE_KEY, base.trim());
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey.trim());
}

// ---------------------------------------------------------------------------
// Fetch core
// ---------------------------------------------------------------------------

async function request<T>(
  method: 'GET' | 'POST',
  path: string,
  body?: unknown,
): Promise<T> {
  const headers: Record<string, string> = { 'x-api-key': getApiKey() };
  // Only set content-type when there is a body: it is on the backend's teleop
  // CORS `allow_headers`, but sending it on a bodyless GET just widens the
  // preflight for nothing.
  if (body !== undefined) headers['content-type'] = 'application/json';
  const res = await fetch(`${getApiBase()}${path}`, {
    method,
    headers,
    cache: 'no-store',
    body: body === undefined ? undefined : JSON.stringify(body),
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
  // SHA-256 digests of a self-signed relay certificate, sent by a self-hosted
  // coordinator whose embedded relay lives at a LAN address that no public CA
  // will issue for. Absent against the hosted relay, which has a real cert.
  // Re-sent on every mint, so a rotation cannot strand a browser on a stale
  // digest.
  server_certificate_hashes?: Array<{ algorithm: string; value: string }> | null;
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
  /** Set when status is `failed` — the reason, worth showing verbatim. */
  error?: string | null;
  [key: string]: unknown;
}

/** Loose subset of the backend's NodeOut (site/app/models/schemas.py). The
 *  create form needs `online` and `current_session_id` to grey out nodes that
 *  cannot take a recording, and shows `robot_type` so you can tell two nodes
 *  apart. `online` is derived server-side from the heartbeat window (~30s). */
export interface NodeOut {
  id: string;
  name: string;
  online?: boolean;
  status?: string;
  robot_type?: string | null;
  /** Non-null while the node is running an inference session. */
  current_session_id?: string | null;
  [key: string]: unknown;
}

/** Loose subset of the backend's EnvironmentOut. Note the id field is
 *  `environment_id`, which is also the name the create body wants. */
export interface EnvironmentOut {
  environment_id: string;
  slug: string;
  display_name: string;
  episode_count?: number;
  [key: string]: unknown;
}

/** Body for `POST /api/v1/teleop-recordings` (TeleopRecordingCreate). Only
 *  the two ids are required; `fps` (30) and `idle_timeout_s` (300) are left to
 *  the server defaults, and `task` falls back to the environment's
 *  task_description when omitted. */
export interface TeleopRecordingCreate {
  environment_id: string;
  node_id: string;
  task?: string;
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

export function listNodes(): Promise<NodeOut[]> {
  return request<NodeOut[]>('GET', '/api/v1/nodes');
}

export function listEnvironments(): Promise<EnvironmentOut[]> {
  return request<EnvironmentOut[]>('GET', '/api/v1/environments');
}

/**
 * Start a teleop recording. Unlike an inference session this needs no GPU/pod
 * choice — the backend dispatches an ephemeral recording job itself — so a
 * node and an environment are the whole input.
 *
 * The returned recording is `provisioning`; it flips to `active` only once
 * that job reports its ingest endpoint, so a caller wanting to drive it must
 * wait for `active` (see App's pending-recording poll).
 *
 * Every precondition is enforced server-side and comes back as a readable
 * `detail` — 409 for an offline or already-busy node, 503 when the deployment
 * has no QUIC relay configured, 502 when the recording job won't start — so
 * callers can surface `Error.message` as-is.
 */
export function createTeleopRecording(
  body: TeleopRecordingCreate,
): Promise<TeleopRecordingOut> {
  // Omit an empty task rather than sending "": the backend then falls back to
  // the environment's task_description (same convention as the `interlatent`
  // CLI's session start).
  const task = (body.task ?? '').trim();
  return request<TeleopRecordingOut>('POST', '/api/v1/teleop-recordings', {
    environment_id: body.environment_id,
    node_id: body.node_id,
    ...(task ? { task } : {}),
  });
}

/** Ask the backend to wind the recording down (status → `stopping`). The node
 *  drops the assignment on its next poll and the episode uploads. */
export function stopTeleopRecording(
  recordingId: string,
): Promise<TeleopRecordingOut> {
  return request<TeleopRecordingOut>(
    'POST',
    `/api/v1/teleop-recordings/${encodeURIComponent(recordingId)}/stop`,
    {},
  );
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
