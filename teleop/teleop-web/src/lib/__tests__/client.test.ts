/**
 * Tests for the write half of the API client — the recording lifecycle the web
 * app drives itself. The read calls are one-liners over the same `request`
 * helper, so what is worth pinning down here is the request *shape* (method,
 * path, headers, body) and that a backend refusal reaches the caller as a
 * readable message.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  API_BASE_STORAGE_KEY,
  API_KEY_STORAGE_KEY,
  createTeleopRecording,
  listEnvironments,
  listNodes,
  listSessions,
  listTeleopRecordings,
  stopTeleopRecording,
} from '../client';

const BASE = 'https://example.test';

/** Minimal localStorage so client.ts's settings reads work under node. */
function installLocalStorage(): void {
  const store = new Map<string, string>();
  (globalThis as unknown as { localStorage: unknown }).localStorage = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    clear: () => store.clear(),
  };
}

function mockJson(body: unknown, status = 200): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async () => ({
    ok: status >= 200 && status < 300,
    status,
    statusText: '',
    json: async () => body,
    text: async () => JSON.stringify(body),
  }));
  (globalThis as unknown as { fetch: unknown }).fetch = fn;
  return fn;
}

/** The (url, init) pair the client passed to fetch. */
function lastCall(fn: ReturnType<typeof vi.fn>): [string, RequestInit] {
  return fn.mock.calls[0] as unknown as [string, RequestInit];
}

beforeEach(() => {
  installLocalStorage();
  localStorage.setItem(API_BASE_STORAGE_KEY, BASE);
  localStorage.setItem(API_KEY_STORAGE_KEY, 'ilat_test');
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('createTeleopRecording', () => {
  it('POSTs the two ids as JSON and leaves fps/idle_timeout to the server', async () => {
    const fetchMock = mockJson({ id: 'rec_1', status: 'provisioning' });
    const rec = await createTeleopRecording({
      environment_id: 'env_1',
      node_id: 'node_1',
      task: 'pick up the cube',
    });

    expect(rec.id).toBe('rec_1');
    const [url, init] = lastCall(fetchMock);
    expect(url).toBe(`${BASE}/api/v1/teleop-recordings`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      environment_id: 'env_1',
      node_id: 'node_1',
      task: 'pick up the cube',
    });
    const headers = init.headers as Record<string, string>;
    expect(headers['x-api-key']).toBe('ilat_test');
    expect(headers['content-type']).toBe('application/json');
  });

  it('omits a blank task so the backend falls back to the env description', async () => {
    const fetchMock = mockJson({ id: 'rec_2', status: 'provisioning' });
    await createTeleopRecording({
      environment_id: 'env_1',
      node_id: 'node_1',
      task: '   ',
    });
    expect(JSON.parse(lastCall(fetchMock)[1].body as string)).toEqual({
      environment_id: 'env_1',
      node_id: 'node_1',
    });
  });

  it("surfaces the backend's refusal detail verbatim", async () => {
    // The 409 create_recording raises for a node with no recent heartbeat.
    mockJson(
      { detail: "Node 'my-arm' is offline (no recent heartbeat)." },
      409,
    );
    await expect(
      createTeleopRecording({ environment_id: 'env_1', node_id: 'node_1' }),
    ).rejects.toThrow("Node 'my-arm' is offline (no recent heartbeat).");
  });

  it('flags a bad key rather than leaking a bare status', async () => {
    mockJson({}, 401);
    await expect(
      createTeleopRecording({ environment_id: 'env_1', node_id: 'node_1' }),
    ).rejects.toThrow(/API key/i);
  });
});

describe('stopTeleopRecording', () => {
  it('POSTs to the recording stop route with an encoded id', async () => {
    const fetchMock = mockJson({ id: 'rec/1', status: 'stopping' });
    await stopTeleopRecording('rec/1');
    const [url, init] = lastCall(fetchMock);
    expect(url).toBe(`${BASE}/api/v1/teleop-recordings/rec%2F1/stop`);
    expect(init.method).toBe('POST');
  });
});

describe('listNodes', () => {
  it('GETs without a body or content-type', async () => {
    const fetchMock = mockJson([{ id: 'node_1', name: 'my-arm' }]);
    const nodes = await listNodes();
    expect(nodes).toHaveLength(1);
    const [url, init] = lastCall(fetchMock);
    expect(url).toBe(`${BASE}/api/v1/nodes`);
    expect(init.method).toBe('GET');
    expect(init.body).toBeUndefined();
    expect((init.headers as Record<string, string>)['content-type']).toBeUndefined();
  });
});

// A hosted deployment answers these routes with a bare array; the self-hosted
// coordinator wraps it under a key. Handing React the wrapper object is what
// crashed the app on `sessions.map is not a function`, so both shapes — and
// anything neither — have to come back as an array.
describe('list responses', () => {
  it('accepts the self-hosted coordinator envelope', async () => {
    mockJson({ sessions: [{ id: 'sess_1', status: 'running' }] });
    expect(await listSessions()).toHaveLength(1);

    mockJson({ recordings: [{ id: 'trec_1', status: 'running' }] });
    expect(await listTeleopRecordings()).toHaveLength(1);

    mockJson({ nodes: [{ id: 'node_1', name: 'my-arm' }] });
    expect(await listNodes()).toHaveLength(1);
  });

  it('accepts the hosted deployment bare array', async () => {
    mockJson([{ id: 'sess_1', status: 'active' }]);
    expect(await listSessions()).toHaveLength(1);
  });

  it('degrades an unrecognized body to an empty list, not a render crash', async () => {
    mockJson({ detail: 'something else entirely' });
    expect(await listSessions()).toEqual([]);
  });

  it('hides teleop recordings that the coordinator also lists as sessions', async () => {
    // The self-hosted coordinator stores a recording *as* the node's session,
    // so it shows up on both routes — tagged, and dropped here.
    mockJson({
      sessions: [
        { id: 'sess_1', status: 'running' },
        {
          id: 'trec_1',
          status: 'running',
          assignment_type: 'teleop_recording',
        },
      ],
    });
    expect((await listSessions()).map((s) => s.id)).toEqual(['sess_1']);
  });
});

describe('listEnvironments', () => {
  it("fills environment_id from the self-hosted coordinator's `id`", async () => {
    mockJson({ environments: [{ id: 'env_abc', slug: 'kitchen' }] });
    const [env] = await listEnvironments();
    expect(env.environment_id).toBe('env_abc');
  });

  it('falls back to the slug, which the create route also resolves', async () => {
    mockJson({ environments: [{ slug: 'kitchen' }] });
    const [env] = await listEnvironments();
    expect(env.environment_id).toBe('kitchen');
  });

  it('leaves a hosted environment_id alone', async () => {
    mockJson([{ environment_id: 'env_1', slug: 'kitchen', id: 'ignored' }]);
    const [env] = await listEnvironments();
    expect(env.environment_id).toBe('env_1');
  });
});
