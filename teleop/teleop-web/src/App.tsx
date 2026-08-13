import { useCallback, useEffect, useRef, useState } from 'react';
import { StartRecordingPanel } from './components/StartRecordingPanel';
import { VRTeleopOverlay } from './components/VRTeleopOverlay';
import {
  InferenceSessionOut,
  TeleopRecordingOut,
  getApiBase,
  getApiKey,
  hasCoordinator,
  listSessions,
  listTeleopRecordings,
  mintRecordingTeleopToken,
  mintSessionTeleopToken,
  saveSettings,
  stopTeleopRecording,
} from './lib/client';

// Statuses a browser producer can actually join. Two vocabularies: a hosted
// deployment provisions and then goes `active`, while the self-hosted
// coordinator (`interlatent up`) assigns a node synchronously and reports
// `running` from the first listing.
const JOINABLE = new Set(['active', 'running']);
// Recording statuses that still hold the node — i.e. worth offering Stop on.
const STOPPABLE = new Set(['provisioning', 'active', 'running']);
// How often to re-check a recording we just started, until it goes live. The
// wait is a compute cold start, so seconds, not milliseconds.
const PENDING_POLL_MS = 2000;

type Tab = 'sessions' | 'recordings';
type Target =
  | { kind: 'session'; id: string }
  | { kind: 'recording'; id: string };

function statusDot(status: string): string {
  if (status === 'active' || status === 'running') return 'bg-status-active';
  if (status === 'provisioning' || status === 'stopping') return 'bg-status-warning animate-pulse';
  if (status === 'failed' || status === 'provision_failed') return 'bg-status-critical';
  return 'bg-text-quaternary';
}

function SettingsPanel({
  onSaved,
  onCancel,
}: {
  onSaved: () => void;
  /** Present only when settings already exist (gear re-entry). */
  onCancel: (() => void) | null;
}) {
  const [apiBase, setApiBase] = useState(getApiBase());
  const [apiKey, setApiKey] = useState(getApiKey());

  return (
    <div className="w-full max-w-md mx-auto mt-16 rounded-lg border border-border-subtle bg-bg-panel p-5">
      <h2 className="text-sm font-mono uppercase tracking-wide text-text-secondary mb-4">
        Settings
      </h2>
      <label className="block mb-3">
        <span className="block text-[11px] font-mono uppercase tracking-wide text-text-tertiary mb-1">
          API base URL
        </span>
        <input
          type="url"
          value={apiBase}
          onChange={(e) => setApiBase(e.target.value)}
          placeholder="http://localhost:8900"
          className="w-full rounded border border-border-subtle bg-bg-elevated px-3 py-2 text-sm text-text-primary font-mono focus:outline-none focus:border-border-emphasis"
        />
      </label>
      <label className="block mb-4">
        <span className="block text-[11px] font-mono uppercase tracking-wide text-text-tertiary mb-1">
          API key
        </span>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="ilop_…"
          autoComplete="off"
          className="w-full rounded border border-border-subtle bg-bg-elevated px-3 py-2 text-sm text-text-primary font-mono focus:outline-none focus:border-border-emphasis"
        />
      </label>
      <div className="flex gap-2">
        <button
          onClick={() => {
            saveSettings(apiBase, apiKey);
            onSaved();
          }}
          disabled={!apiKey.trim()}
          className="px-4 py-2 text-[12px] font-mono uppercase tracking-wide rounded border border-status-info/40 text-status-info hover:bg-status-info/10 disabled:opacity-40"
        >
          Save
        </button>
        {onCancel && (
          <button
            onClick={onCancel}
            className="px-4 py-2 text-[12px] font-mono uppercase tracking-wide rounded border border-border-subtle text-text-tertiary hover:bg-bg-elevated"
          >
            Cancel
          </button>
        )}
      </div>
      <p className="mt-4 text-[11px] text-text-tertiary leading-relaxed">
        The key is stored only in this browser&rsquo;s localStorage and sent as
        the <span className="font-mono">x-api-key</span> header. Use the
        operator key your coordinator printed at{' '}
        <span className="font-mono">interlatent up</span>.
      </p>
    </div>
  );
}

function ItemRow({
  id,
  status,
  environmentId,
  task,
  onJoin,
  onStop,
}: {
  id: string;
  status: string;
  environmentId: string | null;
  task?: string;
  onJoin: (() => void) | null;
  /** Recordings only — omitted for inference sessions, which this app does
   *  not own the lifecycle of. */
  onStop?: (() => void) | null;
}) {
  return (
    <li className="flex items-center gap-3 rounded border border-border-subtle bg-bg-panel px-3 py-2.5">
      <span className={`w-2 h-2 rounded-full shrink-0 ${statusDot(status)}`} />
      <div className="min-w-0 flex-1">
        <div className="text-[12px] font-mono text-text-primary truncate">{id}</div>
        <div className="text-[11px] font-mono text-text-tertiary truncate">
          {status}
          {environmentId ? ` · env ${environmentId}` : ''}
          {task ? ` · ${task}` : ''}
        </div>
      </div>
      {onJoin ? (
        <button
          onClick={onJoin}
          className="shrink-0 px-3 py-1.5 text-[11px] font-mono uppercase tracking-wide rounded border border-status-active/40 text-status-active hover:bg-status-active/10"
        >
          Join
        </button>
      ) : (
        <span className="shrink-0 text-[10px] font-mono uppercase tracking-wide text-text-quaternary">
          not joinable
        </span>
      )}
      {onStop && (
        <button
          onClick={onStop}
          className="shrink-0 px-3 py-1.5 text-[11px] font-mono uppercase tracking-wide rounded border border-status-critical/40 text-status-critical hover:bg-status-critical/10"
        >
          Stop
        </button>
      )}
    </li>
  );
}

export function App() {
  const [hasKey, setHasKey] = useState(() => getApiKey() !== '');
  // There is no default coordinator, so an unconfigured app has nowhere to
  // send its first request. Open settings rather than showing an empty list
  // and a network error the operator cannot act on.
  const [showSettings, setShowSettings] = useState(
    () => getApiKey() === '' || !hasCoordinator(),
  );
  const [tab, setTab] = useState<Tab>('sessions');
  const [sessions, setSessions] = useState<InferenceSessionOut[]>([]);
  const [recordings, setRecordings] = useState<TeleopRecordingOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState<Target | null>(null);
  // A recording we just started and are waiting on. It is created
  // `provisioning` and only becomes joinable once its recording job reports an
  // ingest endpoint, so we watch for `active` and then enter VR unprompted.
  const [pending, setPending] = useState<{ id: string; status: string } | null>(
    null,
  );
  const targetRef = useRef<Target | null>(null);
  targetRef.current = target;

  const refresh = useCallback(async () => {
    if (getApiKey() === '') return;
    setLoading(true);
    setError(null);
    try {
      const [s, r] = await Promise.all([listSessions(), listTeleopRecordings()]);
      setSessions(s);
      setRecordings(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (hasKey) void refresh();
  }, [hasKey, refresh]);

  // Watch a just-started recording until it goes live, then join it. Polls the
  // recordings *collection* rather than the single-recording route on purpose:
  // the collection is already reachable cross-origin, so this needs no extra
  // CORS surface. Keyed on the id alone, so status updates don't restart it.
  const pendingId = pending?.id;
  useEffect(() => {
    if (!pendingId) return;
    let cancelled = false;
    const tick = async () => {
      let rows: TeleopRecordingOut[];
      try {
        rows = await listTeleopRecordings();
      } catch {
        return; // transient — keep waiting
      }
      if (cancelled) return;
      setRecordings(rows);
      const row = rows.find((r) => r.id === pendingId);
      if (!row) return;
      if (JOINABLE.has(row.status)) {
        setPending(null);
        // Don't yank the user out of an overlay they already opened.
        if (targetRef.current === null) {
          setTarget({ kind: 'recording', id: row.id });
        }
      } else if (row.status === 'failed' || row.status === 'stopped') {
        setPending(null);
        setError(
          (typeof row.error === 'string' && row.error) ||
            `Recording ${row.id} ${row.status} before it went live.`,
        );
      } else {
        setPending((cur) =>
          cur && cur.id === row.id && cur.status !== row.status
            ? { id: row.id, status: row.status }
            : cur,
        );
      }
    };
    void tick();
    const timer = setInterval(() => void tick(), PENDING_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [pendingId]);

  const stop = useCallback(
    async (recordingId: string) => {
      setError(null);
      try {
        await stopTeleopRecording(recordingId);
      } catch (e) {
        setError((e as Error).message);
        return;
      }
      setPending((cur) => (cur?.id === recordingId ? null : cur));
      setTarget((cur) =>
        cur?.kind === 'recording' && cur.id === recordingId ? null : cur,
      );
      await refresh();
    },
    [refresh],
  );

  if (showSettings) {
    return (
      <div className="min-h-screen px-4">
        <SettingsPanel
          onSaved={() => {
            setHasKey(true);
            setShowSettings(false);
          }}
          onCancel={hasKey ? () => setShowSettings(false) : null}
        />
      </div>
    );
  }

  const list =
    tab === 'sessions'
      ? sessions.map((s) => (
          <ItemRow
            key={s.id}
            id={s.id}
            status={s.status}
            environmentId={s.environment_id}
            task={s.task}
            onJoin={
              JOINABLE.has(s.status)
                ? () => setTarget({ kind: 'session', id: s.id })
                : null
            }
          />
        ))
      : recordings.map((r) => (
          <ItemRow
            key={r.id}
            id={r.id}
            status={r.status}
            environmentId={r.environment_id}
            task={r.task}
            onJoin={
              JOINABLE.has(r.status)
                ? () => setTarget({ kind: 'recording', id: r.id })
                : null
            }
            onStop={STOPPABLE.has(r.status) ? () => void stop(r.id) : null}
          />
        ));

  return (
    <div className="min-h-screen px-4 pb-10">
      <header className="max-w-2xl mx-auto flex items-center justify-between pt-6 pb-4">
        <h1 className="text-sm font-mono uppercase tracking-kicker text-text-primary">
          Interlatent Teleop
        </h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void refresh()}
            disabled={loading}
            className="px-3 py-1.5 text-[11px] font-mono uppercase tracking-wide rounded border border-border-subtle text-text-secondary hover:bg-bg-elevated disabled:opacity-40"
          >
            {loading ? 'Loading…' : 'Refresh'}
          </button>
          <button
            onClick={() => setShowSettings(true)}
            title="Settings"
            aria-label="Settings"
            className="px-3 py-1.5 text-[13px] rounded border border-border-subtle text-text-secondary hover:bg-bg-elevated"
          >
            ⚙
          </button>
        </div>
      </header>

      <main className="max-w-2xl mx-auto">
        <StartRecordingPanel
          onStarted={(rec) => {
            setError(null);
            setTab('recordings');
            setPending({ id: rec.id, status: rec.status });
            void refresh();
          }}
        />

        {pending && (
          <div className="mb-3 flex items-center gap-3 rounded border border-status-warning/40 bg-bg-panel px-3 py-2.5">
            <span className="w-2 h-2 rounded-full shrink-0 bg-status-warning animate-pulse" />
            <div className="min-w-0 flex-1">
              <div className="text-[12px] font-mono text-text-primary truncate">
                {pending.id}
              </div>
              <div className="text-[11px] font-mono text-text-tertiary truncate">
                {pending.status} · VR opens automatically once it goes live
              </div>
            </div>
            <button
              onClick={() => setPending(null)}
              title="Stop waiting — the recording keeps provisioning"
              className="shrink-0 px-3 py-1.5 text-[11px] font-mono uppercase tracking-wide rounded border border-border-subtle text-text-tertiary hover:bg-bg-elevated"
            >
              Stop waiting
            </button>
          </div>
        )}

        <div className="flex gap-1 mb-3">
          <button
            onClick={() => setTab('sessions')}
            className={`px-3 py-1.5 text-[11px] font-mono uppercase tracking-wide rounded border ${
              tab === 'sessions'
                ? 'border-border-emphasis bg-bg-elevated text-text-primary'
                : 'border-border-subtle text-text-tertiary hover:bg-bg-elevated'
            }`}
          >
            Inference sessions ({sessions.length})
          </button>
          <button
            onClick={() => setTab('recordings')}
            className={`px-3 py-1.5 text-[11px] font-mono uppercase tracking-wide rounded border ${
              tab === 'recordings'
                ? 'border-border-emphasis bg-bg-elevated text-text-primary'
                : 'border-border-subtle text-text-tertiary hover:bg-bg-elevated'
            }`}
          >
            Teleop recordings ({recordings.length})
          </button>
        </div>

        {error && (
          <p className="mb-3 text-[12px] font-mono text-status-critical border border-status-critical/40 rounded px-3 py-2">
            {error}
          </p>
        )}

        {list.length > 0 ? (
          <ul className="space-y-2">{list}</ul>
        ) : (
          !loading &&
          !error && (
            <p className="text-[12px] font-mono text-text-tertiary px-1 py-6">
              {tab === 'sessions'
                ? 'No inference sessions. Start one with `interlatent session start`, then Refresh.'
                : 'No teleop recordings. Start one above.'}
            </p>
          )
        )}

        <p className="mt-8 text-[11px] text-text-quaternary leading-relaxed">
          Only <span className="font-mono">active</span> (self-hosted:{' '}
          <span className="font-mono">running</span>) sessions are joinable.
          Joining opens the WebXR producer — use the Meta Quest Browser with a
          headset connected; grip = clutch, trigger = gripper.
        </p>
      </main>

      {target && (
        <VRTeleopOverlay
          open
          sessionId={target.id}
          onClose={() => setTarget(null)}
          mintToken={
            target.kind === 'session'
              ? () => mintSessionTeleopToken(target.id)
              : () => mintRecordingTeleopToken(target.id)
          }
        />
      )}
    </div>
  );
}
