import { useCallback, useEffect, useState } from 'react';
import { VRTeleopOverlay } from './components/VRTeleopOverlay';
import {
  DEFAULT_API_BASE,
  InferenceSessionOut,
  TeleopRecordingOut,
  getApiBase,
  getApiKey,
  listSessions,
  listTeleopRecordings,
  mintRecordingTeleopToken,
  mintSessionTeleopToken,
  saveSettings,
} from './lib/client';

// Statuses a browser producer can actually join.
const JOINABLE = new Set(['active']);

type Tab = 'sessions' | 'recordings';
type Target =
  | { kind: 'session'; id: string }
  | { kind: 'recording'; id: string };

function statusDot(status: string): string {
  if (status === 'active') return 'bg-status-active';
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
  const [apiBase, setApiBase] = useState(getApiBase() || DEFAULT_API_BASE);
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
          placeholder={DEFAULT_API_BASE}
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
          placeholder="ilat_…"
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
        the <span className="font-mono">x-api-key</span> header. Create one in
        the Interlatent dashboard.
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
}: {
  id: string;
  status: string;
  environmentId: string | null;
  task?: string;
  onJoin: (() => void) | null;
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
    </li>
  );
}

export function App() {
  const [hasKey, setHasKey] = useState(() => getApiKey() !== '');
  const [showSettings, setShowSettings] = useState(() => getApiKey() === '');
  const [tab, setTab] = useState<Tab>('sessions');
  const [sessions, setSessions] = useState<InferenceSessionOut[]>([]);
  const [recordings, setRecordings] = useState<TeleopRecordingOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState<Target | null>(null);

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
                ? 'No inference sessions. Launch one from the dashboard, then Refresh.'
                : 'No teleop recordings. Start one from the dashboard, then Refresh.'}
            </p>
          )
        )}

        <p className="mt-8 text-[11px] text-text-quaternary leading-relaxed">
          Only <span className="font-mono">active</span> sessions are joinable.
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
