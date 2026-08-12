/**
 * Start a teleop recording from the web app.
 *
 * The gap this closes: the app could only *join* work created elsewhere, so a
 * headset-only session meant taking the Quest off to start a recording from
 * the CLI. A recording needs no GPU/pod choice (the backend dispatches its own
 * recording job), so a node and an environment are the whole form.
 *
 * Preconditions are the backend's job, not this component's — `createRecording`
 * surfaces a readable `detail` for an offline node, a node already busy, a
 * deployment with no QUIC relay, or a recording job that won't start. This form
 * only disables what it can see from `NodeOut` (offline, or holding an inference
 * session); a node busy with *another recording* is invisible here and arrives
 * as a 409 instead.
 *
 * Controls are sized for a headset: full-width native selects (the Quest
 * Browser renders these as its own picker), no hover-only affordances.
 */
import { useCallback, useEffect, useState } from 'react';
import {
  EnvironmentOut,
  NodeOut,
  TeleopRecordingOut,
  createTeleopRecording,
  listEnvironments,
  listNodes,
} from '../lib/client';

/** Why a node can't take a recording, or null when it can. */
function nodeBlockedReason(node: NodeOut): string | null {
  if (node.online === false) return 'offline';
  if (node.current_session_id) return 'running a session';
  return null;
}

function nodeLabel(node: NodeOut): string {
  const bits = [node.name];
  if (node.robot_type) bits.push(node.robot_type);
  const blocked = nodeBlockedReason(node);
  if (blocked) bits.push(blocked);
  return bits.join(' · ');
}

function envLabel(env: EnvironmentOut): string {
  const name = env.display_name || env.slug;
  const n = env.episode_count;
  return typeof n === 'number' ? `${name} · ${n} ep` : name;
}

const selectClass =
  'w-full rounded border border-border-subtle bg-bg-elevated px-3 py-2.5 ' +
  'text-[13px] font-mono text-text-primary focus:outline-none ' +
  'focus:border-border-emphasis disabled:opacity-40';

const labelClass =
  'block text-[11px] font-mono uppercase tracking-wide text-text-tertiary mb-1';

export function StartRecordingPanel({
  onStarted,
}: {
  /** Called with the freshly created (still `provisioning`) recording. */
  onStarted: (rec: TeleopRecordingOut) => void;
}) {
  const [nodes, setNodes] = useState<NodeOut[]>([]);
  const [envs, setEnvs] = useState<EnvironmentOut[]>([]);
  const [nodeId, setNodeId] = useState('');
  const [envId, setEnvId] = useState('');
  const [task, setTask] = useState('');
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [n, e] = await Promise.all([listNodes(), listEnvironments()]);
      setNodes(n);
      setEnvs(e);
      // Preselect the only sensible choice when there is exactly one.
      const usable = n.filter((x) => nodeBlockedReason(x) === null);
      setNodeId((cur) => cur || (usable.length === 1 ? usable[0].id : ''));
      setEnvId((cur) => cur || (e.length === 1 ? e[0].environment_id : ''));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      const rec = await createTeleopRecording({
        environment_id: envId,
        node_id: nodeId,
        task,
      });
      setTask('');
      onStarted(rec);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const ready = nodeId !== '' && envId !== '' && !starting;

  return (
    <section className="mb-4 rounded border border-border-subtle bg-bg-panel p-3">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-[11px] font-mono uppercase tracking-wide text-text-secondary">
          Start a recording
        </h2>
        <button
          onClick={() => void load()}
          disabled={loading}
          className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-wide rounded border border-border-subtle text-text-tertiary hover:bg-bg-elevated disabled:opacity-40"
        >
          {loading ? 'Loading…' : 'Reload'}
        </button>
      </div>

      <div className="space-y-3">
        <label className="block">
          <span className={labelClass}>Robot node</span>
          <select
            value={nodeId}
            onChange={(e) => setNodeId(e.target.value)}
            disabled={loading || nodes.length === 0}
            className={selectClass}
          >
            <option value="">
              {nodes.length === 0 ? '(no nodes paired)' : 'Select a node…'}
            </option>
            {nodes.map((n) => (
              <option
                key={n.id}
                value={n.id}
                disabled={nodeBlockedReason(n) !== null}
              >
                {nodeLabel(n)}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className={labelClass}>Environment</span>
          <select
            value={envId}
            onChange={(e) => setEnvId(e.target.value)}
            disabled={loading || envs.length === 0}
            className={selectClass}
          >
            <option value="">
              {envs.length === 0 ? '(no environments)' : 'Select an environment…'}
            </option>
            {envs.map((env) => (
              <option key={env.environment_id} value={env.environment_id}>
                {envLabel(env)}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className={labelClass}>Task (optional)</span>
          <input
            type="text"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="pick up the cube"
            className={selectClass}
          />
        </label>

        {error && (
          <p className="text-[12px] font-mono text-status-critical border border-status-critical/40 rounded px-3 py-2">
            {error}
          </p>
        )}

        <button
          onClick={() => void start()}
          disabled={!ready}
          className="w-full px-4 py-2.5 text-[12px] font-mono uppercase tracking-wide rounded border border-status-active/40 text-status-active hover:bg-status-active/10 disabled:opacity-40"
        >
          {starting ? 'Starting…' : 'Start recording'}
        </button>
        <p className="text-[11px] text-text-quaternary leading-relaxed">
          The recording provisions its own compute, then opens VR automatically
          once it goes live. Task defaults to the environment&rsquo;s description.
        </p>
      </div>
    </section>
  );
}
