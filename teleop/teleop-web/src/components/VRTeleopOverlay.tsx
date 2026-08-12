// Copied from Interlatent-Main site/src/components/teleop/VRTeleopOverlay.tsx @ f7e4bfb6 (2026-07-30). Upstream is the dashboard copy; sync fixes both ways.
import { useEffect, useRef, useState } from 'react';
import { TeleopTokenOut, useTeleopToken } from '../lib/client';
import { ClutchPoseMapper } from '../lib/teleop/clutchPoseMapper';
import { QuicPoseSocket } from '../lib/teleop/quicPoseSocket';
import { describeWtError } from '../lib/teleop/webtransport';
import {
  Mat3,
  Quat,
  Vec3,
  matMul,
  rotateVec,
  rotY,
  yawFromQuat,
} from '../lib/teleop/quat';
import {
  CALIB_DURATION_S,
  CALIB_GRIP_THRESHOLD,
  DEFAULT_WRIST_OFFSET,
  PivotSample,
  WristOffsets,
  clearWristOffsets,
  loadWristOffsets,
  pivotPasses,
  quatToMat3,
  saveWristOffsets,
  solvePivot,
} from '../lib/teleop/wristCalibration';
import {
  HudSnapshot,
  XRScene,
  XRViewerPoseLike,
  XRWebGLLayerLike,
} from '../lib/teleop/xrScene';
import { TeleopProfiler } from '../lib/teleop/teleopProfiler';

/**
 * WebXR DAgger teleop overlay (Meta Quest, browser-native).
 *
 * The browser-native VR counterpart to the keyboard `TeleopOverlay`: the
 * operator opens the dashboard in the Quest Browser, hits "Enter VR", and a
 * WebXR `immersive-vr` session drives the arm through the same relay the
 * keyboard path uses.
 *
 * Split of labor (ADR 0009, second amendment): this overlay runs the
 * CLUTCH MAPPER — grip press captures the controller pose and the robot's
 * current EE pose (from the pod's `ee_state` stream) as anchors; while the
 * grip is held, controller deltas are scaled, yaw-corrected, reach-limited,
 * and composed into an ABSOLUTE EE target in the arm-base frame. The pod's
 * retarget stage runs IK on that target and forwards `mode="targets"` to
 * the node. Releasing the grip disengages: the hand can be repositioned
 * without moving the robot (re-clutching).
 *
 * Wire format (mode="pose"; consumed by the pod, never reaches the node):
 *   { engaged, deadman, seq, mode:"pose", ee_pos:[x,y,z], ee_quat:[x,y,z,w], pinch }
 *   ee_pos/ee_quat are the mapped absolute EE target in arm-base frame.
 * Return stream (pod → this overlay):
 *   { type:"ee_state", ready, ee_pos, ee_quat, obs_age_ms, ik:{...} }
 *
 * Bimanual: when the teleop-token's `ik_hints` carries a `chains` object
 * (one hint set per arm — a bimanual robot bundle), this overlay instead
 * reads BOTH controllers and sends `{ ..., mode:"pose", chains: { left:
 * {engaged, ee_pos, ee_quat, pinch}, right: {...} } }` — one independent
 * `ClutchPoseMapper`/engage-state/`ee_state` per arm, composed pod-side into
 * one combined joint-target vector (an un-clutched arm simply holds its
 * last commanded position). Single-arm sessions are entirely unaffected —
 * this is an additive parallel path, detected purely by `ik_hints` shape.
 *
 * Controls (controllers-first v1):
 *   - squeeze/grip button   → clutch: engage + deadman (release = arm holds)
 *   - trigger               → pinch (gripper close: 1 = closed)
 *   - right controller pose → end-effector target (clutch-relative)
 *     (bimanual: left controller drives the left arm, right the right arm)
 *   - LEFT controller (while disengaged) → camera-layout tool: aim at a video
 *     quad, hold its trigger to carry it, thumbstick to push/pull + resize,
 *     release to drop (persisted); thumbstick-click resets the layout.
 *
 * WebXR notes:
 *   - requires a secure context (HTTPS) — the dashboard already is.
 *   - the session is entered from a user gesture (the button) per spec.
 *   - immersive-vr needs a GL base layer; `XRScene` (lib/teleop/xrScene.ts)
 *     renders into it: one world-anchored video quad per robot camera, side
 *     by side (JPEG frames for every camera pushed pod→browser as binary WS
 *     messages), plus a HUD panel mirroring this overlay's status — so
 *     calibration prompts and engagement state are readable in-headset.
 *
 * The keyboard overlay is unchanged; this is an additive sibling.
 */

// Minimal WebXR typings (avoids a hard dep on @types/webxr). Loose by design.
type XRSessionLike = {
  requestReferenceSpace: (type: string) => Promise<unknown>;
  requestAnimationFrame: (cb: (t: number, frame: XRFrameLike) => void) => number;
  updateRenderState: (state: Record<string, unknown>) => void;
  end: () => Promise<void>;
  addEventListener: (t: string, cb: () => void) => void;
  inputSources: ArrayLike<XRInputSourceLike>;
  renderState: { baseLayer?: unknown };
};
type XRInputSourceLike = {
  handedness: 'left' | 'right' | 'none';
  targetRayMode: string;
  gripSpace?: unknown;
  // Aiming ray (origin + -Z forward), used by the camera-layout tool.
  targetRaySpace?: unknown;
  gamepad?: {
    buttons: ReadonlyArray<{ value: number; pressed: boolean }>;
    axes?: ReadonlyArray<number>;
  };
};
type XRFrameLike = {
  session: XRSessionLike;
  getPose: (
    space: unknown,
    base: unknown,
  ) => { transform: { position: XYZ; orientation: XYZW } } | null;
  // Viewer poses carry per-eye views (projection/view matrices) used by
  // the in-headset XRScene renderer; the yaw-correction path only reads
  // transform.orientation, which is unchanged.
  getViewerPose?: (base: unknown) => XRViewerPoseLike | null;
};
type XYZ = { x: number; y: number; z: number };
type XYZW = { x: number; y: number; z: number; w: number };

// Robot EE state streamed back by the pod's retarget stage.
type EEState = {
  ready: boolean;
  reason?: string;
  ee_pos?: number[];
  ee_quat?: number[];
  obs_age_ms?: number;
  ik?: { pos_err_mm?: number; limit_pressure?: number };
  receivedAt: number; // performance.now() at decode
};

// ee_state arrives at 10-30 Hz; treat anything older than this as gone.
const EE_STATE_STALE_MS = 1500;

// Camera-layout tool (left controller, disengaged only). Grab on trigger;
// thumbstick nudges depth (Y) and size (X) of the carried panel per frame.
const LAYOUT_GRAB_THRESHOLD = 0.6; // left-trigger value to grab
const LAYOUT_STICK_DEADZONE = 0.15;
const LAYOUT_DEPTH_RATE = 0.02; // meters/frame at full stick (~1.8 m/s @90Hz)
const LAYOUT_SCALE_RATE = 0.02; // scale multiplier step/frame at full stick

// Pose-frame send pacing. The XR loop runs at 72-90 Hz but the node's
// control loop consumes at ~30 Hz and the retarget stage is latest-wins,
// so sending every rAF is pure queueing pressure: on an uplink hiccup
// stale poses pile into the socket's kernel buffer and replay late.
// ~40 Hz engaged comfortably out-paces the consumer; disengaged
// keepalives only need to keep the relay's presence/idle-stop signals
// alive. State transitions (engage/disengage, ui changes) bypass the
// throttle so edges are never delayed.
const POSE_SEND_MIN_MS = 25;
const IDLE_SEND_MIN_MS = 200;
// Skip a (non-edge) send while this much is already sitting unsent in
// the socket — the next frame supersedes it anyway.
const SEND_BUFFERED_LIMIT_BYTES = 8192;

// Shared decoder for binary video-frame headers (one per message is
// garbage-collector churn at 10-30 fps × cameras).
const VIDEO_HEADER_DECODER = new TextDecoder();

// Default WebXR-world → arm-base rotation (vr-teleop-kit DEFAULT_R_CALIB):
// arm +X = quest -Z (forward), arm +Y = quest -X (left), arm +Z = quest +Y (up).
const DEFAULT_R_CALIB: Mat3 = [
  [0, 0, -1],
  [-1, 0, 0],
  [0, 1, 0],
];

function freshEEStateFrom(ref: { current: EEState | null }): EEState | null {
  const st = ref.current;
  if (!st || !st.ready) return null;
  if (performance.now() - st.receivedAt > EE_STATE_STALE_MS) return null;
  if (!st.ee_pos || st.ee_pos.length !== 3 || !st.ee_quat || st.ee_quat.length !== 4) {
    return null;
  }
  return st;
}

// QUIC node-sourced spec: the deadline after which we declare the node failed to
// serve its kinematic spec. There is no fallback source — the node is it.
const SPEC_TIMEOUT_MS = 8000;

function buildMapper(hints: Record<string, unknown> | undefined, rCalibRef: { current: Mat3 }) {
  const h = hints ?? {};
  if (Array.isArray(h.webxr_to_base_R) && (h.webxr_to_base_R as unknown[]).length === 3) {
    rCalibRef.current = h.webxr_to_base_R as Mat3;
  }
  return new ClutchPoseMapper({
    R: rCalibRef.current,
    scale: (h.scale_translation as number | undefined) ?? 1.0,
    scaleRotation: (h.scale_rotation as number | undefined) ?? 1.0,
    posReachLimit: (h.pos_reach_limit as number | undefined) ?? 0.25,
    rotReachLimit: (h.rot_reach_limit as number | undefined) ?? 0.6,
  });
}

export function VRTeleopOverlay({
  sessionId,
  open,
  onClose,
  mintToken,
}: {
  sessionId: string;
  open: boolean;
  onClose: () => void;
  /** Override the token mint (e.g. teleop-recording sessions mint against
   *  /teleop-recordings/{id}/teleop-token). Defaults to the inference-
   *  session route. */
  mintToken?: () => Promise<TeleopTokenOut>;
}) {
  const mint = useTeleopToken();
  const [status, setStatus] = useState<
    'idle' | 'unsupported' | 'minting' | 'connecting' | 'ready' | 'in-vr' | 'error'
  >('idle');
  const [error, setError] = useState<string | null>(null);
  // Tracked separately from `status`: the WS/token flow keeps flipping
  // `status` to 'ready' once connected regardless of XR hardware support, so
  // gating the Enter VR button on `status` alone let unsupported browsers
  // reach `enterVR()` and crash with the raw "No XR hardware found" error.
  const [xrSupported, setXrSupported] = useState(true);
  const [engaged, setEngaged] = useState(false);
  const [seq, setSeq] = useState(0);
  const [robotReady, setRobotReady] = useState(false);
  const [robotReason, setRobotReason] = useState<string | null>(null);

  // Bimanual mode (see module docstring): mirrors every single-arm piece of
  // state above, one per arm. `isBimanual` is render-facing; `isBimanualRef`
  // is what `onXRFrame`'s long-lived closure reads (state read inside that
  // closure would be a stale snapshot from whichever render was active when
  // `enterVR()` was called — same reason engage/mapper state already uses
  // refs, not state).
  const [isBimanual, setIsBimanual] = useState(false);
  const isBimanualRef = useRef(false);
  const [engagedLeft, setEngagedLeft] = useState(false);
  const [engagedRight, setEngagedRight] = useState(false);
  const [robotReadyLeft, setRobotReadyLeft] = useState(false);
  const [robotReasonLeft, setRobotReasonLeft] = useState<string | null>(null);
  const [robotReadyRight, setRobotReadyRight] = useState(false);
  const [robotReasonRight, setRobotReasonRight] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const seqRef = useRef(0);
  const xrSessionRef = useRef<XRSessionLike | null>(null);
  const refSpaceRef = useRef<unknown>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const eeStateRef = useRef<EEState | null>(null);
  const mapperRef = useRef<ClutchPoseMapper | null>(null);
  const rCalibRef = useRef<Mat3>(DEFAULT_R_CALIB);
  const engagedRef = useRef(false);

  // In-headset scene (video + HUD quads) + camera stream state. Refs
  // because the WS handler and XR frame loop are long-lived closures.
  // Every camera streams; decode in-flight is tracked per camera so a
  // slow decode on one feed never starves the others.
  const sceneRef = useRef<XRScene | null>(null);
  // Camera-layout (reposition) tool state. `layoutGrabRef` tracks the
  // trigger-held grab so we can edge-detect release; the others feed the HUD.
  const layoutGrabRef = useRef(false);
  const layoutActiveRef = useRef(false);
  const layoutTargetRef = useRef<string | null>(null);
  const layoutStickPressRef = useRef(false);
  const decodeInFlightRef = useRef<Record<string, boolean>>({});
  const lastVideoAtRef = useRef<number | null>(null);
  const camerasRef = useRef<string[]>([]);
  // Video latency stats (see recordVideoFrameStats). ts_ms is the NODE's
  // capture time (its monotonic clock; pod-arrival time only from old
  // nodes), so the skew now spans the full node→pod→browser path — but
  // it's still a DIFFERENT clock, so absolute one-way latency is not
  // measurable; instead track (arrival - ts_ms) skew and report each
  // frame's lag above the minimum skew seen — transit + queue + decode
  // jitter above the fastest path. Logged every window; the latest
  // summary feeds the in-headset HUD.
  const videoStatsRef = useRef({
    windowStart: 0,
    frames: 0,
    droppedDecodes: 0,
    decodeTotal: 0,
    decodeMax: 0,
    lagTotal: 0,
    lagMax: 0,
    minSkew: Infinity,
    // pod_ms decomposition (frames from a new pod only): skew against the
    // pod-arrival clock isolates the pod→display leg; total − pod = the
    // node→pod uplink leg. Which leg is slow decides whether to tune the
    // preview cadence / uplink or move the pod closer.
    podFrames: 0,
    podLagTotal: 0,
    podLagMax: 0,
    minSkewPod: Infinity,
    bytesTotal: 0,
    // ABSOLUTE glass→eye age (node capture ts_ms → arrival here), anchored
    // on the state-datagram clock skew (QuicPoseSocket.getStateSkewMs).
    // Unlike lag-above-fastest below, a STANDING queue delay shows up here
    // — the fastest-frame baseline absorbs it and reads ~0 under sustained
    // congestion (the "HUD says 60ms while eyes say 400ms" failure).
    ageFrames: 0,
    ageTotal: 0,
    ageMax: 0,
  });
  const videoReportRef = useRef<HudSnapshot['videoReport']>(null);

  // Per-VR-session profiler (see lib/teleop/teleopProfiler.ts). Built fresh
  // in enterVR() every time the operator actually enters VR (bimanual/
  // robotKind are already known by then from the token/hints); ticks every
  // second via profTickIntervalRef; downloads a CSV the moment the XR
  // session ends (headset off / Stop Teleop / system Exit-VR — see the
  // session's 'end' listener) — that's the boundary that matches "a
  // recording" from the operator's point of view, NOT the outer overlay-
  // close cleanup below (which only fires if the whole panel is closed,
  // and may never run in a normal enter-VR/exit-VR visit).
  const profRef = useRef<TeleopProfiler | null>(null);
  const profTickIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const VIDEO_STATS_WINDOW_MS = 5000;

  function recordVideoFrameStats(
    recvAt: number,
    podTsMs: number,
    podArrivalMs: number | null,
    decodeMs: number,
    frameBytes: number,
    stateSkewMs: number | null,
  ) {
    const s = videoStatsRef.current;
    const now = performance.now();
    if (s.windowStart === 0) s.windowStart = now;
    s.frames++;
    s.decodeTotal += decodeMs;
    s.decodeMax = Math.max(s.decodeMax, decodeMs);
    s.bytesTotal += frameBytes;
    const skew = recvAt - podTsMs;
    s.minSkew = Math.min(s.minSkew, skew);
    const lag = skew - s.minSkew;
    s.lagTotal += lag;
    s.lagMax = Math.max(s.lagMax, lag);
    if (stateSkewMs != null) {
      // skew − stateSkew = encode + uplink queue + relay + downlink transit
      // above one datagram flight — the absolute video age the operator
      // actually experiences (to arrival; decode adds decodeAvg on top).
      const age = Math.max(0, skew - stateSkewMs);
      s.ageFrames++;
      s.ageTotal += age;
      s.ageMax = Math.max(s.ageMax, age);
    }
    if (podArrivalMs != null) {
      const skewPod = recvAt - podArrivalMs;
      s.minSkewPod = Math.min(s.minSkewPod, skewPod);
      const podLag = skewPod - s.minSkewPod;
      s.podFrames++;
      s.podLagTotal += podLag;
      s.podLagMax = Math.max(s.podLagMax, podLag);
    }

    const elapsed = now - s.windowStart;
    if (elapsed < VIDEO_STATS_WINDOW_MS) return;
    const camCount = Math.max(1, camerasRef.current.length);
    const fpsPerCam = (s.frames / camCount) / (elapsed / 1000);
    const lagAvg = s.lagTotal / s.frames;
    const podLagAvg = s.podFrames > 0 ? s.podLagTotal / s.podFrames : null;
    // Full report for the in-headset HUD, pre-rounded so the HUD only
    // redraws once per window. Console gets the same numbers for
    // desktop debugging.
    videoReportRef.current = {
      windowS: Math.round(elapsed / 1000),
      frames: s.frames,
      fpsPerCam: Math.round(fpsPerCam * 10) / 10,
      ageAvgMs: s.ageFrames > 0 ? Math.round(s.ageTotal / s.ageFrames) : null,
      ageMaxMs: s.ageFrames > 0 ? Math.round(s.ageMax) : null,
      lagAvgMs: Math.round(lagAvg),
      lagMaxMs: Math.round(s.lagMax),
      podLagAvgMs: podLagAvg == null ? null : Math.round(podLagAvg),
      nodeLagAvgMs:
        podLagAvg == null ? null : Math.max(0, Math.round(lagAvg - podLagAvg)),
      bytesAvgKb: Math.round((s.bytesTotal / s.frames / 1024) * 10) / 10,
      decodeAvgMs: Math.round((s.decodeTotal / s.frames) * 10) / 10,
      decodeMaxMs: Math.round(s.decodeMax * 10) / 10,
      dropped: s.droppedDecodes,
    };
    const r = videoReportRef.current;
    console.info(
      `teleop video: ${r.frames} frames in ${r.windowS}s ` +
      `(${r.fpsPerCam} fps/cam × ${camCount}, ${r.bytesAvgKb} KB/frame), ` +
      (r.ageAvgMs != null
        ? `AGE avg ${r.ageAvgMs}ms max ${r.ageMaxMs}ms (absolute, state-anchored), `
        : '') +
      `lag avg ${r.lagAvgMs}ms max ${r.lagMaxMs}ms (above fastest` +
      (r.nodeLagAvgMs != null
        ? `; node→pod ${r.nodeLagAvgMs}ms + pod→display ${r.podLagAvgMs}ms`
        : '') +
      `), decode avg ${r.decodeAvgMs}ms max ${r.decodeMaxMs}ms, ` +
      `${r.dropped} dropped (decode busy)`,
    );
    // Reset the window. Carry min skews as the lag baselines, nudged up
    // 1 ms per window so slow clock drift can't inflate lag forever.
    s.windowStart = now;
    s.frames = 0;
    s.droppedDecodes = 0;
    s.decodeTotal = 0;
    s.decodeMax = 0;
    s.lagTotal = 0;
    s.lagMax = 0;
    s.minSkew += 1;
    s.podFrames = 0;
    s.podLagTotal = 0;
    s.podLagMax = 0;
    s.minSkewPod += 1;
    s.bytesTotal = 0;
    s.ageFrames = 0;
    s.ageTotal = 0;
    s.ageMax = 0;
  }

  const eeStateLeftRef = useRef<EEState | null>(null);
  const eeStateRightRef = useRef<EEState | null>(null);
  const mapperLeftRef = useRef<ClutchPoseMapper | null>(null);
  const mapperRightRef = useRef<ClutchPoseMapper | null>(null);
  const rCalibLeftRef = useRef<Mat3>(DEFAULT_R_CALIB);
  const rCalibRightRef = useRef<Mat3>(DEFAULT_R_CALIB);
  const engagedLeftRef = useRef(false);
  const engagedRightRef = useRef(false);

  // ── wrist-pivot calibration (enforced before teleop; see
  //    lib/teleop/wristCalibration.ts). While `calibRef.active`, the XR
  //    frame loop runs the capture ritual instead of teleop: squeeze the
  //    required grips to start a 5s capture (haptic pulse marks start/end),
  //    rotate the wrists freely while holding the pivots still. Offsets
  //    persist in localStorage per browser/operator station.
  type CalibUiState = 'required' | 'capturing' | 'failed' | 'done';
  const [calibState, setCalibState] = useState<CalibUiState>('required');
  // Shadow of `calibState` for the XR frame loop's HUD snapshot (state
  // reads inside that closure would be stale — same rule as engagedRef).
  const calibStateRef = useRef<CalibUiState>('required');
  function setCalib(s: CalibUiState) {
    calibStateRef.current = s;
    setCalibState(s);
  }
  const wristOffsetsRef = useRef<WristOffsets | null>(null);
  const calibRef = useRef<{
    active: boolean;
    capturing: boolean;
    startedAt: number;
    buffers: { left: PivotSample[]; right: PivotSample[] };
  }>({ active: false, capturing: false, startedAt: 0, buffers: { left: [], right: [] } });

  useEffect(() => {
    const stored = loadWristOffsets();
    if (stored) {
      wristOffsetsRef.current = stored;
      setCalib('done');
    }
  }, []);

  function wristOffsetFor(hand: 'left' | 'right'): Vec3 {
    return wristOffsetsRef.current?.[hand] ?? DEFAULT_WRIST_OFFSET;
  }

  function pulseAllControllers(session: XRSessionLike, intensity: number, ms: number) {
    for (const src of Array.from(session.inputSources)) {
      const ha = (src as unknown as { gamepad?: { hapticActuators?: Array<{ pulse: (i: number, d: number) => void }> } })
        .gamepad?.hapticActuators;
      if (ha && ha[0]) {
        try { ha[0].pulse(intensity, ms); } catch { /* no haptics */ }
      }
    }
  }

  function startRecalibration() {
    clearWristOffsets();
    wristOffsetsRef.current = null;
    calibRef.current = {
      active: true, capturing: false, startedAt: 0,
      buffers: { left: [], right: [] },
    };
    setCalib('required');
  }

  // ── mint token + open the relay WS (same path as the keyboard overlay) ──
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setStatus('minting');
    setError(null);
    setXrSupported(true);
    setIsBimanual(false);
    isBimanualRef.current = false;

    const xr = (navigator as unknown as { xr?: { isSessionSupported: (m: string) => Promise<boolean> } }).xr;
    if (!xr) {
      setStatus('unsupported');
      setXrSupported(false);
      setError('WebXR not available. Open this page in the Meta Quest Browser.');
      return;
    }

    xr.isSessionSupported('immersive-vr').then((supported) => {
      if (cancelled) return;
      if (!supported) {
        setXrSupported(false);
        setError('No VR headset detected. Open this page in the Meta Quest Browser with a headset connected.');
      }
    });

    const onToken = (tok: TeleopTokenOut) => {
      if (cancelled) return;

      // Build the clutch mapper(s) from the node-served `kinematic_spec`,
      // which carries the 5 mapper fields — flat, or per-side under `chains`
      // for a bimanual rig — so one builder handles either shape.
      const applyMapperHints = (src: Record<string, unknown>) => {
        const chains = src.chains as
          | { left?: unknown; right?: unknown }
          | undefined;
        // A `chains` wrapper alone does not mean bimanual: single-arm rigs
        // (a1z, so101) ship it with only `right` populated. Both sides must
        // be present, or this drives a second arm the robot does not have.
        if (chains && chains.left && chains.right) {
          isBimanualRef.current = true;
          setIsBimanual(true);
          mapperLeftRef.current = buildMapper(
            chains.left as Record<string, unknown>, rCalibLeftRef,
          );
          mapperRightRef.current = buildMapper(
            chains.right as Record<string, unknown>, rCalibRightRef,
          );
        } else {
          const single = (chains ? (chains.right ?? chains.left) : src) as
            Record<string, unknown>;
          mapperRef.current = buildMapper(single, rCalibRef);
        }
      };

      // The mapper is built from the node-served kinematic spec, so the
      // build is deferred to onSpec (below) rather than the token's hints.

      setStatus('connecting');

      // Inbound-frame handler. `data` is either a JSON string (ee_state /
      // camera_list) or an ArrayBuffer video frame arriving on a QUIC uni
      // stream (uint16-BE header length + JSON header + raw JPEG).
      const handleData = (data: string | ArrayBuffer) => {
        if (typeof data !== 'string') {
          // Binary = video frame: uint16-BE header length, JSON header
          // {type:"video",cam,seq,ts_ms}, then raw JPEG bytes. One frame
          // per camera per pod tick.
          if (!(data instanceof ArrayBuffer)) return;
          const recvAt = performance.now();
          const bytes = new Uint8Array(data);
          if (bytes.length < 2) return;
          const hlen = (bytes[0] << 8) | bytes[1];
          if (bytes.length < 2 + hlen) return;
          let header: { type?: string; cam?: string; ts_ms?: number; pod_ms?: number } = {};
          try {
            header = JSON.parse(VIDEO_HEADER_DECODER.decode(bytes.subarray(2, 2 + hlen)));
          } catch {
            return;
          }
          if (header.type !== 'video' || typeof header.cam !== 'string') return;
          const cam = header.cam;
          // QUIC sends no camera_list — register cams lazily from frame
          // headers so the HUD's per-camera stats are right. (Harmless on
          // WS: camera_list overwrites; the scene creates quads lazily too.)
          if (!camerasRef.current.includes(cam)) {
            camerasRef.current = [...camerasRef.current, cam];
          }
          const podTsMs = typeof header.ts_ms === 'number' ? header.ts_ms : null;
          const podArrivalMs = typeof header.pod_ms === 'number' ? header.pod_ms : null;
          const frameBytes = bytes.length - 2 - hlen;
          // Latest-wins per camera: drop this frame if a decode for the
          // same camera is still in flight — the pusher sends more.
          if (decodeInFlightRef.current[cam]) {
            videoStatsRef.current.droppedDecodes++;
            return;
          }
          decodeInFlightRef.current[cam] = true;
          createImageBitmap(new Blob([bytes.subarray(2 + hlen)], { type: 'image/jpeg' }))
            .then((bmp) => {
              const decodedAt = performance.now();
              lastVideoAtRef.current = decodedAt;
              if (podTsMs != null) {
                // Clock-skew anchor for absolute video age — only the QUIC
                // shim provides it (duck-typed; a plain WebSocket yields
                // null and the report simply omits the age line).
                const stateSkewMs =
                  (wsRef.current as unknown as {
                    getStateSkewMs?: () => number | null;
                  } | null)?.getStateSkewMs?.() ?? null;
                recordVideoFrameStats(
                  recvAt, podTsMs, podArrivalMs, decodedAt - recvAt,
                  frameBytes, stateSkewMs,
                );
              }
              const scene = sceneRef.current;
              if (scene) scene.setVideoFrame(cam, bmp);
              else bmp.close();
            })
            .catch(() => {})
            .finally(() => {
              decodeInFlightRef.current[cam] = false;
            });
          return;
        }
        try {
          const obj = JSON.parse(data);
          if (obj && obj.type === 'camera_list') {
            const cams: string[] = Array.isArray(obj.cameras)
              ? obj.cameras.map(String)
              : [];
            camerasRef.current = cams;
            sceneRef.current?.setCameras(cams);
            return;
          }
          if (obj && obj.type === 'ee_state') {
            profRef.current?.recordEeState();
            if (obj.chains) {
              // The pod is authoritative about arm count: chains-shaped
              // ee_state means a bimanual bundle is loaded pod-side. If the
              // token's ik_hints didn't say so (missing robot_kind, stale
              // hint cache), flip to bimanual here with default mappers —
              // otherwise these messages land in refs the single-arm path
              // never reads and the clutch can never engage.
              if (!isBimanualRef.current) {
                console.warn('teleop: pod reports bimanual ee_state but ik_hints were single-arm; switching to bimanual with default mapper hints');
                isBimanualRef.current = true;
                setIsBimanual(true);
                if (!mapperLeftRef.current) mapperLeftRef.current = buildMapper(undefined, rCalibLeftRef);
                if (!mapperRightRef.current) mapperRightRef.current = buildMapper(undefined, rCalibRightRef);
              }
              const now = performance.now();
              if (obj.chains.left) {
                eeStateLeftRef.current = { ...obj.chains.left, receivedAt: now };
                setRobotReadyLeft(!!obj.chains.left.ready);
                setRobotReasonLeft(obj.chains.left.ready ? null : (obj.chains.left.reason ?? null));
              }
              if (obj.chains.right) {
                eeStateRightRef.current = { ...obj.chains.right, receivedAt: now };
                setRobotReadyRight(!!obj.chains.right.ready);
                setRobotReasonRight(obj.chains.right.ready ? null : (obj.chains.right.reason ?? null));
              }
            } else {
              eeStateRef.current = { ...obj, receivedAt: performance.now() };
              setRobotReady(!!obj.ready);
              setRobotReason(obj.ready ? null : (obj.reason ?? null));
            }
          }
        } catch {
          // non-JSON or unknown message — ignore
        }
      };
      const handleClose = (reason?: string) => {
        if (cancelled) return;
        // A failed open fires onerror THEN onclose. Resetting to 'idle'
        // unconditionally wiped the 'error' status one tick after it was set,
        // leaving the button stuck on 'Connecting…' forever.
        setStatus((s) => (s === 'error' ? s : 'idle'));
        if (reason) setError((prev) => prev ?? `QUIC session closed — ${reason}`);
      };

      // QUIC transport: browser-side IK over WebTransport. The kinematic spec
      // comes from the NODE over the relay (built from its installed robot
      // data) and is the ONLY source — open the shim first, ask for the spec,
      // build the solver + mapper when it lands. There is deliberately no
      // platform/S3 fallback: a node that can't serve its own embodiment is a
      // real misconfiguration, and silently papering over it with a hosted copy
      // would hide the failure AND let the browser solve against kinematics that
      // aren't the ones this node is driving. Fail loud instead (see the
      // SPEC_TIMEOUT_MS deadline below). Live video arrives on the same
      // session's inbound uni streams and passes through the shim's onmessage
      // as ArrayBuffers (same framing as WS binary video — handleData shared).
      if (tok.transport === 'quic' && tok.webtransport_url) {
        const wtUrl = tok.webtransport_url;
        let opened = false;
        let specDone = false;
        // Enter-VR only once BOTH the transport is open and the spec is applied
        // — either alone is not enough to engage.
        const maybeReady = () => {
          if (opened && specDone && !cancelled) setStatus('ready');
        };

        let shim: QuicPoseSocket;
        try {
          shim = new QuicPoseSocket(wtUrl, tok.token, undefined, {
            onSpec: (spec) => {
              if (cancelled) return;
              specDone = true;
              applyMapperHints(spec as unknown as Record<string, unknown>);
              maybeReady();
            },
            // Present only against a self-hosted coordinator; the hosted relay
            // has a real certificate and needs no pinning.
            serverCertificateHashes: tok.server_certificate_hashes ?? undefined,
            // Lets the socket survive a relay restart. It re-mints rather than
            // reusing the original token, so a relay that moved (or rotated
            // its certificate) is picked up rather than dialled at a stale
            // address forever.
            // Only offered when the caller gave us a minter; without one a
            // reconnect could only reuse a token the relay may have dropped,
            // so the socket keeps its old close-once behaviour.
            remint: mintToken && (async () => {
              const fresh = await mintToken();
              if (!fresh.webtransport_url) {
                throw new Error('no webtransport_url on re-mint');
              }
              return {
                url: fresh.webtransport_url,
                token: fresh.token,
                serverCertificateHashes:
                  fresh.server_certificate_hashes ?? undefined,
              };
            }),
          });
        } catch (e) {
          setStatus('error');
          setError((e as Error).message);
          return;
        }
        wsRef.current = shim as unknown as WebSocket;
        shim.onopen = () => {
          if (cancelled) return shim.close();
          opened = true;
          maybeReady();
        };
        shim.onmessage = (ev) => handleData(ev.data);
        shim.onclose = (ev) => handleClose(ev?.reason);
        shim.onerror = (e) => {
          if (!cancelled) {
            setStatus('error');
            // Keep the browser's reason. 'QUIC connection failed' on its own
            // cannot distinguish a blocked UDP path from a rejected token
            // from a relay that closed the session after accepting it.
            setError(`QUIC connection failed — ${describeWtError(e)}`);
          }
        };

        // Hard deadline: the node never served a spec. Surface it rather than
        // leaving the operator staring at a disabled Enter-VR button.
        window.setTimeout(() => {
          if (cancelled || shim.hasSpec) return;
          setStatus('error');
          setError(
            'No kinematic spec from the node. Robot data ships with the SDK, '
            + "so its --robot kind is unknown to the node's interlatent "
            + 'version — check the kind, or upgrade interlatent there; the '
            + 'node log should say "serving local kinematic_spec".',
          );
        }, SPEC_TIMEOUT_MS);
        return;
      }

      // Teleop is QUIC-only (the WS relay path was removed). A token that
      // isn't 'quic' or carries no webtransport_url means the deployment
      // isn't QUIC-configured — a misconfiguration, not a fallback.
      setStatus('error');
      setError(
        'This session has no QUIC teleop endpoint. Teleop is QUIC-only — '
        + 'check that the deployment is QUIC-configured '
        + '(INTERLATENT_TELEOP_QUIC_RELAY_HOST).',
      );
    };

    if (mintToken) {
      mintToken().then(onToken).catch((err) => {
        if (cancelled) return;
        setStatus('error');
        setError((err as Error).message);
      });
    } else {
      mint.mutate(
        { sessionId },
        {
          onSuccess: onToken,
          onError: (err) => {
            if (cancelled) return;
            setStatus('error');
            setError((err as Error).message);
          },
        },
      );
    }

    return () => {
      cancelled = true;
      sendRelease();
      void xrSessionRef.current?.end().catch(() => {});
      xrSessionRef.current = null;
      if (profTickIntervalRef.current != null) {
        clearInterval(profTickIntervalRef.current);
        profTickIntervalRef.current = null;
      }
      // logToConsole() FIRST — the guaranteed channel (see its docstring:
      // downloadCsv() can silently no-op on this platform when triggered
      // from an async callback, which this cleanup is).
      profRef.current?.logToConsole();
      profRef.current?.downloadCsv();
      profRef.current = null;
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
      sceneRef.current?.dispose();
      sceneRef.current = null;
      lastVideoAtRef.current = null;
      decodeInFlightRef.current = {};
      camerasRef.current = [];
      videoReportRef.current = null;
      videoStatsRef.current = {
        windowStart: 0, frames: 0, droppedDecodes: 0,
        decodeTotal: 0, decodeMax: 0, lagTotal: 0, lagMax: 0,
        minSkew: Infinity,
        podFrames: 0, podLagTotal: 0, podLagMax: 0,
        minSkewPod: Infinity, bytesTotal: 0,
        ageFrames: 0, ageTotal: 0, ageMax: 0,
      };
      eeStateRef.current = null;
      engagedRef.current = false;
      setEngaged(false);
      setRobotReady(false);
      eeStateLeftRef.current = null;
      eeStateRightRef.current = null;
      mapperLeftRef.current = null;
      mapperRightRef.current = null;
      engagedLeftRef.current = false;
      engagedRightRef.current = false;
      setEngagedLeft(false);
      setEngagedRight(false);
      setRobotReadyLeft(false);
      setRobotReadyRight(false);
      isBimanualRef.current = false;
      setIsBimanual(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, sessionId]);

  function send(frame: Record<string, unknown>) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify(frame));
    } catch {
      // next frame will see the closed socket
    }
  }

  // Last transmitted frame's pacing state (see POSE_SEND_MIN_MS).
  const lastTxRef = useRef({ at: 0, engaged: false, ui: '' });

  /** Rate-limited send for the per-XR-frame handlers. Engagement/ui
   *  EDGES always go out immediately; steady-state frames are paced
   *  (engaged ~40 Hz, disengaged 5 Hz) and skipped entirely while the
   *  socket already has unsent bytes queued — a newer pose supersedes
   *  a stale one everywhere downstream (latest-wins relay + stage +
   *  node), so transmitting the backlog only adds latency. */
  function sendPaced(frame: Record<string, unknown>) {
    const now = performance.now();
    const last = lastTxRef.current;
    const engagedNow = !!frame.engaged;
    const ui = typeof frame.ui === 'string' ? frame.ui : '';
    const isEdge = engagedNow !== last.engaged || ui !== last.ui;
    if (!isEdge) {
      if (now - last.at < (engagedNow ? POSE_SEND_MIN_MS : IDLE_SEND_MIN_MS)) {
        profRef.current?.recordPoseSkipped('pace');
        return;
      }
      const ws = wsRef.current;
      if (ws && ws.bufferedAmount > SEND_BUFFERED_LIMIT_BYTES) {
        // The socket isn't draining — direct evidence of relay/network
        // backpressure on the control channel, the exact accumulation
        // signature we're chasing.
        profRef.current?.recordPoseSkipped('buffer');
        return;
      }
    }
    last.at = now;
    last.engaged = engagedNow;
    last.ui = ui;
    send(frame);
    profRef.current?.recordPoseSent(wsRef.current?.bufferedAmount ?? 0);
  }

  function sendRelease() {
    const frame = { engaged: false, deadman: false, seq: ++seqRef.current, mode: 'pose', ui: 'release' };
    lastTxRef.current = { at: performance.now(), engaged: false, ui: 'release' };
    send(frame);
  }

  // Status snapshot for the in-headset HUD. Reads REFS only (this runs
  // inside onXRFrame's long-lived closure). seq and video age are
  // quantized so the HUD canvas redraws at a few Hz, not 72.
  function buildHudSnapshot(): HudSnapshot {
    const armStatus = (ref: { current: EEState | null }) => {
      const st = ref.current;
      if (!st) return { ready: false, reason: 'no signal' };
      if (performance.now() - st.receivedAt > EE_STATE_STALE_MS) {
        return { ready: false, reason: 'stale signal' };
      }
      return { ready: !!st.ready, reason: st.ready ? null : (st.reason ?? null) };
    };
    const single = armStatus(eeStateRef);
    const left = armStatus(eeStateLeftRef);
    const right = armStatus(eeStateRightRef);
    const ikErr = eeStateRef.current?.ik?.pos_err_mm;
    const videoAt = lastVideoAtRef.current;
    return {
      bimanual: isBimanualRef.current,
      engaged: engagedRef.current,
      engagedLeft: engagedLeftRef.current,
      engagedRight: engagedRightRef.current,
      robotReady: single.ready,
      robotReason: single.reason,
      robotReadyLeft: left.ready,
      robotReasonLeft: left.reason,
      robotReadyRight: right.ready,
      robotReasonRight: right.reason,
      calibState: calibStateRef.current,
      ikPosErrMm: typeof ikErr === 'number' ? Math.round(ikErr) : null,
      seq: Math.floor(seqRef.current / 36) * 36,
      cameraCount: camerasRef.current.length,
      videoAgeMs:
        videoAt == null
          ? null
          : Math.floor((performance.now() - videoAt) / 500) * 500,
      videoReport: videoReportRef.current,
      layoutActive: layoutActiveRef.current,
      layoutGrabbing: layoutGrabRef.current,
      layoutTarget: layoutTargetRef.current,
    };
  }

  // ── enter the WebXR session (must be called from a user gesture) ────────
  async function enterVR() {
    if (!xrSupported) return;
    const xr = (navigator as unknown as {
      xr?: { requestSession: (m: string, o?: unknown) => Promise<XRSessionLike> };
    }).xr;
    if (!xr) return;
    try {
      const session = await xr.requestSession('immersive-vr', {
        optionalFeatures: ['local-floor'],
      });
      xrSessionRef.current = session;

      // One profiler per VR session (not per WS connection): the operator's
      // mental model of "a recording" is "the time I'm in the headset", so
      // that's the boundary the CSV download is tied to below, in the
      // session's own 'end' listener — not the outer overlay-close cleanup,
      // which the operator may never trigger (they just take the headset
      // off). Recreated on every enterVR() call so re-entering VR within
      // the same page-open session starts a fresh file.
      profRef.current = new TeleopProfiler({
        sessionId,
        userAgent: navigator.userAgent,
        bimanual: isBimanualRef.current,
        robotKind: isBimanualRef.current ? 'bimanual' : 'single-arm',
      });
      if (profTickIntervalRef.current == null) {
        profTickIntervalRef.current = setInterval(() => {
          profRef.current?.tick(videoReportRef.current);
        }, 1000);
      }

      // Minimal GL base layer so the session delivers frames.
      const canvas = document.createElement('canvas');
      canvasRef.current = canvas;
      const gl = canvas.getContext('webgl', { xrCompatible: true }) as
        | WebGLRenderingContext
        | null;
      glRef.current = gl;
      const XRWebGLLayerCtor = (window as unknown as { XRWebGLLayer?: new (s: unknown, g: unknown) => unknown }).XRWebGLLayer;
      if (gl && XRWebGLLayerCtor) {
        session.updateRenderState({ baseLayer: new XRWebGLLayerCtor(session, gl) });
      }
      // In-headset scene: per-camera video quads + HUD panel (see
      // lib/teleop/xrScene.ts). Seed the camera set if the pod's
      // camera_list already arrived while the WS was open pre-VR.
      if (gl) {
        sceneRef.current?.dispose();
        sceneRef.current = new XRScene(gl);
        if (camerasRef.current.length > 0) {
          sceneRef.current.setCameras(camerasRef.current);
        }
      }

      refSpaceRef.current = await session
        .requestReferenceSpace('local-floor')
        .catch(() => session.requestReferenceSpace('local'));

      session.addEventListener('end', () => {
        xrSessionRef.current = null;
        sceneRef.current?.dispose();
        sceneRef.current = null;
        // Abandon a mid-ritual capture cleanly; re-entering VR restarts it.
        calibRef.current.capturing = false;
        if (calibRef.current.active) setCalib('required');
        mapperRef.current?.disengage();
        mapperLeftRef.current?.disengage();
        mapperRightRef.current?.disengage();
        engagedRef.current = false;
        engagedLeftRef.current = false;
        engagedRightRef.current = false;
        setEngaged(false);
        setEngagedLeft(false);
        setEngagedRight(false);
        setStatus('ready');

        // This is the actual "operator stopped recording" moment (headset
        // off / Stop Teleop / system Exit-VR gesture). logToConsole() is
        // the GUARANTEED channel here — a download triggered from this
        // async event listener can be silently dropped by the browser's
        // user-gesture-trust rules with no error at all (see
        // teleopProfiler.ts's logToConsole() docstring); downloadCsv() is
        // kept as a bonus in case the platform allows it. Null out
        // afterward so the outer cleanup's own calls (belt-and-suspenders
        // for the unmount-while-still-in-VR case) find nothing to redo.
        if (profTickIntervalRef.current != null) {
          clearInterval(profTickIntervalRef.current);
          profTickIntervalRef.current = null;
        }
        profRef.current?.logToConsole();
        profRef.current?.downloadCsv();
        profRef.current = null;
      });

      // Enforce wrist-pivot calibration before any teleop: without stored
      // offsets the frame loop runs the calibration ritual instead of
      // sending pose frames.
      if (!wristOffsetsRef.current) {
        calibRef.current = {
          active: true, capturing: false, startedAt: 0,
          buffers: { left: [], right: [] },
        };
        setCalib('required');
      }

      setStatus('in-vr');
      session.requestAnimationFrame(onXRFrame);
    } catch (e) {
      setStatus('error');
      setError(`Could not start VR session: ${(e as Error).message}`);
    }
  }

  function onXRFrame(_t: number, frame: XRFrameLike) {
    const session = frame.session;
    session.requestAnimationFrame(onXRFrame);
    profRef.current?.recordFrame();

    // Render the in-headset scene (video + HUD quads). Falls back to a
    // bare clear when the scene/pose isn't available yet — the runtime
    // needs the layer touched every frame or it stops scheduling.
    const refSpace = refSpaceRef.current;
    const scene = sceneRef.current;
    const layer = session.renderState.baseLayer as XRWebGLLayerLike | undefined;
    const viewer = refSpace ? frame.getViewerPose?.(refSpace) : null;
    if (scene && layer && viewer) {
      if (!scene.placed) {
        const q = viewer.transform.orientation;
        scene.place(viewer.transform.position, yawFromQuat([q.x, q.y, q.z, q.w]));
      }
      scene.setHud(buildHudSnapshot());
      scene.render(layer, viewer);
    } else {
      const gl = glRef.current;
      if (gl) {
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      }
    }

    if (!refSpace) return;

    const next = ++seqRef.current;
    // The DOM overlay isn't visible in-headset; updating React state at
    // 72-90 Hz just forces re-renders on the Quest browser's single main
    // thread, competing with WS receive, JPEG decode dispatch, and the
    // sends themselves. ~2 Hz keeps the desktop mirror readable.
    if (next % 36 === 0) setSeq(next);

    if (calibRef.current.active) {
      handleCalibrationFrame(frame, refSpace, next);
      return;
    }

    // Camera-layout tool: only while fully disengaged (never fights teleop).
    // If a grab is somehow live when an arm engages, drop it so it persists.
    const engagedNow = isBimanualRef.current
      ? engagedLeftRef.current || engagedRightRef.current
      : engagedRef.current;
    if (!engagedNow) {
      handleLayout(frame, refSpace);
    } else {
      if (layoutGrabRef.current) {
        sceneRef.current?.endGrab();
        layoutGrabRef.current = false;
      }
      layoutActiveRef.current = false;
      layoutTargetRef.current = null;
    }

    if (isBimanualRef.current) {
      handleBimanualFrame(frame, refSpace, next);
    } else {
      handleSingleArmFrame(frame, refSpace, next);
    }
  }

  // ── wrist-pivot calibration ritual (runs instead of teleop while
  //    calibRef.active). Squeeze the required grips to start; samples raw
  //    (un-offset) grip poses for CALIB_DURATION_S while the operator
  //    rotates their wrists with pivots held still; solves + persists.
  function handleCalibrationFrame(frame: XRFrameLike, refSpace: unknown, next: number) {
    const session = frame.session;
    const c = calibRef.current;
    // Keep the relay disengaged for the whole ritual so calibration motion
    // can never command the robot. The `ui` tag surfaces in the pod log so
    // "browser won't engage" is diagnosable server-side.
    sendPaced({
      engaged: false, deadman: false, seq: next, mode: 'pose',
      ui: c.capturing ? 'calib_capturing' : 'calib_wait',
    });

    const required: Array<'left' | 'right'> =
      isBimanualRef.current ? ['left', 'right'] : ['right'];
    const poses: Partial<Record<'left' | 'right', { p: Vec3; q: Quat; grip: number }>> = {};
    for (const src of Array.from(session.inputSources)) {
      if (src.handedness !== 'left' && src.handedness !== 'right') continue;
      if (!src.gripSpace) continue;
      const pose = frame.getPose(src.gripSpace, refSpace);
      if (!pose) continue;
      const p = pose.transform.position;
      const q = pose.transform.orientation;
      const btn = src.gamepad?.buttons?.[1];
      poses[src.handedness] = {
        p: [p.x, p.y, p.z],
        q: [q.x, q.y, q.z, q.w],
        grip: btn ? (btn.value || (btn.pressed ? 1 : 0)) : 0,
      };
    }

    if (!c.capturing) {
      const allPressed = required.every(
        (h) => (poses[h]?.grip ?? 0) >= CALIB_GRIP_THRESHOLD,
      );
      if (allPressed) {
        c.capturing = true;
        c.startedAt = performance.now();
        c.buffers = { left: [], right: [] };
        pulseAllControllers(session, 1.0, 200); // start signal
        setCalib('capturing');
      }
      return;
    }

    for (const h of ['left', 'right'] as const) {
      const ps = poses[h];
      if (ps) c.buffers[h].push({ p: ps.p, R: quatToMat3(ps.q) });
    }
    if ((performance.now() - c.startedAt) / 1000 < CALIB_DURATION_S) return;

    c.capturing = false;
    pulseAllControllers(session, 1.0, 400); // end signal — long pulse
    const solved: Partial<Record<'left' | 'right', Vec3>> = {};
    let allOk = true;
    for (const h of required) {
      const r = solvePivot(c.buffers[h]);
      if (!pivotPasses(r)) {
        console.warn(`wrist calibration ${h} failed`, r);
        allOk = false;
        continue;
      }
      console.info(
        `wrist calibration ${h}: o=[${r.o.map((x) => x.toFixed(3)).join(', ')}] ` +
        `residual=${(r.rms * 1000).toFixed(1)}mm RMS (n=${r.n})`,
      );
      solved[h] = r.o;
    }
    if (!allOk) {
      // Stay in calibration mode; squeezing the grips again retries.
      setCalib('failed');
      return;
    }
    const offsets: WristOffsets = {
      left: solved.left ?? DEFAULT_WRIST_OFFSET,
      right: solved.right ?? DEFAULT_WRIST_OFFSET,
    };
    saveWristOffsets(offsets);
    wristOffsetsRef.current = offsets;
    c.active = false;
    setCalib('done');
  }

  // ── camera-layout tool (left controller) ──────────────────────────────
  // Runs only while fully disengaged, so it can never compete with active
  // teleop (single-arm teleop uses the RIGHT controller; the left grip that
  // clutches in bimanual mode ends layout the instant it engages). Aim the
  // left controller at a camera quad, hold its trigger to carry the panel
  // along the ray, and use the thumbstick to push/pull (Y) and resize (X);
  // release to drop and persist. Click the thumbstick to reset the layout.
  function handleLayout(frame: XRFrameLike, refSpace: unknown) {
    const scene = sceneRef.current;
    if (!scene || !scene.placed) {
      layoutActiveRef.current = false;
      return;
    }
    const session = frame.session;
    const sources = Array.from(session.inputSources);
    const src =
      sources.find((s) => s.handedness === 'left' && s.targetRaySpace) ??
      sources.find((s) => s.targetRaySpace);
    if (!src || !src.targetRaySpace) {
      if (layoutGrabRef.current) {
        scene.endGrab();
        layoutGrabRef.current = false;
      }
      layoutActiveRef.current = false;
      layoutTargetRef.current = null;
      return;
    }
    const pose = frame.getPose(src.targetRaySpace, refSpace);
    if (!pose) {
      layoutActiveRef.current = false;
      return;
    }
    layoutActiveRef.current = true;

    const o = pose.transform.position;
    const q = pose.transform.orientation;
    const origin: Vec3 = [o.x, o.y, o.z];
    // Ray forward is the controller's local -Z rotated into the ref space.
    const dir = rotateVec([0, 0, -1], [q.x, q.y, q.z, q.w]) as Vec3;

    const buttons = src.gamepad?.buttons ?? [];
    const axes = src.gamepad?.axes ?? [];
    const held = (buttons[0]?.value ?? 0) > LAYOUT_GRAB_THRESHOLD;
    // Quest thumbstick lives at axes[2]/axes[3]; some runtimes report [0]/[1].
    const dz = (v: number) => (Math.abs(v) < LAYOUT_STICK_DEADZONE ? 0 : v);
    const stickX = dz(axes[2] ?? axes[0] ?? 0);
    const stickY = dz(axes[3] ?? axes[1] ?? 0);
    const stickPress = buttons[3]?.pressed ?? false;

    // Thumbstick click resets the whole layout (only when not carrying).
    if (stickPress && !layoutStickPressRef.current && !layoutGrabRef.current) {
      scene.resetLayout();
    }
    layoutStickPressRef.current = stickPress;

    if (held && !layoutGrabRef.current) {
      if (scene.beginGrab(origin, dir)) layoutGrabRef.current = true;
    } else if (held && layoutGrabRef.current) {
      // Stick up (negative Y) pushes the panel farther away.
      scene.updateGrab(origin, dir, -stickY * LAYOUT_DEPTH_RATE, stickX * LAYOUT_SCALE_RATE);
    } else if (!held && layoutGrabRef.current) {
      scene.endGrab();
      layoutGrabRef.current = false;
    }
    layoutTargetRef.current = scene.layoutTarget(origin, dir);
  }

  // Original single-arm frame handling — unchanged from before the bimanual
  // extension, just relocated out of onXRFrame's body.
  function handleSingleArmFrame(frame: XRFrameLike, refSpace: unknown, next: number) {
    const mapper = mapperRef.current;
    if (!mapper) return;

    const session = frame.session;
    // Prefer the right controller; fall back to any tracked-pointer with a grip.
    const sources = Array.from(session.inputSources);
    const src =
      sources.find((s) => s.handedness === 'right' && s.gripSpace) ??
      sources.find((s) => s.gripSpace);
    if (!src || !src.gripSpace) return;

    const pose = frame.getPose(src.gripSpace, refSpace);
    if (!pose) return;
    // Aim (targetRay) orientation: the true pointing axis, used by the
    // mapper's swing–twist split (grip -Z is the handle, tilted from it).
    const aimPose = src.targetRaySpace
      ? frame.getPose(src.targetRaySpace, refSpace) : null;
    const aq = aimPose?.transform.orientation;
    const aimQuat: Quat | null = aq ? [aq.x, aq.y, aq.z, aq.w] : null;

    const buttons = src.gamepad?.buttons ?? [];
    const trigger = buttons[0]?.value ?? 0; // index trigger → pinch
    const grip = buttons[1]?.pressed ?? false; // squeeze → clutch (engage + deadman)

    const p = pose.transform.position;
    const q = pose.transform.orientation;
    const ctrlQuat: Quat = [q.x, q.y, q.z, q.w];
    // Shift the readout from the palm (gripSpace origin) to the operator's
    // calibrated wrist pivot so pure wrist twists produce ~zero translation.
    const offW = rotateVec(
      wristOffsetFor(src.handedness === 'left' ? 'left' : 'right'), ctrlQuat,
    );
    const ctrlPos: Vec3 = [p.x + offW[0], p.y + offW[1], p.z + offW[2]];

    const wasEngaged = engagedRef.current;
    const ee = freshEEStateFrom(eeStateRef);

    if (grip && !wasEngaged) {
      // Rising clutch edge. Refuse to engage until the pod reports a live
      // robot EE pose — the anchor for the clutch-relative mapping.
      if (!ee) {
        sendPaced({ engaged: false, deadman: false, seq: next, mode: 'pose', ui: 'grip_no_anchor' });
        return;
      }
      // Headset-yaw correction: "controller forward" should mean "robot
      // forward" regardless of where the operator's body faces. Same
      // recipe as vr-teleop-kit: R_engage = R_calib · R_y(-yaw_now).
      const viewer = frame.getViewerPose?.(refSpace);
      if (viewer) {
        const vq = viewer.transform.orientation;
        const yaw = yawFromQuat([vq.x, vq.y, vq.z, vq.w]);
        mapper.setR(matMul(rCalibRef.current, rotY(-yaw)));
      } else {
        mapper.setR(rCalibRef.current);
      }
      mapper.engage(
        ctrlPos,
        ctrlQuat,
        ee.ee_pos as Vec3,
        ee.ee_quat as Quat,
      );
      engagedRef.current = true;
      setEngaged(true);
    } else if (!grip && wasEngaged) {
      // Falling edge: disengage — hand repositioning moves nothing.
      mapper.disengage();
      engagedRef.current = false;
      setEngaged(false);
    }

    if (!engagedRef.current) {
      sendPaced({ engaged: false, deadman: false, seq: next, mode: 'pose', ui: 'no_grip' });
      return;
    }

    // Engaged: map the controller pose to an absolute EE target. Losing
    // ee_state mid-engage degrades to the anchor-relative path (mapper
    // called without current EE pose = legacy absolute mapping, still
    // safe — the pod re-clamps via IK caps + node SafetyGate).
    const tgt = mapper.target(
      ctrlPos,
      ctrlQuat,
      ee ? (ee.ee_pos as Vec3) : null,
      ee ? (ee.ee_quat as Quat) : null,
      aimQuat,
    );
    if (!tgt) return;

    sendPaced({
      engaged: true,
      deadman: true, // engagement IS the deadman, matching the keyboard overlay
      seq: next,
      mode: 'pose',
      ee_pos: tgt.pos,
      ee_quat: tgt.quat,
      pinch: trigger,
    });
  }

  // One arm's worth of clutch/engage/target logic, shared by both hands in
  // bimanual mode. Returns the wire payload for this arm this tick, or null
  // when disengaged/no controller/no target — the caller sends `chains`
  // with whichever sides are non-null, letting an un-clutched (or absent)
  // arm hold its last commanded position via the pod's passthrough.
  function processHand(
    frame: XRFrameLike,
    refSpace: unknown,
    src: XRInputSourceLike | undefined,
    mapperRef: { current: ClutchPoseMapper | null },
    engagedRef: { current: boolean },
    eeStateRef: { current: EEState | null },
    rCalibRef: { current: Mat3 },
    setEngaged: (v: boolean) => void,
  ): { engaged: true; ee_pos: number[]; ee_quat: number[]; pinch: number } | null {
    const mapper = mapperRef.current;
    if (!mapper || !src || !src.gripSpace) {
      if (engagedRef.current) {
        mapper?.disengage();
        engagedRef.current = false;
        setEngaged(false);
      }
      return null;
    }

    const pose = frame.getPose(src.gripSpace, refSpace);
    if (!pose) return null;
    // Aim orientation for the mapper's swing–twist split (see single-arm path).
    const aimPose = src.targetRaySpace
      ? frame.getPose(src.targetRaySpace, refSpace) : null;
    const aq = aimPose?.transform.orientation;
    const aimQuat: Quat | null = aq ? [aq.x, aq.y, aq.z, aq.w] : null;

    const buttons = src.gamepad?.buttons ?? [];
    const trigger = buttons[0]?.value ?? 0;
    const grip = buttons[1]?.pressed ?? false;

    const p = pose.transform.position;
    const q = pose.transform.orientation;
    const ctrlQuat: Quat = [q.x, q.y, q.z, q.w];
    // Same wrist-pivot readout shift as the single-arm path.
    const offW = rotateVec(
      wristOffsetFor(src.handedness === 'left' ? 'left' : 'right'), ctrlQuat,
    );
    const ctrlPos: Vec3 = [p.x + offW[0], p.y + offW[1], p.z + offW[2]];

    const wasEngaged = engagedRef.current;
    const ee = freshEEStateFrom(eeStateRef);

    if (grip && !wasEngaged) {
      if (!ee) return null; // refuse to engage without a live anchor
      const viewer = frame.getViewerPose?.(refSpace);
      if (viewer) {
        const vq = viewer.transform.orientation;
        const yaw = yawFromQuat([vq.x, vq.y, vq.z, vq.w]);
        mapper.setR(matMul(rCalibRef.current, rotY(-yaw)));
      } else {
        mapper.setR(rCalibRef.current);
      }
      mapper.engage(ctrlPos, ctrlQuat, ee.ee_pos as Vec3, ee.ee_quat as Quat);
      engagedRef.current = true;
      setEngaged(true);
    } else if (!grip && wasEngaged) {
      mapper.disengage();
      engagedRef.current = false;
      setEngaged(false);
    }

    if (!engagedRef.current) return null;

    const tgt = mapper.target(
      ctrlPos,
      ctrlQuat,
      ee ? (ee.ee_pos as Vec3) : null,
      ee ? (ee.ee_quat as Quat) : null,
      aimQuat,
    );
    if (!tgt) return null;

    return { engaged: true, ee_pos: tgt.pos, ee_quat: tgt.quat, pinch: trigger };
  }

  function handleBimanualFrame(frame: XRFrameLike, refSpace: unknown, next: number) {
    const session = frame.session;
    const sources = Array.from(session.inputSources);
    const leftSrc = sources.find((s) => s.handedness === 'left' && s.gripSpace);
    const rightSrc = sources.find((s) => s.handedness === 'right' && s.gripSpace);

    const left = processHand(
      frame, refSpace, leftSrc,
      mapperLeftRef, engagedLeftRef, eeStateLeftRef, rCalibLeftRef, setEngagedLeft,
    );
    const right = processHand(
      frame, refSpace, rightSrc,
      mapperRightRef, engagedRightRef, eeStateRightRef, rCalibRightRef, setEngagedRight,
    );

    if (!left && !right) {
      // Distinguish "not squeezing" from "squeezing but refused" (usually a
      // missing/stale ee_state anchor) so the pod log names the blocker.
      const gripHeld = [leftSrc, rightSrc].some(
        (s) => s?.gamepad?.buttons?.[1]?.pressed,
      );
      sendPaced({
        engaged: false, deadman: false, seq: next, mode: 'pose',
        ui: gripHeld ? 'grip_no_anchor' : 'no_grip',
      });
      return;
    }

    sendPaced({
      engaged: true,
      deadman: true,
      seq: next,
      mode: 'pose',
      chains: {
        left: left ?? { engaged: false },
        right: right ?? { engaged: false },
      },
    });
  }

  if (!open) return null;

  const supported = xrSupported;
  const ik = eeStateRef.current?.ik;
  const anyEngaged = isBimanual ? (engagedLeft || engagedRight) : engaged;
  const anyRobotReady = isBimanual ? (robotReadyLeft && robotReadyRight) : robotReady;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      tabIndex={-1}
    >
      <div className="w-[520px] max-w-[95vw] rounded-lg border border-status-warning/40 bg-bg-panel px-5 py-4 shadow-xl">
        <div className="flex items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                status === 'in-vr'
                  ? anyEngaged
                    ? 'bg-status-warning animate-pulse'
                    : 'bg-status-warning'
                  : status === 'error' || status === 'unsupported'
                    ? 'bg-status-critical'
                    : 'bg-text-tertiary'
              }`}
            />
            <span className="text-[10px] font-mono uppercase tracking-[0.14em] text-status-warning">
              {status === 'in-vr'
                ? calibState !== 'done'
                  ? calibState === 'capturing'
                    ? 'VR Teleop · CALIBRATING — rotate wrists, hold pivots still'
                    : 'VR Teleop · calibrate: squeeze grips to start'
                  : anyEngaged
                    ? 'VR Teleop · ENGAGED'
                    : anyRobotReady
                      ? 'VR Teleop · ready (hold grip)'
                      : 'VR Teleop · waiting for robot'
                : `VR Teleop · ${status}`}
            </span>
          </div>
          <button
            onClick={onClose}
            className="px-2.5 py-1 text-[11px] font-mono uppercase tracking-[0.14em] rounded border border-status-critical/40 text-status-critical hover:bg-status-critical/10"
          >
            Disengage
          </button>
        </div>

        {error && (
          <p className="text-[11px] font-mono text-status-critical mb-3">{error}</p>
        )}

        {supported && status !== 'in-vr' && (
          <button
            onClick={enterVR}
            disabled={status !== 'ready'}
            className="w-full mb-3 px-3 py-2 text-[12px] font-mono uppercase tracking-[0.14em] rounded border border-status-warning/40 text-status-warning hover:bg-status-warning/10 disabled:opacity-40"
          >
            {status === 'ready' ? 'Enter VR' : 'Connecting…'}
          </button>
        )}

        <div className="space-y-2 text-[11px] font-mono text-text-secondary">
          <div className="flex flex-wrap gap-3 gap-y-1">
            <span>Grip = clutch (deadman)</span>
            <span>Trigger = gripper</span>
            <span>Move controller = arm{isBimanual ? ' (per hand)' : ''}</span>
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-text-tertiary uppercase tracking-wider text-[10px]">
              wrist
            </span>
            <span className={calibState === 'done' ? 'text-text-primary' : 'text-status-warning'}>
              {calibState === 'done'
                ? 'calibrated'
                : calibState === 'capturing'
                  ? 'capturing — rotate wrists, keep pivots still'
                  : calibState === 'failed'
                    ? 'failed (arm moved?) — squeeze grips in VR to retry'
                    : 'required — enter VR, then squeeze grips to start (5s)'}
            </span>
            {calibState === 'done' && (
              <button
                onClick={startRecalibration}
                className="ml-auto px-2 py-0.5 text-[10px] font-mono uppercase tracking-[0.14em] rounded border border-border-subtle text-text-tertiary hover:bg-bg-hover"
              >
                Recalibrate
              </button>
            )}
          </div>
          {isBimanual ? (
            <>
              <div className="flex items-baseline gap-3 pt-2 border-t border-border-subtle">
                <span className="text-text-tertiary uppercase tracking-wider text-[10px]">
                  left arm
                </span>
                <span className={robotReadyLeft ? 'text-text-primary' : 'text-status-critical'}>
                  {robotReadyLeft ? 'live' : (robotReasonLeft ?? 'no signal')}
                </span>
              </div>
              <div className="flex items-baseline gap-3">
                <span className="text-text-tertiary uppercase tracking-wider text-[10px]">
                  right arm
                </span>
                <span className={robotReadyRight ? 'text-text-primary' : 'text-status-critical'}>
                  {robotReadyRight ? 'live' : (robotReasonRight ?? 'no signal')}
                </span>
                <span className="ml-auto text-text-tertiary text-[10px] tabular-nums">
                  seq {seq}
                </span>
              </div>
            </>
          ) : (
            <div className="flex items-baseline gap-3 pt-2 border-t border-border-subtle">
              <span className="text-text-tertiary uppercase tracking-wider text-[10px]">
                robot
              </span>
              <span className={robotReady ? 'text-text-primary' : 'text-status-critical'}>
                {robotReady ? 'live' : (robotReason ?? 'no signal')}
              </span>
              {ik && typeof ik.pos_err_mm === 'number' && (
                <span className="text-text-tertiary text-[10px] tabular-nums">
                  ik err {ik.pos_err_mm.toFixed(0)}mm
                </span>
              )}
              <span className="ml-auto text-text-tertiary text-[10px] tabular-nums">
                seq {seq}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
