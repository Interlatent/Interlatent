/**
 * In-headset scene for the WebXR teleop overlay — raw WebGL1, no deps.
 *
 * Renders world-anchored textured quads inside the immersive-vr session
 * that `VRTeleopOverlay` opens:
 *
 *   - one VIDEO quad per robot camera, side by side in a row (JPEG
 *     frames for every camera pushed pod -> browser over the teleop WS,
 *     decoded to ImageBitmaps by the overlay and uploaded here), and
 *   - a HUD quad beneath the row, mirroring the DOM overlay's status
 *     text (engaged, robot ready, calibration ritual prompts, IK error)
 *     via an offscreen 2D canvas texture redrawn only on state change.
 *
 * All quads are placed once (`place()`) from the first viewer pose
 * after entering VR: the video row centered 1.2 m ahead of the head
 * along the viewer's yaw at head height, the HUD directly beneath it,
 * pitched toward the operator. World-anchored (not head-locked) for
 * comfort — re-entering VR re-places.
 *
 * Same typing philosophy as VRTeleopOverlay: minimal local WebXR
 * typings, no @types/webxr dependency.
 */

// ---------------------------------------------------------------------------
// Loose WebXR typings (render-path additions to the overlay's own set)
// ---------------------------------------------------------------------------

type XYZ = { x: number; y: number; z: number };
type XYZW = { x: number; y: number; z: number; w: number };

export type XRViewLike = {
  projectionMatrix: Float32Array;
  transform: {
    position: XYZ;
    orientation: XYZW;
    inverse: { matrix: Float32Array };
  };
};

export type XRViewerPoseLike = {
  transform: { position: XYZ; orientation: XYZW };
  views: ArrayLike<XRViewLike>;
};

export type XRWebGLLayerLike = {
  framebuffer: WebGLFramebuffer | null;
  getViewport: (view: XRViewLike) => {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null;
};

// ---------------------------------------------------------------------------
// HUD snapshot — everything the in-headset status panel displays
// ---------------------------------------------------------------------------

export type HudSnapshot = {
  bimanual: boolean;
  engaged: boolean;
  engagedLeft: boolean;
  engagedRight: boolean;
  robotReady: boolean;
  robotReason: string | null;
  robotReadyLeft: boolean;
  robotReasonLeft: string | null;
  robotReadyRight: boolean;
  robotReasonRight: string | null;
  calibState: 'required' | 'capturing' | 'failed' | 'done';
  ikPosErrMm: number | null;
  seq: number;
  cameraCount: number;
  /** Milliseconds since the last decoded video frame (any camera); null = never. */
  videoAgeMs: number | null;
  /** Full latency report for the last stats window; null until the first
   *  window closes. All values pre-rounded so the snapshot key is stable
   *  between windows. Lag is measured above the fastest observed frame
   *  (pod and browser clocks are not synchronized, so absolute one-way
   *  latency is unknowable without a sync handshake). */
  videoReport: {
    windowS: number;
    frames: number;
    fpsPerCam: number;
    // ABSOLUTE glass→eye video age, anchored on the state-datagram clock
    // skew (QUIC transport only; null on WS or old nodes without state
    // ts_ms). Unlike lagAvgMs below, a standing queue delay shows up here.
    ageAvgMs: number | null;
    ageMaxMs: number | null;
    lagAvgMs: number;
    lagMaxMs: number;
    // Lag decomposition (null when the pod predates pod_ms headers):
    // nodeLagAvgMs = node→pod uplink leg, podLagAvgMs = pod→display leg.
    podLagAvgMs: number | null;
    nodeLagAvgMs: number | null;
    bytesAvgKb: number;
    decodeAvgMs: number;
    decodeMaxMs: number;
    dropped: number;
  } | null;
  /** Camera-reposition ("layout") mode state, for the in-headset hint line.
   *  Active only while disengaged; the left controller is the layout tool. */
  layoutActive: boolean;
  layoutGrabbing: boolean;
  /** Camera the layout ray is on (or the one being carried); null = none. */
  layoutTarget: string | null;
};

// ---------------------------------------------------------------------------
// Minimal column-major mat4 helpers (only what static quads need)
// ---------------------------------------------------------------------------

type Mat4 = Float32Array;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Model matrix: translate to `pos`, yaw (Y) then pitch (X) rotate. */
function mat4FromYawPitchTranslation(
  yaw: number,
  pitch: number,
  pos: [number, number, number],
): Mat4 {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  // R = Ry(yaw) * Rx(pitch), column-major.
  const m = new Float32Array(16);
  m[0] = cy;        m[1] = 0;   m[2] = -sy;       m[3] = 0;
  m[4] = sy * sp;   m[5] = cp;  m[6] = cy * sp;   m[7] = 0;
  m[8] = sy * cp;   m[9] = -sp; m[10] = cy * cp;  m[11] = 0;
  m[12] = pos[0];   m[13] = pos[1]; m[14] = pos[2]; m[15] = 1;
  return m;
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const VS = `
attribute vec3 a_pos;
attribute vec2 a_uv;
uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
varying vec2 v_uv;
void main() {
  gl_Position = u_proj * u_view * u_model * vec4(a_pos, 1.0);
  v_uv = a_uv;
}
`;

const FS = `
precision mediump float;
uniform sampler2D u_tex;
uniform float u_opacity;
varying vec2 v_uv;
void main() {
  vec4 c = texture2D(u_tex, v_uv);
  gl_FragColor = vec4(c.rgb, c.a) * u_opacity;
}
`;

// Video row placement (meters, relative to the viewer at place() time).
const VIDEO_DISTANCE_M = 1.2;
const VIDEO_WIDTH_M = 0.95;  // per-quad width for a single camera
const MAX_ROW_WIDTH_M = 2.4; // total row width cap — quads shrink to fit
const QUAD_GAP_M = 0.05;
// User-repositioned panels can be pushed/pulled and scaled within these
// bounds (meters of grab distance / multiplier on the base quad size).
const GRAB_MIN_DIST_M = 0.4;
const GRAB_MAX_DIST_M = 3.0;
const PANEL_MIN_SCALE = 0.4;
const PANEL_MAX_SCALE = 3.0;
const LAYOUT_STORAGE_PREFIX = 'il_teleop_layout_v1:';
const HUD_WIDTH_M = 0.5;
const HUD_GAP_M = 0.06;
const HUD_PITCH_RAD = -15 * (Math.PI / 180); // top tilted toward the head
// Dim a video quad when no frame has arrived for this long.
const VIDEO_STALE_MS = 2000;

const HUD_CANVAS_W = 1024;
const HUD_CANVAS_H = 768; // tall enough for the latency report section
const HUD_ASPECT = HUD_CANVAS_H / HUD_CANVAS_W;
const DEFAULT_VIDEO_ASPECT = 3 / 4; // height / width, until the first frame

type VideoPanel = {
  tex: WebGLTexture;
  aspect: number; // height / width
  hasFrame: boolean;
  lastFrameAt: number;
  model: Mat4;
  halfW: number;
  halfH: number;
  // Layout: when the operator drags a panel out of the auto row, it becomes
  // `userPlaced` and holds its own position (`localPos`, in the placed frame:
  // x=row-right, y=up, z=forward/depth) and `scale`. Auto-placed panels
  // ignore these and flow back into the row. `worldCenter` is the last
  // rebuilt world-space center, used for ray picking.
  userPlaced: boolean;
  localPos: [number, number, number];
  scale: number;
  worldCenter: [number, number, number];
};

export class XRScene {
  private gl: WebGLRenderingContext;
  private program: WebGLProgram | null = null;
  private quadBuf: WebGLBuffer | null = null;
  private aPos = -1;
  private aUv = -1;
  private uProj: WebGLUniformLocation | null = null;
  private uView: WebGLUniformLocation | null = null;
  private uModel: WebGLUniformLocation | null = null;
  private uTex: WebGLUniformLocation | null = null;
  private uOpacity: WebGLUniformLocation | null = null;

  /** Per-camera video panels in row order (camera_list order). */
  private videos = new Map<string, VideoPanel>();
  private hudTex: WebGLTexture | null = null;
  private hudCanvas: HTMLCanvasElement;
  private hudCtx: CanvasRenderingContext2D | null;
  private lastHudKey = '';
  private hudModel: Mat4 = mat4FromYawPitchTranslation(0, 0, [0, 0, 0]);
  private hudHalfH = (HUD_WIDTH_M * HUD_ASPECT) / 2;

  private isPlaced = false;
  private placedPos: [number, number, number] = [0, 0, 0];
  private placedYaw = 0;

  // Layout ("reposition") drag state.
  private grabbedCam: string | null = null;
  private grabDist = VIDEO_DISTANCE_M;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this.hudCanvas = document.createElement('canvas');
    this.hudCanvas.width = HUD_CANVAS_W;
    this.hudCanvas.height = HUD_CANVAS_H;
    this.hudCtx = this.hudCanvas.getContext('2d');
    this.initGL();
  }

  get placed(): boolean {
    return this.isPlaced;
  }

  private initGL() {
    const gl = this.gl;
    const compile = (type: number, src: string): WebGLShader | null => {
      const sh = gl.createShader(type);
      if (!sh) return null;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.warn('xrScene shader compile failed:', gl.getShaderInfoLog(sh));
        gl.deleteShader(sh);
        return null;
      }
      return sh;
    };
    const vs = compile(gl.VERTEX_SHADER, VS);
    const fs = compile(gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) return;
    const prog = gl.createProgram();
    if (!prog) return;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.warn('xrScene program link failed:', gl.getProgramInfoLog(prog));
      gl.deleteProgram(prog);
      return;
    }
    this.program = prog;
    this.aPos = gl.getAttribLocation(prog, 'a_pos');
    this.aUv = gl.getAttribLocation(prog, 'a_uv');
    this.uProj = gl.getUniformLocation(prog, 'u_proj');
    this.uView = gl.getUniformLocation(prog, 'u_view');
    this.uModel = gl.getUniformLocation(prog, 'u_model');
    this.uTex = gl.getUniformLocation(prog, 'u_tex');
    this.uOpacity = gl.getUniformLocation(prog, 'u_opacity');

    // Single shared vertex buffer; each drawQuad uploads its own sized
    // vertices (a handful per frame — bufferData cost is negligible).
    this.quadBuf = gl.createBuffer();
    this.hudTex = this.makeTexture();
  }

  private makeTexture(): WebGLTexture | null {
    const gl = this.gl;
    const t = gl.createTexture();
    if (!t) return null;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // 1x1 dark placeholder so unrendered textures are visible panels.
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([16, 18, 22, 255]),
    );
    return t;
  }

  // -------------------------------------------------------------------
  // Camera set + placement
  // -------------------------------------------------------------------

  /** Sync the panel set to the pod's camera list (row order = list order).
   *  Creates panels for new cameras, drops removed ones. */
  setCameras(names: string[]): void {
    const gl = this.gl;
    for (const [name, panel] of this.videos) {
      if (!names.includes(name)) {
        gl.deleteTexture(panel.tex);
        this.videos.delete(name);
      }
    }
    // Rebuild in list order so row position matches the pod's ordering.
    const ordered = new Map<string, VideoPanel>();
    for (const name of names) {
      const existing = this.videos.get(name);
      if (existing) {
        ordered.set(name, existing);
        continue;
      }
      const tex = this.makeTexture();
      if (!tex) continue;
      ordered.set(name, {
        tex,
        aspect: DEFAULT_VIDEO_ASPECT,
        hasFrame: false,
        lastFrameAt: 0,
        model: mat4FromYawPitchTranslation(0, 0, [0, 0, 0]),
        halfW: VIDEO_WIDTH_M / 2,
        halfH: (VIDEO_WIDTH_M * DEFAULT_VIDEO_ASPECT) / 2,
        userPlaced: false,
        localPos: [0, 0, 0],
        scale: 1,
        worldCenter: [0, 0, 0],
      });
    }
    this.videos = ordered;
    this.applySavedLayout();
    if (this.isPlaced) this.rebuildModels();
  }

  place(viewerPos: XYZ, viewerYaw: number): void {
    // Forward in WebXR world space for a yaw-only orientation is -Z
    // rotated by yaw around +Y.
    const fx = -Math.sin(viewerYaw);
    const fz = -Math.cos(viewerYaw);
    this.placedPos = [
      viewerPos.x + fx * VIDEO_DISTANCE_M,
      viewerPos.y,
      viewerPos.z + fz * VIDEO_DISTANCE_M,
    ];
    this.placedYaw = viewerYaw;
    this.rebuildModels();
    this.isPlaced = true;
  }

  private rebuildModels() {
    const panels = [...this.videos.values()];
    // Only panels still in the auto row share the width cap; user-placed
    // panels are pulled out and keep a full base width they can scale.
    const rowN = panels.filter((p) => !p.userPlaced).length;
    const quadW =
      rowN > 1
        ? Math.min(
            VIDEO_WIDTH_M,
            (MAX_ROW_WIDTH_M - (rowN - 1) * QUAD_GAP_M) / rowN,
          )
        : VIDEO_WIDTH_M;
    const rowW = rowN > 0 ? rowN * quadW + (rowN - 1) * QUAD_GAP_M : 0;

    // HUD hangs under the nominal auto row, so its drop is driven by the
    // tallest row panel (ignores panels the operator moved elsewhere).
    let rowMaxHalfH = (quadW * DEFAULT_VIDEO_ASPECT) / 2;
    let autoIdx = 0;
    for (const panel of panels) {
      const baseW = panel.userPlaced ? VIDEO_WIDTH_M : quadW;
      panel.halfW = (baseW / 2) * panel.scale;
      panel.halfH = ((baseW * panel.aspect) / 2) * panel.scale;
      let local: [number, number, number];
      if (panel.userPlaced) {
        local = panel.localPos;
      } else {
        const centerOffset =
          -rowW / 2 + quadW / 2 + autoIdx * (quadW + QUAD_GAP_M);
        local = [centerOffset, 0, 0];
        rowMaxHalfH = Math.max(rowMaxHalfH, panel.halfH);
        autoIdx++;
      }
      const world = this.localToWorld(local);
      panel.worldCenter = world;
      panel.model = mat4FromYawPitchTranslation(this.placedYaw, 0, world);
    }

    this.hudHalfH = (HUD_WIDTH_M * HUD_ASPECT) / 2;
    const hudWorld = this.localToWorld([
      0,
      -rowMaxHalfH - HUD_GAP_M - this.hudHalfH,
      0,
    ]);
    this.hudModel = mat4FromYawPitchTranslation(
      this.placedYaw, HUD_PITCH_RAD, hudWorld,
    );
  }

  // -------------------------------------------------------------------
  // Layout (reposition) — ray pick + carry a single panel, persisted
  // -------------------------------------------------------------------

  /** Placed-frame local → world. Local axes: x=row-right, y=up, z=forward. */
  private localToWorld(
    l: [number, number, number],
  ): [number, number, number] {
    const yaw = this.placedYaw;
    const rgtX = Math.cos(yaw), rgtZ = -Math.sin(yaw);
    const fwdX = -Math.sin(yaw), fwdZ = -Math.cos(yaw);
    return [
      this.placedPos[0] + rgtX * l[0] + fwdX * l[2],
      this.placedPos[1] + l[1],
      this.placedPos[2] + rgtZ * l[0] + fwdZ * l[2],
    ];
  }

  /** World → placed-frame local (inverse of localToWorld). */
  private worldToLocal(
    w: [number, number, number],
  ): [number, number, number] {
    const yaw = this.placedYaw;
    const rgtX = Math.cos(yaw), rgtZ = -Math.sin(yaw);
    const fwdX = -Math.sin(yaw), fwdZ = -Math.cos(yaw);
    const dx = w[0] - this.placedPos[0];
    const dy = w[1] - this.placedPos[1];
    const dz = w[2] - this.placedPos[2];
    return [dx * rgtX + dz * rgtZ, dy, dx * fwdX + dz * fwdZ];
  }

  /** Nearest panel a world-space ray hits (front face), or null. */
  private pick(
    origin: [number, number, number],
    dir: [number, number, number],
  ): { cam: string; t: number } | null {
    const yaw = this.placedYaw;
    // Panel front normal (faces the viewer) = -forward.
    const nx = Math.sin(yaw), nz = Math.cos(yaw);
    const rgtX = Math.cos(yaw), rgtZ = -Math.sin(yaw);
    let best: { cam: string; t: number } | null = null;
    for (const [cam, p] of this.videos) {
      const c = p.worldCenter;
      const denom = dir[0] * nx + dir[2] * nz;
      if (Math.abs(denom) < 1e-6) continue;
      const t = ((c[0] - origin[0]) * nx + (c[2] - origin[2]) * nz) / denom;
      if (t <= 0) continue;
      const hx = origin[0] + dir[0] * t;
      const hy = origin[1] + dir[1] * t;
      const hz = origin[2] + dir[2] * t;
      const dx = hx - c[0], dy = hy - c[1], dz = hz - c[2];
      const lx = dx * rgtX + dz * rgtZ; // along row-right
      const ly = dy;                    // along up
      if (Math.abs(lx) <= p.halfW && Math.abs(ly) <= p.halfH) {
        if (!best || t < best.t) best = { cam, t };
      }
    }
    return best;
  }

  /** Which panel a layout ray is on right now (for the HUD hint). */
  layoutTarget(
    origin: [number, number, number],
    dir: [number, number, number],
  ): string | null {
    if (this.grabbedCam) return this.grabbedCam;
    return this.pick(origin, dir)?.cam ?? null;
  }

  get grabbing(): boolean {
    return this.grabbedCam != null;
  }

  /** Try to grab the panel under the ray; returns true if one was hit. */
  beginGrab(
    origin: [number, number, number],
    dir: [number, number, number],
  ): boolean {
    const hit = this.pick(origin, dir);
    if (!hit) return false;
    this.grabbedCam = hit.cam;
    this.grabDist = hit.t;
    return true;
  }

  /** While grabbing: carry the panel along the ray; nudge depth/scale. */
  updateGrab(
    origin: [number, number, number],
    dir: [number, number, number],
    depthDelta: number,
    scaleDelta: number,
  ): void {
    if (!this.grabbedCam) return;
    const p = this.videos.get(this.grabbedCam);
    if (!p) { this.grabbedCam = null; return; }
    this.grabDist = clamp(
      this.grabDist + depthDelta, GRAB_MIN_DIST_M, GRAB_MAX_DIST_M,
    );
    const world: [number, number, number] = [
      origin[0] + dir[0] * this.grabDist,
      origin[1] + dir[1] * this.grabDist,
      origin[2] + dir[2] * this.grabDist,
    ];
    p.userPlaced = true;
    p.localPos = this.worldToLocal(world);
    if (scaleDelta) {
      p.scale = clamp(
        p.scale * (1 + scaleDelta), PANEL_MIN_SCALE, PANEL_MAX_SCALE,
      );
    }
    this.rebuildModels();
  }

  /** Release the carried panel and persist the layout. */
  endGrab(): void {
    this.grabbedCam = null;
    this.saveLayout();
  }

  /** Flow every panel back into the auto row and clear the saved layout. */
  resetLayout(): void {
    for (const p of this.videos.values()) {
      p.userPlaced = false;
      p.localPos = [0, 0, 0];
      p.scale = 1;
    }
    this.grabbedCam = null;
    if (this.isPlaced) this.rebuildModels();
    this.saveLayout();
  }

  private layoutStorageKey(): string {
    return LAYOUT_STORAGE_PREFIX + [...this.videos.keys()].sort().join(',');
  }

  private applySavedLayout(): void {
    try {
      const raw = localStorage.getItem(this.layoutStorageKey());
      if (!raw) return;
      const saved = JSON.parse(raw) as Record<
        string, { lp: number[]; scale: number }
      >;
      for (const [cam, p] of this.videos) {
        const s = saved[cam];
        if (s && Array.isArray(s.lp) && s.lp.length === 3) {
          p.userPlaced = true;
          p.localPos = [s.lp[0], s.lp[1], s.lp[2]];
          p.scale = s.scale || 1;
        }
      }
    } catch {
      /* localStorage unavailable / malformed — fall back to the auto row */
    }
  }

  private saveLayout(): void {
    try {
      const out: Record<string, { lp: number[]; scale: number }> = {};
      for (const [cam, p] of this.videos) {
        if (p.userPlaced) out[cam] = { lp: p.localPos, scale: p.scale };
      }
      const key = this.layoutStorageKey();
      if (Object.keys(out).length === 0) localStorage.removeItem(key);
      else localStorage.setItem(key, JSON.stringify(out));
    } catch {
      /* best-effort persistence only */
    }
  }

  // -------------------------------------------------------------------
  // Texture updates (called from WS handler / render loop, not per-frame)
  // -------------------------------------------------------------------

  setVideoFrame(cam: string, bmp: ImageBitmap): void {
    const gl = this.gl;
    let panel = this.videos.get(cam);
    if (!panel) {
      // Frame beat the camera_list — create the panel on the fly.
      this.setCameras([...this.videos.keys(), cam]);
      panel = this.videos.get(cam);
    }
    if (!panel) {
      bmp.close();
      return;
    }
    const aspect = bmp.width > 0 ? bmp.height / bmp.width : panel.aspect;
    gl.bindTexture(gl.TEXTURE_2D, panel.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
    bmp.close();
    panel.hasFrame = true;
    panel.lastFrameAt = performance.now();
    if (Math.abs(aspect - panel.aspect) > 1e-3) {
      panel.aspect = aspect;
      if (this.isPlaced) this.rebuildModels();
    }
  }

  setHud(snap: HudSnapshot): void {
    const key = JSON.stringify(snap);
    if (key === this.lastHudKey) return;
    this.lastHudKey = key;
    const ctx = this.hudCtx;
    const gl = this.gl;
    if (!ctx || !this.hudTex) return;

    const W = HUD_CANVAS_W;
    const H = HUD_CANVAS_H;
    ctx.clearRect(0, 0, W, H);
    // Panel background.
    ctx.fillStyle = 'rgba(10, 12, 16, 0.85)';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(250, 200, 60, 0.5)';
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, W - 4, H - 4);

    const anyEngaged = snap.bimanual
      ? snap.engagedLeft || snap.engagedRight
      : snap.engaged;
    const allReady = snap.bimanual
      ? snap.robotReadyLeft && snap.robotReadyRight
      : snap.robotReady;

    // Headline: calibration takes priority (the whole point of the HUD),
    // then engagement state.
    let headline: string;
    let headlineColor = '#facc3c';
    if (snap.calibState === 'capturing') {
      headline = 'CALIBRATING — rotate wrists, hold pivots still';
    } else if (snap.calibState === 'failed') {
      headline = 'calibration FAILED — squeeze grips to retry';
      headlineColor = '#f87171';
    } else if (snap.calibState === 'required') {
      headline = 'calibrate: squeeze grips to start (5s)';
    } else if (anyEngaged) {
      headline = 'ENGAGED';
    } else if (allReady) {
      headline = 'ready — hold grip to engage';
    } else {
      headline = 'waiting for robot';
      headlineColor = '#f87171';
    }
    ctx.fillStyle = headlineColor;
    ctx.font = 'bold 52px monospace';
    ctx.textBaseline = 'top';
    ctx.fillText(headline, 36, 40, W - 72);

    ctx.font = '36px monospace';
    let y = 140;
    const line = (label: string, value: string, ok: boolean) => {
      ctx.fillStyle = '#8a92a0';
      ctx.fillText(label, 36, y);
      ctx.fillStyle = ok ? '#e5e9f0' : '#f87171';
      ctx.fillText(value, 320, y, W - 356);
      y += 56;
    };
    if (snap.bimanual) {
      line(
        'left arm',
        snap.robotReadyLeft ? (snap.engagedLeft ? 'engaged' : 'live') : (snap.robotReasonLeft ?? 'no signal'),
        snap.robotReadyLeft,
      );
      line(
        'right arm',
        snap.robotReadyRight ? (snap.engagedRight ? 'engaged' : 'live') : (snap.robotReasonRight ?? 'no signal'),
        snap.robotReadyRight,
      );
    } else {
      line(
        'robot',
        snap.robotReady ? 'live' : (snap.robotReason ?? 'no signal'),
        snap.robotReady,
      );
      if (snap.ikPosErrMm != null) {
        line('ik err', `${snap.ikPosErrMm.toFixed(0)} mm`, true);
      }
    }
    const video =
      snap.videoAgeMs == null
        ? 'no video'
        : snap.videoAgeMs > VIDEO_STALE_MS
          ? `${snap.cameraCount} cam (stale, ${(snap.videoAgeMs / 1000).toFixed(1)}s)`
          : `${snap.cameraCount} cam live`;
    line('video', video, snap.videoAgeMs != null && snap.videoAgeMs <= VIDEO_STALE_MS);

    // ── camera layout hint (only while disengaged, left controller) ──
    if (snap.layoutActive) {
      ctx.fillStyle = '#7dd3fc';
      const hint = snap.layoutGrabbing
        ? `moving ${snap.layoutTarget ?? 'camera'} — stick: depth/size, release to drop`
        : snap.layoutTarget
          ? `${snap.layoutTarget}: left trigger to grab`
          : 'layout: aim left controller at a camera (stick-click resets)';
      ctx.fillText(hint, 36, y, W - 72);
      y += 56;
    }

    // ── latency report (last stats window) ──
    y += 8;
    ctx.strokeStyle = 'rgba(138, 146, 160, 0.35)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(36, y);
    ctx.lineTo(W - 36, y);
    ctx.stroke();
    y += 16;
    ctx.fillStyle = '#8a92a0';
    ctx.font = 'bold 30px monospace';
    const r = snap.videoReport;
    ctx.fillText(
      r ? `VIDEO LATENCY (last ${r.windowS}s)` : 'VIDEO LATENCY',
      36, y,
    );
    y += 46;
    ctx.font = '32px monospace';
    const rline = (label: string, value: string) => {
      ctx.fillStyle = '#8a92a0';
      ctx.fillText(label, 36, y);
      ctx.fillStyle = '#e5e9f0';
      ctx.fillText(value, 320, y, W - 356);
      y += 46;
    };
    if (r) {
      rline('rate', `${r.fpsPerCam.toFixed(1)} fps/cam · ${r.bytesAvgKb.toFixed(0)} KB/frame`);
      if (r.ageAvgMs != null) {
        rline('age', `avg ${r.ageAvgMs} ms · max ${r.ageMaxMs} ms (glass→eye)`);
      }
      rline('lag', `avg ${r.lagAvgMs} ms · max ${r.lagMaxMs} ms (above fastest)`);
      if (r.nodeLagAvgMs != null && r.podLagAvgMs != null) {
        rline('path', `node→pod ${r.nodeLagAvgMs} ms · pod→eye ${r.podLagAvgMs} ms`);
      }
      rline('decode', `avg ${r.decodeAvgMs.toFixed(1)} ms · max ${r.decodeMaxMs.toFixed(1)} ms`);
      rline('drops', r.dropped === 0 ? 'none' : `${r.dropped} (decode busy)`);
    } else {
      ctx.fillStyle = '#5a6270';
      ctx.fillText('collecting…', 36, y);
      y += 46;
    }

    ctx.fillStyle = '#5a6270';
    ctx.font = '28px monospace';
    ctx.fillText(`seq ${snap.seq}`, 36, H - 56);

    gl.bindTexture(gl.TEXTURE_2D, this.hudTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.hudCanvas);
  }

  // -------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------

  private drawQuad(
    model: Mat4,
    halfW: number,
    halfH: number,
    tex: WebGLTexture | null,
    opacity: number,
  ) {
    const gl = this.gl;
    if (!tex || !this.quadBuf) return;
    // Interleaved x,y,z,u,v — two triangles. V flipped: canvas/bitmap
    // uploads put row 0 at the top, GL UV 0 at the bottom.
    // prettier-ignore
    const verts = new Float32Array([
      -halfW, -halfH, 0, 0, 1,
       halfW, -halfH, 0, 1, 1,
       halfW,  halfH, 0, 1, 0,
      -halfW, -halfH, 0, 0, 1,
       halfW,  halfH, 0, 1, 0,
      -halfW,  halfH, 0, 0, 0,
    ]);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aPos);
    gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(this.aUv);
    gl.vertexAttribPointer(this.aUv, 2, gl.FLOAT, false, 20, 12);
    gl.uniformMatrix4fv(this.uModel, false, model);
    gl.uniform1f(this.uOpacity, opacity);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(this.uTex, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  render(layer: XRWebGLLayerLike, pose: XRViewerPoseLike): void {
    const gl = this.gl;
    if (!this.program) {
      // Shader failed to build — fall back to the bare clear so the XR
      // runtime keeps delivering frames.
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      return;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.CULL_FACE);

    const now = performance.now();
    const views = pose.views;
    for (let i = 0; i < views.length; i++) {
      const view = views[i];
      const vp = layer.getViewport(view);
      if (!vp) continue;
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      gl.uniformMatrix4fv(this.uProj, false, view.projectionMatrix);
      gl.uniformMatrix4fv(this.uView, false, view.transform.inverse.matrix);
      for (const [name, panel] of this.videos) {
        const stale = panel.hasFrame && now - panel.lastFrameAt > VIDEO_STALE_MS;
        // The panel being carried stays fully opaque so it reads as active.
        const opacity =
          name === this.grabbedCam ? 1.0 : panel.hasFrame ? (stale ? 0.35 : 1.0) : 0.9;
        this.drawQuad(panel.model, panel.halfW, panel.halfH, panel.tex, opacity);
      }
      this.drawQuad(this.hudModel, HUD_WIDTH_M / 2, this.hudHalfH, this.hudTex, 1.0);
    }
  }

  dispose(): void {
    const gl = this.gl;
    if (this.program) gl.deleteProgram(this.program);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    for (const panel of this.videos.values()) gl.deleteTexture(panel.tex);
    this.videos.clear();
    if (this.hudTex) gl.deleteTexture(this.hudTex);
    this.program = null;
    this.quadBuf = null;
    this.hudTex = null;
    this.isPlaced = false;
  }
}
