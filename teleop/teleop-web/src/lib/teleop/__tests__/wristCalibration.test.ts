// Tests for the in-VR wrist-pivot calibration solve.
//
// WebXR reports the palm, but the operator rotates about their wrist. Feed
// the palm pose straight into the clutch mapper and every pure wrist twist
// also sweeps the readout point through an arc, which the IK chases as
// ghost translation. The solve recovers the controller-local offset that
// makes `p + R·o` stationary.
//
// The gates matter as much as the solve: a capture where the arm actually
// moved yields a plausible-looking offset with a large residual, and
// accepting it would bake a permanent error into every subsequent session.
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { Mat3, Quat, Vec3 } from '../quat';
import {
  CALIB_GRIP_THRESHOLD,
  CALIB_OFFSET_MAX,
  CALIB_RESIDUAL_MAX,
  DEFAULT_WRIST_OFFSET,
  PivotSample,
  clearWristOffsets,
  loadWristOffsets,
  pivotPasses,
  quatToMat3,
  saveWristOffsets,
  solvePivot,
} from '../wristCalibration';

/** Deterministic LCG — the capture has to be reproducible across runs. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function axisAngle(axis: Vec3, angle: number): Quat {
  const s = Math.sin(angle / 2);
  const n = Math.hypot(...axis);
  return [(axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s, Math.cos(angle / 2)];
}

function matVec3(R: Mat3, v: Vec3): Vec3 {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

/**
 * Synthesize a capture: the wrist pivot sits still at `pivot` while the
 * controller is freely rotated, so `p = pivot - R·o` for the true offset.
 * `jitter` adds hand tremor (or a drifting arm, at large values).
 */
function capture(
  o: Vec3,
  { n = 120, pivot = [0.1, -0.2, 0.3] as Vec3, jitter = 0, seed = 7 } = {},
): PivotSample[] {
  const rand = rng(seed);
  const out: PivotSample[] = [];
  for (let i = 0; i < n; i++) {
    const q = axisAngle(
      [rand() - 0.5, rand() - 0.5, rand() - 0.5],
      (rand() - 0.5) * 2.0,
    );
    const R = quatToMat3(q);
    const Ro = matVec3(R, o);
    out.push({
      R,
      p: [
        pivot[0] - Ro[0] + (rand() - 0.5) * jitter,
        pivot[1] - Ro[1] + (rand() - 0.5) * jitter,
        pivot[2] - Ro[2] + (rand() - 0.5) * jitter,
      ],
    });
  }
  return out;
}

describe('quatToMat3', () => {
  it('turns identity into the identity matrix', () => {
    expect(quatToMat3([0, 0, 0, 1])).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  });

  it('produces a proper rotation (orthonormal, det +1)', () => {
    const R = quatToMat3(axisAngle([0.3, -0.7, 0.5], 1.2));
    for (let i = 0; i < 3; i++) {
      expect(Math.hypot(...(R[i] as Vec3))).toBeCloseTo(1, 12);
      for (let j = i + 1; j < 3; j++) {
        const dot = R[i][0] * R[j][0] + R[i][1] * R[j][1] + R[i][2] * R[j][2];
        expect(dot).toBeCloseTo(0, 12);
      }
    }
    const det =
      R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
      R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
      R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    expect(det).toBeCloseTo(1, 12);
  });

  it('rotates +X to +Y for a +90° turn about +Z', () => {
    const R = quatToMat3(axisAngle([0, 0, 1], Math.PI / 2));
    const v = matVec3(R, [1, 0, 0]);
    expect(v[0]).toBeCloseTo(0, 12);
    expect(v[1]).toBeCloseTo(1, 12);
  });
});

describe('solvePivot', () => {
  it('recovers the offset from a clean capture', () => {
    const o: Vec3 = [0.01, -0.02, 0.06];
    const r = solvePivot(capture(o));
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    r.o.forEach((v, i) => expect(v).toBeCloseTo(o[i], 6));
    expect(r.rms).toBeLessThan(1e-9);
    expect(r.n).toBe(120);
  });

  it('recovers a plausible offset through hand tremor', () => {
    const o: Vec3 = [0.0, 0.0, 0.05];
    const r = solvePivot(capture(o, { jitter: 0.002, seed: 11 }));
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    r.o.forEach((v, i) => expect(Math.abs(v - o[i])).toBeLessThan(0.01));
    expect(pivotPasses(r)).toBe(true);
  });

  it('refuses a capture that is too short to constrain the solve', () => {
    const r = solvePivot(capture([0, 0, 0.05], { n: 29 }));
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.reason).toContain('29');
  });

  it('refuses an empty capture', () => {
    expect(solvePivot([]).ok).toBe(false);
  });

  it('refuses a capture with no rotation to solve against', () => {
    // Holding the controller still gives dR = 0 — the normal matrix is
    // singular and any offset fits equally well.
    const R = quatToMat3([0, 0, 0, 1]);
    const samples: PivotSample[] = Array.from({ length: 100 }, () => ({
      R, p: [0.1, 0.2, 0.3] as Vec3,
    }));
    const r = solvePivot(samples);
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.reason).toContain('ill-conditioned');
  });

  it('reports a large residual when the arm moved during the capture', () => {
    // The whole point of the residual: the solve still returns SOMETHING,
    // so the caller has to be able to tell a bad capture from a good one.
    const drifting = capture([0, 0, 0.05], { jitter: 0.2, seed: 3 });
    const r = solvePivot(drifting);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.rms).toBeGreaterThan(CALIB_RESIDUAL_MAX);
    expect(pivotPasses(r)).toBe(false);
  });
});

describe('pivotPasses', () => {
  const good = { ok: true as const, o: [0.01, 0, 0.05] as Vec3, rms: 0.001, n: 120 };

  it('accepts a still, small-offset solve', () => {
    expect(pivotPasses(good)).toBe(true);
  });

  it('rejects a failed solve outright', () => {
    expect(pivotPasses({ ok: false, reason: 'too few samples (3)' })).toBe(false);
  });

  it('rejects a residual over the gate', () => {
    expect(pivotPasses({ ...good, rms: CALIB_RESIDUAL_MAX + 1e-6 })).toBe(false);
    expect(pivotPasses({ ...good, rms: CALIB_RESIDUAL_MAX })).toBe(true);
  });

  it('rejects an implausibly large offset on any axis', () => {
    const over = CALIB_OFFSET_MAX + 0.01;
    expect(pivotPasses({ ...good, o: [over, 0, 0] })).toBe(false);
    expect(pivotPasses({ ...good, o: [0, -over, 0] })).toBe(false);
    expect(pivotPasses({ ...good, o: [0, 0, over] })).toBe(false);
  });
});

describe('constants', () => {
  it('keeps the calibration gates at their documented values', () => {
    expect(CALIB_RESIDUAL_MAX).toBe(0.015); // 15 mm RMS
    expect(CALIB_OFFSET_MAX).toBe(0.2);
    expect(CALIB_GRIP_THRESHOLD).toBeGreaterThan(0);
    expect(CALIB_GRIP_THRESHOLD).toBeLessThanOrEqual(1);
    expect(DEFAULT_WRIST_OFFSET).toHaveLength(3);
  });
});

describe('persistence', () => {
  function fakeStorage(initial: Record<string, string> = {}) {
    const store = { ...initial };
    return {
      store,
      getItem: (k: string) => (k in store ? store[k] : null),
      setItem: (k: string, v: string) => { store[k] = v; },
      removeItem: (k: string) => { delete store[k]; },
    };
  }

  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('round-trips both hands', () => {
    vi.stubGlobal('localStorage', fakeStorage());
    const offsets = { left: [0.01, 0, 0.05] as Vec3, right: [-0.01, 0.002, 0.04] as Vec3 };
    saveWristOffsets(offsets);
    expect(loadWristOffsets()).toEqual(offsets);
  });

  it('returns null when nothing is stored', () => {
    vi.stubGlobal('localStorage', fakeStorage());
    expect(loadWristOffsets()).toBeNull();
  });

  it('treats corrupted storage as uncalibrated', () => {
    vi.stubGlobal('localStorage', fakeStorage({
      'interlatent:vrteleop:wrist_offset_v1': '{not json',
    }));
    expect(loadWristOffsets()).toBeNull();
  });

  it('rejects a stored value of the wrong shape', () => {
    for (const bad of ['{"left":[1,2,3]}', '{"left":[1,2],"right":[1,2,3]}',
                       '{"left":"x","right":"y"}', 'null', '[]']) {
      vi.stubGlobal('localStorage', fakeStorage({
        'interlatent:vrteleop:wrist_offset_v1': bad,
      }));
      expect(loadWristOffsets()).toBeNull();
    }
  });

  it('clears the stored calibration', () => {
    const storage = fakeStorage();
    vi.stubGlobal('localStorage', storage);
    saveWristOffsets({ left: [0, 0, 0.05], right: [0, 0, 0.05] });
    clearWristOffsets();
    expect(loadWristOffsets()).toBeNull();
    expect(Object.keys(storage.store)).toHaveLength(0);
  });

  it('survives storage being unavailable', () => {
    // Private browsing / a blocked origin throws on every access. The
    // calibration should still apply for the session.
    const throwing = {
      getItem: () => { throw new Error('denied'); },
      setItem: () => { throw new Error('denied'); },
      removeItem: () => { throw new Error('denied'); },
    };
    vi.stubGlobal('localStorage', throwing);
    expect(loadWristOffsets()).toBeNull();
    expect(() => saveWristOffsets({ left: [0, 0, 0], right: [0, 0, 0] })).not.toThrow();
    expect(() => clearWristOffsets()).not.toThrow();
  });
});
