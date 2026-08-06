// Arm-count resolution for a served kinematic spec.
//
// Regression for a production teleop outage: `interlatent_robots/a1z` and
// `so101` are SINGLE-ARM rigs, but their `kinematic_spec.json` still ships the
// bimanual `chains` wrapper with only the `right` side populated. The browser
// treated "has a `chains` key" as "is bimanual" and built both solvers, so
// `new DlsSolver(spec.chains.left)` received `undefined` and threw
//
//     TypeError: Cannot read properties of undefined (reading 'n_ik_joints')
//
// from inside `setSpec`'s try block. That surfaced to the operator as the
// generic "QUIC connection failed" — with a healthy QUIC session the whole
// time. Bimanual rigs (nori, yam) carry both sides, which is why teleop
// worked on a yam and only ever failed on the a1z.
//
// The rule under test: arm count comes from the sides actually PRESENT, never
// from the wrapper existing.
import { describe, expect, it } from 'vitest';

import { KinematicSpecBundle, resolveSpecArms } from '../kinematics';
import { planar3R } from './specFixtures';

describe('resolveSpecArms', () => {
  it('treats a flat spec as single-arm', () => {
    const flat = planar3R();
    const arms = resolveSpecArms(flat);
    expect(arms.bimanual).toBe(false);
    if (!arms.bimanual) expect(arms.single).toBe(flat);
  });

  it('treats a chains wrapper with only `right` as single-arm (the a1z shape)', () => {
    const right = planar3R();
    const arms = resolveSpecArms({ version: 1, chains: { right } });
    expect(arms.bimanual).toBe(false);
    if (!arms.bimanual) expect(arms.single).toBe(right);
  });

  it('treats a chains wrapper with only `left` as single-arm', () => {
    const left = planar3R();
    const arms = resolveSpecArms({ version: 1, chains: { left } });
    expect(arms.bimanual).toBe(false);
    if (!arms.bimanual) expect(arms.single).toBe(left);
  });

  it('treats a chains wrapper with both sides as bimanual', () => {
    const left = planar3R();
    const right = planar3R();
    const arms = resolveSpecArms({ version: 1, chains: { left, right } });
    expect(arms.bimanual).toBe(true);
    if (arms.bimanual) {
      expect(arms.left).toBe(left);
      expect(arms.right).toBe(right);
    }
  });

  it('never yields an undefined chain — the exact crash that shipped', () => {
    // Pre-fix this path produced `new DlsSolver(undefined)`; the guarantee is
    // that every arm handed back is a real spec with `n_ik_joints` readable.
    for (const bundle of [
      planar3R(),
      { version: 1, chains: { right: planar3R() } },
      { version: 1, chains: { left: planar3R() } },
      { version: 1, chains: { left: planar3R(), right: planar3R() } },
    ] as KinematicSpecBundle[]) {
      const arms = resolveSpecArms(bundle);
      const chains = arms.bimanual ? [arms.left, arms.right] : [arms.single];
      for (const c of chains) {
        expect(c).toBeDefined();
        expect(typeof c.n_ik_joints).toBe('number');
      }
    }
  });

  it('throws a named error rather than a TypeError when no arm is present', () => {
    expect(() => resolveSpecArms({ version: 1, chains: {} })).toThrow(/no arm/);
  });
});
