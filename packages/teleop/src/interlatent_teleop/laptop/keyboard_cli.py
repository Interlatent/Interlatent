"""Keyboard teleop CLI.

Two modes (select with --mode):

  joint     (default; recommended)
            One key per joint. Hold to move continuously, release to
            stop. No IK in the loop, so it works regardless of how
            your SO-101 was calibrated — useful both for normal use
            and for diagnosing what each joint actually does on your
            assembly.

              A D     shoulder_pan       -/+
              W S     shoulder_lift      +/-
              E C     elbow_flex         +/-
              I K     wrist_flex         +/-
              J L     wrist_roll         -/+
              SPACE N gripper            close / open
              [ ]     gripper            close / open (aliases)
              R       reset to home pose
              TAB     toggle deadman (or --auto-arm)
              SHIFT   3x speed while held
              ESC     quit

            If a joint moves the wrong direction on your assembly, flip
            it with --joint-key-signs (e.g. '1,-1,1,1,1,1' inverts
            shoulder_lift so W actually lifts up).

  cartesian Position + pitch + roll via IK in kinematics.ik_full.
            Requires the FK math frame to match your motor calibration
            — works if it does, breaks if it doesn't. Same key bindings
            as joint mode but they move the gripper in 3D space.

Held-key model:

  Input comes from a pynput global listener so we can advance the
  target trajectory each tick by `rate * dt` while a key is physically
  held. This produces a smooth ramp instead of the OS-key-repeat
  staircase you get with cv2.waitKey alone.

  macOS: the first run will prompt for Accessibility permission. Until
  you grant it (System Settings → Privacy & Security → Accessibility →
  enable for your terminal), the listener won't see any keys.

  Because the listener is global, keystrokes in *other* apps will also
  drive the arm while it's armed. Keep deadman on (SPACE) unless you're
  sure that's fine, and prefer SPACE-toggle over --auto-arm.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import sys
import threading
import time

import numpy as np

from .client import TeleopClient
from .kinematics import SO101Kinematics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="interlatent-teleop-keyboard")
    parser.add_argument("--pi", required=True, help="Pi gRPC address, e.g. 100.x.y.z:50061")
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument("--client-id", default=os.environ.get("USER", "keyboard"))
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--session-token", default=os.environ.get("INTERLATENT_TELEOP_TOKEN", ""))
    parser.add_argument("--mode", choices=("joint", "cartesian"), default="joint",
                        help="control mode. 'joint' = one key per motor (recommended; no IK). "
                             "'cartesian' = position + pitch + roll via IK (requires calibrated FK).")
    parser.add_argument("--auto-arm", action="store_true",
                        help="treat the deadman as always armed (no SPACE needed). "
                             "Unsafe with the pynput global listener — your typing in "
                             "other apps will move the arm.")
    parser.add_argument("--tick-hz", type=float, default=120.0,
                        help="main loop tick rate. Higher = finer trajectory ramps. "
                             "Also drives the send rate (capped at --send-hz).")
    parser.add_argument("--send-hz", type=float, default=60.0,
                        help="rate at which targets are sent to the Pi.")
    parser.add_argument("--joint-rate-deg-per-s", type=float, default=60.0,
                        help="how fast each joint's target advances while its key is "
                             "held, in deg/sec. Replaces the old per-keypress step "
                             "model with continuous time-based motion. SHIFT multiplies "
                             "this by --shift-mul.")
    parser.add_argument("--gripper-rate-pct-per-s", type=float, default=120.0,
                        help="how fast the gripper target advances while a gripper "
                             "key is held, in percent/sec (0=closed, 100=open).")
    parser.add_argument("--gripper-invert", action="store_true",
                        help="reverse SPACE/N (and [/]) so SPACE opens and N closes. "
                             "Only needed if your arm's gripper calibration is reversed "
                             "from lerobot's 0=closed/100=open convention.")
    parser.add_argument("--max-joint-lead-deg", type=float, default=45.0,
                        help="how far ahead of the actual joint the commanded target "
                             "can drift in joint mode. Smaller = arm catches up to the "
                             "command faster on release; larger = the user can pile up "
                             "more lead while the arm is slow to physically respond. "
                             "Set very high (e.g. 360) to effectively disable.")
    parser.add_argument("--joint-key-signs", type=str, default="-1,-1,-1,-1,-1,-1",
                        help="comma-separated +1/-1 per joint, in order "
                             "(pan, lift, elbow, wflex, wroll, gripper). Flips the "
                             "direction of the corresponding keypress so the keymap "
                             "matches your assembly. Default is all -1 because the "
                             "stock lerobot SO-101 calibration runs every joint's "
                             "positive direction opposite to the natural keymap "
                             "intent (W=up, A=left, etc.). Pass '1,1,1,1,1,1' to "
                             "use raw motor signs; flip individual entries if only "
                             "some joints disagree with the keymap on your arm.")
    parser.add_argument("--xy-rate-m-per-s", type=float, default=0.10,
                        help="cartesian XYZ target rate while a translation key is held.")
    parser.add_argument("--pitch-rate-deg-per-s", type=float, default=45.0,
                        help="cartesian pitch rate while I/K is held.")
    parser.add_argument("--roll-rate-deg-per-s", type=float, default=60.0,
                        help="cartesian roll rate while J/L is held.")
    parser.add_argument("--shift-mul", type=float, default=3.0,
                        help="rate multiplier while SHIFT is held.")
    parser.add_argument("--max-lead-m", type=float, default=0.015,
                        help="target can lead actual arm position by at most this many "
                             "meters (cartesian).")
    parser.add_argument("--max-lead-pitch-deg", type=float, default=12.0,
                        help="target pitch can lead actual by at most this much (cartesian).")
    parser.add_argument("--max-lead-roll-deg", type=float, default=15.0,
                        help="target roll can lead actual by at most this much (cartesian).")
    # Sign overrides for the motor↔math convention. Defaults match the
    # lerobot-kinematics SO-101 model. Flip a sign if a joint moves the
    # wrong direction on your assembly.
    parser.add_argument("--lift-sign", type=float, default=1.0)
    parser.add_argument("--elbow-sign", type=float, default=1.0)
    parser.add_argument("--wflex-sign", type=float, default=1.0)
    parser.add_argument("--wroll-sign", type=float, default=1.0)
    parser.add_argument("--pan-sign", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import cv2
    except ImportError:
        print("opencv-python is required: pip install opencv-python", file=sys.stderr)
        return 2

    try:
        from pynput import keyboard as pkb
    except ImportError:
        print("pynput is required for held-key teleop. Install with: "
              "pip install 'interlatent-teleop[laptop]'  (or: pip install pynput)",
              file=sys.stderr)
        return 2

    kin = SO101Kinematics(
        pan_sign=args.pan_sign,
        lift_sign=args.lift_sign,
        elbow_sign=args.elbow_sign,
        wflex_sign=args.wflex_sign,
        wroll_sign=args.wroll_sign,
    )

    client = TeleopClient(
        address=args.pi, client_id=args.client_id, robot_id=args.robot_id,
        session_token=args.session_token, control_hz=args.control_hz,
    )
    session = client.open()
    logging.info("connected; home pose=%s", np.round(session.home_joints, 1).tolist())

    home_joints = session.home_joints.astype(np.float32)
    home_xyz = kin.fk(home_joints)
    home_pitch = kin.gripper_pitch(home_joints)
    home_roll_deg = float(home_joints[4])  # wrist_roll is direct

    target_joints = home_joints.copy()
    target_xyz = home_xyz.copy()
    target_pitch = float(home_pitch)
    target_roll = math.radians(home_roll_deg)
    gripper_pct = float(home_joints[5])
    deadman_held = False

    canvas = np.full((320, 600, 3), 245, dtype=np.uint8)

    def put_text(img, text, pos, scale=0.5, thickness=1, fill=(20, 20, 20)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    fill, thickness, cv2.LINE_AA)

    max_lead_pitch_rad = math.radians(args.max_lead_pitch_deg)
    max_lead_roll_rad = math.radians(args.max_lead_roll_deg)

    try:
        joint_key_signs = [float(s.strip()) for s in args.joint_key_signs.split(",")]
        if len(joint_key_signs) != len(session.joint_names):
            raise ValueError("wrong length")
    except Exception as e:  # noqa: BLE001
        logging.error("invalid --joint-key-signs %r: %s; defaulting to all -1",
                      args.joint_key_signs, e)
        joint_key_signs = [-1.0] * len(session.joint_names)
    logging.info("joint_key_signs = %s", joint_key_signs)

    # --- pynput held-key state ----------------------------------------
    # `held` is the set of characters currently physically held down
    # (plus 'shift' for either shift modifier). `discrete_q` is a queue
    # of one-shot events (esc/space/r) the main loop drains each tick.
    held: set[str] = set()
    discrete_q: "queue.Queue[str]" = queue.Queue()
    held_lock = threading.Lock()

    # Continuous-while-held keys. SPACE and N are also held, but they
    # come through pynput as Key.space (no char) and 'n' respectively;
    # SPACE is handled separately in on_press/on_release below.
    HELD_CHARS = set("wasdeciklj[]n")

    def _char(key) -> str | None:
        try:
            if key.char is None:
                return None
            return key.char.lower()
        except AttributeError:
            return None

    def on_press(key):
        if key == pkb.Key.esc:
            discrete_q.put("esc")
            return False  # stop listener
        if key == pkb.Key.tab:
            discrete_q.put("deadman_toggle")
            return
        if key == pkb.Key.space:
            # Continuous gripper close while held.
            with held_lock:
                held.add("close")
            return
        if key in (pkb.Key.shift, pkb.Key.shift_l, pkb.Key.shift_r):
            with held_lock:
                held.add("shift")
            return
        c = _char(key)
        if c is None:
            return
        if c == "r":
            discrete_q.put("r")
            return
        if c in HELD_CHARS:
            with held_lock:
                held.add(c)

    def on_release(key):
        if key == pkb.Key.space:
            with held_lock:
                held.discard("close")
            return
        if key in (pkb.Key.shift, pkb.Key.shift_l, pkb.Key.shift_r):
            with held_lock:
                held.discard("shift")
            return
        c = _char(key)
        if c is None:
            return
        with held_lock:
            held.discard(c)

    listener = pkb.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    logging.info("pynput listener started; on macOS you may need to grant "
                 "Accessibility permission to your terminal for keys to register.")

    # Joint mode: key -> (joint_idx, direction_sign)
    JOINT_KEYS = {
        "a": (0, -1), "d": (0, +1),
        "w": (1, +1), "s": (1, -1),
        "e": (2, +1), "c": (2, -1),
        "i": (3, +1), "k": (3, -1),
        "j": (4, -1), "l": (4, +1),
    }

    tick_period = 1.0 / max(1.0, args.tick_hz)
    send_period = 1.0 / max(1.0, args.send_hz)
    last_send = 0.0
    last_tick = time.monotonic()
    quit_requested = False

    try:
        while not quit_requested:
            now = time.monotonic()
            dt = now - last_tick
            # Cap dt so a hiccup (window drag, GC pause) doesn't fling
            # the target far ahead in one tick.
            if dt > 0.1:
                dt = 0.1
            last_tick = now

            # --- drain discrete events
            try:
                while True:
                    ev = discrete_q.get_nowait()
                    if ev == "esc":
                        quit_requested = True
                    elif ev == "deadman_toggle":
                        deadman_held = not deadman_held
                        logging.info("deadman %s", "ARMED" if deadman_held else "safe")
                    elif ev == "r":
                        target_joints = home_joints.copy()
                        target_xyz = home_xyz.copy()
                        target_pitch = float(home_pitch)
                        target_roll = math.radians(home_roll_deg)
                        gripper_pct = float(home_joints[5])
                        logging.info("reset to home")
            except queue.Empty:
                pass

            with held_lock:
                held_snapshot = set(held)
            mul = args.shift_mul if "shift" in held_snapshot else 1.0

            # --- continuous gripper (both modes)
            # SPACE / '[' = close, N / ']' = open. The gripper uses
            # lerobot's fixed 0=closed / 100=open convention, so we do
            # NOT apply joint_key_signs[5] here — that would double-flip
            # against a default of -1 (which is correct for rotational
            # joints on this calibration but not for the gripper). If
            # your specific arm reads gripper backwards, flip with
            # --gripper-invert.
            grip_rate = args.gripper_rate_pct_per_s * mul
            if args.gripper_invert:
                grip_rate = -grip_rate
            open_held = ("n" in held_snapshot) or ("]" in held_snapshot)
            close_held = ("close" in held_snapshot) or ("[" in held_snapshot)
            if open_held:
                gripper_pct = float(np.clip(gripper_pct + grip_rate * dt, 0.0, 100.0))
            if close_held:
                gripper_pct = float(np.clip(gripper_pct - grip_rate * dt, 0.0, 100.0))

            ack = client.latest_ack
            if ack is not None and len(ack.current_joints) == len(session.joint_names):
                actual_joints = np.array(ack.current_joints, dtype=np.float32)
            else:
                actual_joints = home_joints

            if args.mode == "joint":
                rate = args.joint_rate_deg_per_s * mul
                for k, (idx, sign) in JOINT_KEYS.items():
                    if k in held_snapshot:
                        target_joints[idx] += sign * joint_key_signs[idx] * rate * dt
                # Sync gripper target with gripper_pct.
                target_joints[5] = gripper_pct
                # Lead-clip so target can't run too far ahead of actual.
                for i in range(len(target_joints)):
                    delta = target_joints[i] - actual_joints[i]
                    if abs(delta) > args.max_joint_lead_deg:
                        target_joints[i] = actual_joints[i] + math.copysign(
                            args.max_joint_lead_deg, delta
                        )
                joints = target_joints.copy()
            else:
                # Cartesian mode: continuous XYZ + pitch + roll.
                xy_rate = args.xy_rate_m_per_s * mul
                p_rate = math.radians(args.pitch_rate_deg_per_s) * mul
                r_rate = math.radians(args.roll_rate_deg_per_s) * mul
                if "w" in held_snapshot: target_xyz[0] += xy_rate * dt
                if "s" in held_snapshot: target_xyz[0] -= xy_rate * dt
                if "a" in held_snapshot: target_xyz[1] += xy_rate * dt
                if "d" in held_snapshot: target_xyz[1] -= xy_rate * dt
                if "e" in held_snapshot: target_xyz[2] += xy_rate * dt
                if "c" in held_snapshot: target_xyz[2] -= xy_rate * dt
                if "i" in held_snapshot: target_pitch += p_rate * dt
                if "k" in held_snapshot: target_pitch -= p_rate * dt
                if "j" in held_snapshot: target_roll  -= r_rate * dt
                if "l" in held_snapshot: target_roll  += r_rate * dt

                actual_xyz = kin.fk(actual_joints)
                actual_pitch = kin.gripper_pitch(actual_joints)
                actual_roll = math.radians(float(actual_joints[4]))

                xyz_delta = target_xyz - actual_xyz
                xyz_mag = float(np.linalg.norm(xyz_delta))
                if xyz_mag > args.max_lead_m:
                    target_xyz = actual_xyz + xyz_delta * (args.max_lead_m / xyz_mag)

                pitch_delta = target_pitch - actual_pitch
                if abs(pitch_delta) > max_lead_pitch_rad:
                    target_pitch = actual_pitch + math.copysign(max_lead_pitch_rad, pitch_delta)

                roll_delta = target_roll - actual_roll
                if abs(roll_delta) > max_lead_roll_rad:
                    target_roll = actual_roll + math.copysign(max_lead_roll_rad, roll_delta)

                ik = kin.ik_full(
                    target_xyz=target_xyz,
                    target_pitch_rad=target_pitch,
                    target_roll_rad=target_roll,
                    current_joints=actual_joints,
                )
                joints = home_joints.copy()
                joints[0:5] = ik
                joints[5] = gripper_pct

            joints = np.clip(joints, session.joint_min, session.joint_max)

            if now - last_send >= send_period:
                armed = deadman_held or args.auto_arm
                client.send_target(joints=joints, deadman=armed, confidence=1.0)
                last_send = now

            # --- render
            canvas[:] = 245
            armed_label = "ARMED" if (deadman_held or args.auto_arm) else "safe"
            armed_color = (0, 130, 0) if (deadman_held or args.auto_arm) else (0, 80, 200)
            if args.auto_arm:
                armed_label += " (auto)"
            put_text(canvas, f"deadman: {armed_label}    mode: {args.mode}", (10, 30),
                     scale=0.7, thickness=2, fill=armed_color)
            if args.mode == "joint":
                help_line = ("AD=pan WS=lift EC=elbow IK=wflex JL=wroll  "
                             "SPACE=close N=open  SHIFT=3x  TAB=deadman  R=reset  ESC=quit")
            else:
                help_line = ("WASD=XY  EC=up/dn  IK=pitch  JL=roll  "
                             "SPACE=close N=open  SHIFT=3x  TAB=deadman  R=reset  ESC=quit")
            put_text(canvas, help_line, (10, 58), scale=0.38)

            if args.mode == "joint":
                d = target_joints - home_joints
                put_text(canvas,
                         f"target delta (deg): pan={d[0]:+5.1f} lift={d[1]:+5.1f} "
                         f"elb={d[2]:+5.1f} wfl={d[3]:+5.1f} wrl={d[4]:+5.1f}",
                         (10, 90), scale=0.45, fill=(120, 0, 80))
            else:
                d = (target_xyz - home_xyz) * 1000
                put_text(canvas,
                         f"pos delta (mm):  fwd={d[0]:+5.0f}   lat={d[1]:+5.0f}   up={d[2]:+5.0f}",
                         (10, 90), scale=0.5, fill=(120, 0, 80))
                put_text(canvas,
                         f"orient (deg):    pitch={math.degrees(target_pitch):+5.0f}   "
                         f"roll={math.degrees(target_roll):+5.0f}",
                         (10, 115), scale=0.5, fill=(120, 60, 0))

            put_text(canvas, f"gripper: {gripper_pct:5.1f} / 100", (10, 140), scale=0.5)
            held_chars = sorted(c for c in held_snapshot
                                if c not in ("shift", "close"))
            parts = []
            if "close" in held_snapshot:
                parts.append("SPACE")
            parts.extend(held_chars)
            held_str = " ".join(parts) or "-"
            if "shift" in held_snapshot:
                held_str = "SHIFT+" + held_str
            put_text(canvas, f"held: {held_str}", (10, 165), scale=0.45, fill=(0, 60, 120))
            tgt_str = " ".join(f"{n[:3]}={v:+5.0f}" for n, v in zip(session.joint_names, joints))
            put_text(canvas, f"tgt: {tgt_str}", (10, 190), scale=0.42, fill=(0, 100, 0))
            if ack is not None:
                cur_str = " ".join(f"{n[:3]}={v:+5.0f}"
                                   for n, v in zip(session.joint_names, ack.current_joints))
                put_text(canvas, f"cur: {cur_str}", (10, 215), scale=0.42, fill=(120, 0, 0))
                put_text(canvas, f"pi:  {ack.status_message}", (10, 245), scale=0.5)
            cv2.imshow("interlatent-teleop-keyboard", canvas)
            # cv2.waitKey is only here to pump the GUI event loop so the
            # window renders and stays responsive. Key state itself comes
            # from the pynput listener above.
            cv2.waitKey(1)

            # Sleep until next tick.
            elapsed = time.monotonic() - now
            sleep_t = tick_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            listener.stop()
        except Exception:  # noqa: BLE001
            pass
        client.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
