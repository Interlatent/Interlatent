"""CLI: `interlatent-teleop-laptop --pi <host:port>`.

Opens the camera, runs MediaPipe, retargets to SO-101 joint targets,
streams them to the Pi. Deadman: hold SPACE in the preview window to
arm motion. Press 'c' to (re)calibrate the neutral pose. Press 'q' or
ESC to quit.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import numpy as np

from .client import TeleopClient
from .hand_tracker import HandTracker
from .kinematics import SO101Kinematics
from .retargeting import RetargetConfig, Retargeter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="interlatent-teleop-laptop")
    parser.add_argument("--pi", required=True, help="Pi gRPC address, e.g. 100.x.y.z:50061")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument("--client-id", default=os.environ.get("USER", "laptop"))
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--session-token", default=os.environ.get("INTERLATENT_TELEOP_TOKEN", ""))
    parser.add_argument("--hand", choices=("Right", "Left"), default="Right")
    # Cartesian retargeter knobs.
    parser.add_argument("--workspace-scale", type=float, default=0.20,
                        help="meters of arm motion per unit of normalized hand motion. "
                             "Lower = precise (small arm motion per hand motion). "
                             "Higher = expansive (small hand motion drives big arm motion).")
    parser.add_argument("--depth-scale-boost", type=float, default=5.0,
                        help="gain on the depth signal (hand size in image -> arm height). "
                             "Higher = more height motion per cm of hand-toward-camera.")
    parser.add_argument("--deadband-m", type=float, default=0.008,
                        help="meters; hand jitter below this threshold (in robot space) is "
                             "ignored so the arm holds still when your hand is still. "
                             "0 disables.")
    parser.add_argument("--gripper-pitch-deg", type=float, default=None,
                        help="desired gripper pitch in degrees (math frame). Default: "
                             "preserve whatever pitch the gripper had at calibration. "
                             "Set e.g. -90 to force gripper-down, 0 for horizontal.")
    parser.add_argument("--no-mirror", action="store_true",
                        help="disable lateral mirroring. Set if 'hand right' makes the arm go "
                             "left (non-selfie camera).")
    parser.add_argument("--smoothing", type=float, default=0.6,
                        help="0..1 low-pass coefficient on retargeted joints. Higher = "
                             "smoother but laggier.")

    # Kinematics overrides (rarely needed; defaults match typical SO-101).
    parser.add_argument("--l1", type=float, default=0.108, help="upper arm length (m)")
    parser.add_argument("--l2", type=float, default=0.108, help="forearm length (m)")
    parser.add_argument("--l3", type=float, default=0.110, help="wrist-to-gripper-tip (m)")
    parser.add_argument("--base-height", type=float, default=0.080,
                        help="base plate to shoulder pivot (m)")
    parser.add_argument("--lift-sign", type=float, default=1.0,
                        help="motor sign for shoulder_lift; flip to -1 if up is down")
    parser.add_argument("--elbow-sign", type=float, default=-1.0,
                        help="motor sign for elbow_flex; flip if elbow folds wrong way")
    parser.add_argument("--wrist-sign", type=float, default=1.0,
                        help="motor sign for wrist_flex; flip if wrist auto-levels wrong way")
    parser.add_argument("--no-preview", action="store_true",
                        help="don't open a preview window (headless laptop)")
    parser.add_argument("--auto-arm", action="store_true",
                        help="treat any confidently-tracked hand as deadman-armed "
                             "(no SPACE press needed). Convenient for MVP; still safe "
                             "because losing the hand still releases motion and the Pi "
                             "freezes on staleness timeout.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Deferred so missing OpenCV / MediaPipe surfaces with a clear error.
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("opencv-python is required for the laptop CLI: pip install opencv-python",
              file=sys.stderr)
        return 2
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print("mediapipe is required for the laptop CLI: pip install mediapipe",
              file=sys.stderr)
        return 2
    import cv2

    tracker = HandTracker(camera_index=args.camera, preferred_handedness=args.hand)
    tracker.open()

    client = TeleopClient(
        address=args.pi,
        client_id=args.client_id,
        robot_id=args.robot_id,
        session_token=args.session_token,
        control_hz=args.control_hz,
    )
    session = client.open()
    logging.info("connected; home pose=%s", np.round(session.home_joints, 1).tolist())

    retargeter = Retargeter(
        config=RetargetConfig(
            workspace_scale=args.workspace_scale,
            depth_scale_boost=args.depth_scale_boost,
            deadband_m=args.deadband_m,
            gripper_pitch_rad=(np.deg2rad(args.gripper_pitch_deg)
                               if args.gripper_pitch_deg is not None else None),
            mirror_x=not args.no_mirror,
            smoothing=args.smoothing,
        ),
        kinematics=SO101Kinematics(
            L1=args.l1,
            L2=args.l2,
            L3=args.l3,
            base_height=args.base_height,
            lift_sign=args.lift_sign,
            elbow_sign=args.elbow_sign,
            wrist_sign=args.wrist_sign,
        ),
    )

    # Deadman is a keypress: tracked via cv2 waitKey state. The window
    # must be focused for keypresses to register. For headless laptops
    # without a display, --no-preview falls back to "deadman always on
    # while a hand is detected" (still safe because the Pi staleness
    # timeout freezes the arm when no targets arrive).
    deadman_held = False
    last_send = 0.0
    send_period = 1.0 / 30.0  # cap producer rate at 30 Hz
    last_target: Optional[np.ndarray] = None

    def put_text(img, text, pos, scale=0.5, thickness=1, fill=(0, 0, 0)):
        """Outlined text: thick white halo, then thin colored fill on top.
        Readable against any background."""
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    fill, thickness, cv2.LINE_AA)

    try:
        for frame, obs in tracker.frames():
            now = time.monotonic()

            if obs is None:
                # No hand visible. Stop commanding motion. The Pi will
                # also freeze on its staleness timeout.
                if now - last_send >= send_period:
                    client.send_target(
                        joints=session.home_joints,
                        deadman=False,
                        confidence=0.0,
                    )
                    last_send = now
            else:
                if retargeter.calibration is None:
                    # Auto-calibrate on first hand detection so the user
                    # doesn't have to press 'c' to get started.
                    retargeter.calibrate(obs, session.home_joints)
                    logging.info("auto-calibrated neutral pose")

                target = retargeter.map(obs)
                if target is not None:
                    # Clamp to the safety envelope the Pi advertised so
                    # we don't waste bandwidth shipping rejected targets.
                    target = np.clip(target, session.joint_min, session.joint_max)
                    last_target = target
                    if now - last_send >= send_period:
                        if args.auto_arm or args.no_preview:
                            effective_deadman = True
                        else:
                            effective_deadman = deadman_held
                        client.send_target(
                            joints=target,
                            deadman=effective_deadman,
                            confidence=obs.confidence,
                            ts_ns=obs.timestamp_ns,
                        )
                        last_send = now

            # Preview + key handling. Runs every frame regardless of
            # whether a hand was detected — macOS freezes any cv2 window
            # that isn't pumped on every tick.
            if not args.no_preview:
                preview = frame.copy()
                ack = client.latest_ack
                status = ack.status_message if ack is not None else "(no ack)"
                armed_now = (args.auto_arm and obs is not None) or deadman_held
                # Saturated, dark colors so they stay readable against the
                # white halo. Green for armed, dark orange-ish for safe.
                deadman_color = (0, 130, 0) if armed_now else (0, 80, 200)
                hand_state = "hand: tracking" if obs is not None else "hand: NONE"
                deadman_label = "ARMED" if armed_now else "safe"
                if args.auto_arm:
                    deadman_label += " (auto)"
                put_text(preview, f"deadman: {deadman_label}", (10, 30),
                         scale=0.8, thickness=2, fill=deadman_color)
                put_text(preview, hand_state, (10, 60), scale=0.6, thickness=1)
                put_text(preview, f"pi: {status}", (10, 85), scale=0.6, thickness=1)
                if last_target is not None:
                    tgt_str = " ".join(f"{n[:3]}={v:+5.0f}" for n, v in zip(session.joint_names, last_target))
                    put_text(preview, f"tgt: {tgt_str}", (10, 110),
                             scale=0.45, thickness=1, fill=(0, 100, 0))
                if client.latest_ack is not None:
                    cur = list(client.latest_ack.current_joints)
                    cur_str = " ".join(f"{n[:3]}={v:+5.0f}" for n, v in zip(session.joint_names, cur))
                    put_text(preview, f"cur: {cur_str}", (10, 130),
                             scale=0.45, thickness=1, fill=(120, 0, 0))
                if retargeter.calibration is not None and obs is not None:
                    sz = retargeter.last_hand_size
                    ds = retargeter.last_d_size
                    put_text(preview,
                             f"depth: size={sz:.3f}  d={ds:+.3f}  (calib={retargeter.calibration.hand_size:.3f})",
                             (10, 155), scale=0.45, thickness=1, fill=(0, 80, 130))
                if retargeter.last_target_delta is not None:
                    dx, dy, dz = (retargeter.last_target_delta * 1000).tolist()
                    put_text(preview,
                             f"xyz target delta (mm): fwd={dx:+5.0f}  lat={dy:+5.0f}  up={dz:+5.0f}",
                             (10, 175), scale=0.45, thickness=1, fill=(120, 0, 80))
                cv2.imshow("interlatent-teleop", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    deadman_held = not deadman_held
                    logging.info("deadman %s", "ARMED" if deadman_held else "released")
                elif key == ord('c') and obs is not None:
                    retargeter.calibrate(obs, session.home_joints)
                    logging.info("recalibrated")
                elif key in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.close()
        finally:
            tracker.close()
            if not args.no_preview:
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
