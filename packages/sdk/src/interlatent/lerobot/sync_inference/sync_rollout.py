"""Interlatent-instrumented recording for lerobot robots.

Drop-in replacement for ``lerobot-record`` that captures neural network
activations during policy-driven recording and uploads them to the
Interlatent platform for interpretability analysis.

Usage — same arguments as lerobot-record, plus ``--interlatent.*``:

    interlatent-sync-rollout \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras='{"cam": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \\
        --robot.id=my_arm \\
        --dataset.repo_id=user/dataset \\
        --dataset.num_episodes=1 \\
        --dataset.single_task="Pick up the cube" \\
        --policy.path=user/my_policy \\
        --interlatent.api_key=ilat_xxx \\
        --interlatent.layer=auto

If ``--interlatent.api_key`` is omitted, falls back to the
``INTERLATENT_API_KEY`` environment variable.  If neither is set,
recording proceeds without Interlatent (identical to lerobot-record).
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

import numpy as np
import torch

# ── lerobot type registrations (needed for draccus CLI parsing) ──────
try:
    import lerobot  # noqa: F401
except ImportError as _e:
    raise SystemExit(
        "interlatent-sync-rollout drives real lerobot robots and needs the "
        "lerobot extra installed:\n\n    pip install 'interlatent[lerobot]'\n"
    ) from _e

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401

try:
    from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
except ImportError:
    pass

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so_follower,
)

# Import additional robot types — some may not be installed
for _mod in [
    "bi_so_follower", "koch_follower", "omx_follower",
    "openarm_follower", "bi_openarm_follower", "hope_jr",
    "earthrover_mini_plus", "reachy2",
]:
    try:
        __import__(f"lerobot.robots.{_mod}")
    except ImportError:
        pass

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)

for _mod in [
    "so_leader", "koch_leader", "omx_leader", "bi_so_leader",
    "openarm_leader", "openarm_mini", "bi_openarm_leader",
    "homunculus", "reachy2_teleoperator",
]:
    try:
        __import__(f"lerobot.teleoperators.{_mod}")
    except ImportError:
        pass

from lerobot.teleoperators import so_leader, koch_leader, omx_leader  # noqa: F401

# ── lerobot core imports ─────────────────────────────────────────────
# control_utils moved from lerobot.utils to lerobot.common in lerobot >=
# main (post 2026-04-12, after v0.5.1).  Support both layouts.
try:
    from lerobot.common.control_utils import (
        init_keyboard_listener,
        is_headless,
        predict_action,
        sanity_check_dataset_name,
        sanity_check_dataset_robot_compatibility,
    )
except ImportError:
    from lerobot.utils.control_utils import (
        init_keyboard_listener,
        is_headless,
        predict_action,
        sanity_check_dataset_name,
        sanity_check_dataset_robot_compatibility,
    )
from lerobot.configs import parser

# datasets — top-level re-exports were added post-v0.5.1; use deep paths
# that exist in both v0.5.1 and main.
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.image_writer import safe_stop_image_writer

# policies — top-level re-exports were added post-v0.5.1; use deep paths.
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action

# ActionInterpolator moved out of policies/rtc into utils on lerobot main.
try:
    from lerobot.policies.rtc.action_interpolator import ActionInterpolator
except ImportError:
    from lerobot.utils.action_interpolator import ActionInterpolator

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
# rename_stats is only re-exported from lerobot.processor on main.
from lerobot.processor.rename_processor import rename_stats

from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device

# feature_utils lives under datasets in v0.5.1 and was duplicated under
# utils on main.
try:
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
except ImportError:
    from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ── lerobot record config ───────────────────────────────────────────
from lerobot.scripts.lerobot_record import RecordConfig

logger = logging.getLogger(__name__)


# =====================================================================
# Helpers
# =====================================================================


def _derive_env_slug(policy_path: str | None, override: str | None = None) -> str:
    """Derive a backend environment slug from a policy path.

    - override provided → use it
    - HF repo "org/model-name" → "model-name"
    - local path "/path/to/model.pt" → "model"
    """
    if override:
        return override
    if policy_path is None:
        return str(int(time.time()))
    if "/" in policy_path and not os.path.exists(policy_path):
        return policy_path.split("/")[-1]
    basename = os.path.basename(policy_path)
    name = os.path.splitext(basename)[0]
    return name or basename or str(int(time.time()))


def _flatten_action_dict(action_values: Any) -> np.ndarray | None:
    """Flatten a robot action (dict or array-like) into a 1-D float32 array."""
    if action_values is None:
        return None
    try:
        if isinstance(action_values, dict):
            flat: list[float] = []
            for v in action_values.values():
                arr = np.asarray(v).flatten()
                flat.extend(float(x) for x in arr)
            return np.asarray(flat, dtype=np.float32) if flat else None
        return np.asarray(action_values).reshape(-1).astype(np.float32)
    except (ValueError, TypeError):
        return None


def _tick_interlatent(
    il_client,
    obs: dict,
    action_values: Any,
    camera_names: set[str],
) -> None:
    """Extract frames + proprio from a raw robot obs and tick the SDK."""
    frames: dict[str, Any] = {}
    proprio_values: list[float] = []

    for key, value in obs.items():
        if key in camera_names:
            frames[key] = value
        elif key == "task":
            continue
        else:
            try:
                arr = np.asarray(value).flatten()
                proprio_values.extend(float(v) for v in arr)
            except (ValueError, TypeError):
                pass

    obs_vec = np.asarray(proprio_values, dtype=np.float32) if proprio_values else None
    action_arr = _flatten_action_dict(action_values)

    il_client.tick(
        obs=obs_vec,
        action=action_arr,
        frame=frames if frames else None,
    )


# =====================================================================
# Config
# =====================================================================


@dataclass
class InterlatentConfig:
    """Interlatent capture settings (all optional)."""

    # API key. Falls back to INTERLATENT_API_KEY env var.
    api_key: str | None = None
    # Layer(s) to hook for activation capture.
    layer: str = "auto"
    # Override backend env slug (default: derived from policy path).
    env_slug: str | None = None
    # Override human-readable environment name (default: lerobot_<robot_type>).
    env_name: str | None = None
    # Upload captured data after recording finishes.
    upload: bool = True


@dataclass
class InterlatentRecordConfig(RecordConfig):
    """RecordConfig extended with Interlatent settings."""

    interlatent: InterlatentConfig = field(default_factory=InterlatentConfig)


# =====================================================================
# Modified record_loop (adds Interlatent tick before each inference)
# =====================================================================


@safe_stop_image_writer
def interlatent_record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    interpolator: ActionInterpolator | None = None,
    display_compressed_images: bool = False,
    # ── Interlatent additions ──
    il_client=None,
    camera_names: set | None = None,
):
    """record_loop with Interlatent activation capture.

    Identical to lerobot's ``record_loop`` except for a single
    ``il_client.tick()`` call inserted after each ``predict_action()`` —
    the SDK requires both ``obs`` and ``action`` per step, so the tick
    must happen once the action has been computed.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. "
                "Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # Reset interpolator if provided
    if interpolator is not None:
        interpolator.reset()

    # Calculate control interval based on interpolation
    use_interpolation = interpolator is not None and interpolator.enabled and policy is not None
    control_interval = interpolator.get_control_interval(fps) if interpolator else 1 / fps
    action_keys = sorted(robot.action_features) if use_interpolation else []

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        is_record_frame = True

        # Get action from either policy or teleop
        if policy is not None and preprocessor is not None and postprocessor is not None:
            if use_interpolation:
                ran_inference = False

                if interpolator.needs_new_action():
                    action_values = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=single_task,
                        robot_type=robot.robot_type,
                    )
                    act_processed_policy = make_robot_action(action_values, dataset.features)
                    robot_action_to_send = robot_action_processor((act_processed_policy, obs))

                    action_tensor = torch.tensor([robot_action_to_send[k] for k in action_keys])
                    interpolator.add(action_tensor)
                    ran_inference = True

                interp_action = interpolator.get()
                if interp_action is not None:
                    robot_action_to_send = {k: interp_action[i].item() for i, k in enumerate(action_keys)}
                    action_values = robot_action_to_send
                else:
                    continue

                is_record_frame = ran_inference
            else:
                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )
                act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))
                action_values = robot_action_to_send

        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No policy or teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # Send action to robot
        _sent_action = robot.send_action(robot_action_to_send)

        # ── Interlatent: tick once obs + action are both known ──
        # The hook context_supplier reads _step_ctx populated by the
        # previous tick(), so activations captured during predict_action()
        # above are tagged with the prior step's ctx — that's the existing
        # watcher contract. We just need both obs and action present here.
        if il_client is not None and policy is not None and is_record_frame:
            try:
                _tick_interlatent(il_client, obs, action_values, camera_names or set())
            except Exception:
                logger.warning("Interlatent tick failed", exc_info=True)
        # ─────────────────────────────────────────────────────────

        # Write to dataset
        if dataset is not None and is_record_frame:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). "
                f"Dataset frames might be dropped and robot control might be unstable."
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


# =====================================================================
# Main entry point
# =====================================================================


@parser.wrap()
def interlatent_record(cfg: InterlatentRecordConfig) -> LeRobotDataset:
    """Record a dataset with Interlatent activation capture."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    il_client = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # Load pretrained policy
        policy = (
            None
            if cfg.policy is None
            else make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.dataset.rename_map)
        )
        preprocessor = None
        postprocessor = None
        interpolator = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )
            if cfg.interpolation_multiplier > 1:
                interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
                logging.info(f"Action interpolation enabled: {cfg.interpolation_multiplier}x control rate")

        # ── Set up Interlatent ───────────────────────────────────────
        camera_names: set[str] = set()
        api_key = cfg.interlatent.api_key or os.environ.get("INTERLATENT_API_KEY")

        if api_key and policy is not None:
            from interlatent import Interlatent

            il_client = Interlatent(api_key=api_key)
            env_slug = _derive_env_slug(
                getattr(cfg.policy, "pretrained_path", None),
                cfg.interlatent.env_slug,
            )
            env_name = cfg.interlatent.env_name or f"lerobot_{robot.name}"

            il_client.watch(
                policy,
                env_name=env_name,
                environment=env_slug.lower().replace(" ", "-"),
                layer=cfg.interlatent.layer,
                capture_frames=True,
            )

            if hasattr(robot, "cameras") and robot.cameras:
                camera_names = set(robot.cameras.keys())
                il_client.register_cameras(sorted(camera_names))

            logger.info(
                "Interlatent: capturing activations | layer='%s' environment='%s' env_name='%s' cameras=%s",
                cfg.interlatent.layer,
                env_slug,
                env_name,
                sorted(camera_names) if camera_names else "none",
            )
        elif api_key and policy is None:
            logger.warning(
                "INTERLATENT_API_KEY is set but no policy was loaded. "
                "Activation capture requires a policy — proceeding without Interlatent."
            )
        # ─────────────────────────────────────────────────────────────

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. Consider enabling it for faster episode saving: "
                "--dataset.streaming_encoding=true --dataset.encoder_threads=2"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                interlatent_record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    interpolator=interpolator,
                    display_compressed_images=display_compressed_images,
                    il_client=il_client,
                    camera_names=camera_names,
                )

                # Signal episode boundary to Interlatent. tick() now
                # requires obs+action, so we roll over the watcher's
                # episode UUID directly instead of synthesizing a fake
                # terminal step.
                if il_client is not None and il_client._watcher is not None:
                    try:
                        il_client._watcher.reset_episode()
                    except Exception:
                        logger.warning("Interlatent reset_episode failed", exc_info=True)

                # Reset time between episodes
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)

                    # Use original record_loop for reset (no Interlatent, no policy)
                    interlatent_record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        # ── Interlatent: upload and close ────────────────────────────
        if il_client is not None and cfg.interlatent.upload:
            try:
                logger.info("Uploading activation data to Interlatent...")
                il_client.upload()
                logger.info("Interlatent upload complete.")
            except Exception as exc:
                logger.warning("Interlatent upload failed: %s", exc)
            finally:
                il_client.close()
        elif il_client is not None:
            il_client.close()
        # ─────────────────────────────────────────────────────────────

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    interlatent_record()


if __name__ == "__main__":
    main()
