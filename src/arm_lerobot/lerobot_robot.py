"""LeRobot ``Robot`` adapter for the 27_engineer three-axis Damiao arm.

Architecture reminder (see docs/REQUIREMENTS.md)
------------------------------------------------
The STM32H723 owns real time: it runs the 1 kHz MIT servo loop, gravity
feed-forward, soft limits, the shoulder/elbow interlock and motor heartbeats.
The PC never commands the arm in phase 1.  It observes.

So this class is deliberately a *read-only* Robot for now:

    observation  <- `watch` telemetry, `qb/qa/qe` (measured joint angles),
                    time-aligned to each camera frame's capture instant
    camera       <- Orbbec Gemini 2, via the LeRobot Camera adapter
    action       <- supplied by ArmTelemetry (the `qbt/qat/qet` the H7 is
                    already executing on the operator's behalf)
    send_action  <- a no-op

That last point is what makes phase 1 possible with **zero firmware changes**,
and it is also why the recorded action is exactly the quantity phase 2 will
transmit back to the H7: the dataset's action space and the inference command
are the same physical quantity.

Verified interface facts (LeRobot 0.6.1, read from source not assumed)
---------------------------------------------------------------------
  * ``Robot`` is an ABC in ``robots/robot.py``; the abstract members implemented
    below are observation_features, action_features, is_connected,
    connect(calibrate=True), is_calibrated, calibrate, configure,
    get_observation, send_action, disconnect (robot.py:88-211).
  * ``Robot.__init__`` creates the calibration directory, so ``name`` must be set
    as a class attribute before that runs (robot.py:47-53).
  * ``observation_features`` keys must match ``get_observation()`` keys exactly.
    ``build_dataset_frame`` reads state values by feature *name*, so a mismatch
    surfaces as a KeyError deep inside the record loop (utils/feature_utils.py:132).
  * ``lerobot_record.py:447`` calls ``len(robot.cameras)`` without a guard, so a
    custom robot must expose a ``cameras`` dict even when it has none.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .arm_model import check_limits
from .telemetry import TelemetryReader, acquire_reader, release_reader

logger = logging.getLogger(__name__)

__all__ = ["ArmRobot", "ArmRobotConfig", "JOINT_NAMES"]

# The observation/action vector, in order.  These names appear verbatim as the
# `names` of `observation.state` and `action` in the dataset, and the Teleoperator
# must produce actions under exactly the same keys.
JOINT_NAMES: tuple[str, str, str] = ("q_base", "q_shoulder", "q_elbow")


@RobotConfig.register_subclass("arm_27")
@dataclass(kw_only=True)
class ArmRobotConfig(RobotConfig):
    """Configuration for the 27_engineer arm.

    ``cameras`` is declared here rather than inherited: ``RobotConfig`` carries
    only ``id`` and ``calibration_dir`` (robots/config.py:22-27) and merely
    *checks* for a ``cameras`` attribute when a subclass defines one
    (robots/config.py:29-36).  Every camera entry must have width/height/fps set,
    or that check raises during construction.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    port: str = "COM8"
    baud: int = 921600

    # Phase 1: never write to the arm.  Flipping this to False raises from
    # send_action(), because the PC->H7 channel does not exist yet (phase 2).
    read_only: bool = True

    # Refuse to emit an observation when the newest telemetry is older than this.
    # Deliberately a hard failure: recording a stale state alongside a fresh image
    # produces a dataset that looks fine and trains a bad policy, which is the
    # exact failure this whole alignment layer exists to prevent.  Losing one
    # episode to a USB hiccup is the cheaper outcome.
    max_state_age_s: float = 0.5

    # Screen every emitted state against the firmware's own soft limits and the
    # shoulder/elbow interlock.  Violations are logged, never corrected here -
    # the firmware already refuses to deepen them.
    validate_limits: bool = True


class ArmRobot(Robot):
    """Read-only LeRobot Robot backed by UART7 telemetry."""

    config_class = ArmRobotConfig
    name = "arm_27"

    def __init__(self, config: ArmRobotConfig):
        super().__init__(config)
        self.config = config
        self._reader: TelemetryReader | None = None
        self.cameras: dict[str, Any] = {}
        self.limit_violations = 0
        self.stale_observations = 0

    # ------------------------------------------------------------- feature spec
    @property
    def observation_features(self) -> dict[str, Any]:
        """Joint names -> float, camera name -> (H, W, 3).

        These are *hardware* names; LeRobot turns them into `observation.state`
        and `observation.images.<name>` via `hw_to_dataset_features`.  Declaring a
        raw `observation.state` key here would be wrong.

        Must work before `connect()`, so camera geometry comes from the config
        rather than from live camera objects (robot.py:98).
        """
        features: dict[str, Any] = {name: float for name in JOINT_NAMES}
        for cam_name, cam_cfg in (self.config.cameras or {}).items():
            height, width = cam_cfg.height, cam_cfg.width
            if height is None or width is None:
                raise ValueError(
                    f"camera '{cam_name}' needs both width and height set; "
                    f"got width={width!r} height={height!r}"
                )
            features[cam_name] = (int(height), int(width), 3)
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        return {name: float for name in JOINT_NAMES}

    # --------------------------------------------------------------- lifecycle
    @property
    def is_connected(self) -> bool:
        return self._reader is not None

    def connect(self, calibrate: bool = True) -> None:
        if self._reader is not None:
            return

        # Shared with the Teleoperator: two readers on one COM port would each
        # steal roughly half the telemetry lines.
        self._reader = acquire_reader(self.config.port, self.config.baud)

        if self.config.cameras:
            self.cameras = make_cameras_from_configs(self.config.cameras)
        for name, cam in self.cameras.items():
            logger.info("connecting camera %s", name)
            cam.connect()

        logger.info(
            "arm_27 connected on %s @ %d (read_only=%s)",
            self.config.port, self.config.baud, self.config.read_only,
        )

    @property
    def is_calibrated(self) -> bool:
        """Always True: the STM32 owns zeroing and joint calibration.

        The firmware's own motor->joint mapping and its `ARM_KIN_CALIBRATED` flag
        (arm.h:386-396) are authoritative.  Re-deriving offsets on the PC would
        duplicate, and could contradict, that.
        """
        return True

    def calibrate(self) -> None:
        """No-op by design.  See is_calibrated."""

    def configure(self) -> None:
        """No-op: nothing is written to the arm in phase 1."""

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:  # never let cleanup mask the real error
                logger.warning("camera disconnect failed: %s", exc)
        self.cameras = {}

        if self._reader is not None:
            release_reader(self.config.port, self.config.baud)
            self._reader = None

    # -------------------------------------------------------------- observation
    def get_observation(self) -> dict[str, Any]:
        """One aligned (images, joint state) sample.

        Order matters: images are read first, each camera reports the host time at
        which its frame was actually captured, and the arm state is then
        interpolated to that instant.  Reading the state first and the image
        afterwards would pair a fresh image with a stale state.
        """
        if self._reader is None:
            raise RuntimeError("ArmRobot is not connected; call connect() first")

        images: dict[str, Any] = {}
        for name, cam in self.cameras.items():
            images[name] = cam.async_read()

        # With one camera this is exact.  With several, the observation carries a
        # single `observation.state`, so we align to the earliest capture and
        # accept the spread between cameras; per-camera state would need separate
        # state features.
        host_time = time.monotonic()
        capture_times = [
            t for t in (getattr(c, "last_frame_host_time", None)
                        for c in self.cameras.values())
            if t is not None
        ]
        if capture_times:
            host_time = min(capture_times)

        state = self._reader.snapshot(host_time)
        age = self._reader.state_age_s(host_time)

        if state is None or (age is not None and age > self.config.max_state_age_s):
            self.stale_observations += 1
            raise TimeoutError(
                f"no usable arm telemetry at host_time={host_time:.4f}: "
                f"aligned state={'none' if state is None else 'ok'}, "
                f"age={('unknown' if age is None else f'{age:.3f}s')} "
                f"(limit {self.config.max_state_age_s}s). "
                f"Reader: {self._reader.stats()}"
            )

        if self.config.validate_limits:
            violations = check_limits(*state.q)
            if violations:
                self.limit_violations += 1
                logger.warning("joint limit violation in recorded state: %s", violations)

        observation: dict[str, Any] = {
            "q_base": float(state.q[0]),
            "q_shoulder": float(state.q[1]),
            "q_elbow": float(state.q[2]),
        }
        observation.update(images)
        return observation

    def arm_frame(self, max_age_s: float = 0.2):
        """The most recent aligned state, taking a fresh one if the cache is stale.

        Not part of the LeRobot interface; the record loop only sees the three
        joint angles.  Depth-based success checks, the motion trigger and
        diagnostics want the rest (velocities, torques, enable/online flags).

        The cached snapshot is only produced by ``get_observation()``.  Callers
        that need the state *before* the recording loop starts - the motion
        trigger, for instance - would otherwise get None even though telemetry is
        flowing perfectly well.  So fall back to sampling a fresh one.
        """
        if self._reader is None:
            return None
        cached = self._reader.last_snapshot(max_age_s)
        if cached is not None:
            return cached
        return self._reader.snapshot(time.monotonic())

    # ------------------------------------------------------------------ action
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Phase 1: intentionally a no-op that returns the action unchanged.

        The HT-10A transmitter, through the STM32, remains the only source of arm
        commands.  LeRobot's record loop discards this return value and writes the
        *Teleoperator's* action into the dataset, so the recorded action is the
        joint target the H7 was already executing - which is precisely what phase 2
        will transmit back over UART7.

        Writing to the arm here would either fight the remote or require the
        reverse channel that phase 2 adds.
        """
        if not self.config.read_only:
            raise NotImplementedError(
                "PC -> H7 target-angle frames are a phase 2 deliverable; the "
                "reverse UART7 protocol does not exist yet. Keep read_only=True."
            )
        return action
