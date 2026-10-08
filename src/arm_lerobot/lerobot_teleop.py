"""LeRobot ``Teleoperator`` that serves actions straight out of arm telemetry.

Why this exists
---------------
LeRobot's model of teleoperation is a *leader* device that the PC reads:
``teleop.get_action()`` returns the command the operator wants, and the record
loop then writes it into the dataset as the action.

In this project the leader is not attached to the PC at all.  The HT-10A
transmitter is wired to the STM32, the STM32 integrates the sticks into joint
targets, and the ``watch`` telemetry line reports both the measured angle
(``qb/qa/qe``) and the commanded target (``qbt/qat/qet``).

So the Teleoperator's whole job is to expose those already-computed targets as the
action.  That has three useful consequences:

  1. Phase 1 needs **no firmware changes** - the PC is a pure observer.
  2. The operator keeps using the HT-10A with its existing feel; nothing about the
     teleoperation experience changes.
  3. The recorded action is exactly the quantity phase 2 will transmit back to the
     H7, so the training action space and the inference command are the same
     physical thing rather than merely similar.

Action/observation synchronisation
----------------------------------
LeRobot's record loop is::

    obs = robot.get_observation()   # aligned to a camera frame at instant T
    act = teleop.get_action()       # called separately, would sample T' > T
    robot.send_action(act)
    dataset.add_frame({**observation_frame, **action_frame, "task": ...})

There is no mechanism to pass T from the Robot to the Teleoperator.  Both share
one ``TelemetryReader`` (see the shared-reader registry in ``telemetry.py``), so
this class reuses the snapshot the Robot just produced.  Without that, the recorded
action would be sampled a few milliseconds after the image and state it is paired
with.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from .lerobot_robot import JOINT_NAMES
from .telemetry import acquire_reader, release_reader

logger = logging.getLogger(__name__)

__all__ = ["ArmTelemetry", "ArmTelemetryConfig"]


@TeleoperatorConfig.register_subclass("arm_telemetry")
@dataclass
class ArmTelemetryConfig(TeleoperatorConfig):
    """Points at the same serial port the Robot uses; the reader is shared."""

    port: str = "COM8"
    baud: int = 921600

    # How long a Robot-produced snapshot stays reusable.  Comfortably longer than
    # one record-loop iteration, short enough that a stalled loop does not pair an
    # ancient action with a fresh frame.
    max_snapshot_age_s: float = 0.2


class ArmTelemetry(Teleoperator):
    """Exposes the STM32's joint targets as the teleoperation action."""

    config_class = ArmTelemetryConfig
    name = "arm_telemetry"

    def __init__(self, config: ArmTelemetryConfig):
        super().__init__(config)
        self.config = config
        self._reader = None
        self.reused_snapshots = 0
        self.fresh_snapshots = 0

    # ------------------------------------------------------------- feature spec
    @property
    def action_features(self) -> dict[str, Any]:
        """Must match ``ArmRobot.action_features`` key for key.

        ``build_dataset_frame`` looks action values up by these names
        (utils/feature_utils.py:132), so any mismatch is a KeyError at record time.
        """
        return {name: float for name in JOINT_NAMES}

    @property
    def feedback_features(self) -> dict[str, Any]:
        """Empty: the HT-10A has no force-feedback path to the operator."""
        return {}

    # --------------------------------------------------------------- lifecycle
    @property
    def is_connected(self) -> bool:
        return self._reader is not None

    def connect(self, calibrate: bool = True) -> None:
        if self._reader is None:
            self._reader = acquire_reader(self.config.port, self.config.baud)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """No-op: the STM32 owns joint calibration."""

    def configure(self) -> None:
        """No-op: nothing is sent to the arm in phase 1."""

    def disconnect(self) -> None:
        if self._reader is not None:
            release_reader(self.config.port, self.config.baud)
            self._reader = None

    # ------------------------------------------------------------------ action
    def get_action(self) -> dict[str, Any]:
        """The joint targets the STM32 is currently executing."""
        if self._reader is None:
            raise RuntimeError("ArmTelemetry is not connected; call connect() first")

        state = self._reader.last_snapshot(self.config.max_snapshot_age_s)
        if state is not None:
            self.reused_snapshots += 1
        else:
            # Nobody produced a snapshot recently - the Teleoperator is being used
            # without a Robot in the same process (e.g. lerobot-teleoperate).  Align
            # to now instead.
            self.fresh_snapshots += 1
            state = self._reader.snapshot(time.monotonic())

        if state is None:
            raise TimeoutError(
                "no usable arm telemetry; cannot produce an action. "
                f"Reader: {self._reader.stats()}"
            )
        if not state.enabled:
            logger.warning("arm is disabled but an action was requested")
        if not state.remote_online:
            logger.warning("HT-10A link is down; the action is whatever the STM32 holds")

        return {
            "q_base": float(state.q_target[0]),
            "q_shoulder": float(state.q_target[1]),
            "q_elbow": float(state.q_target[2]),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Unsupported, and deliberately not an error.

        There is no path from the PC to the HT-10A transmitter, so the operator
        cannot be given force or vibration feedback.  Raising here would break any
        generic LeRobot driver that calls it unconditionally.
        """
        logger.debug("send_feedback ignored (no feedback path to the HT-10A)")

    # ------------------------------------------------------------ diagnostics
    def stats(self) -> dict[str, int]:
        return {
            "reused_snapshots": self.reused_snapshots,
            "fresh_snapshots": self.fresh_snapshots,
        }
