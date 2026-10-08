"""Kinematic model of the 27_engineer three-axis Damiao arm.

Every constant is copied from the firmware so PC and STM32 agree exactly.
Sources, all read from D:\\MyTrain\\projects\\27_engineer:

    project/application/arm/arm.h:386-396     geometry + calibration constants
    project/application/arm/arm.h:47-89       joint limits
    project/modules/alg/arm_kin/arm_kin.h:28  shoulder reduction 1:2
    tools/arm_watch.py:118-150                the same FK, already verified on hardware

Joint convention
----------------
    th2 = S2 * q_shoulder / 2 + C2    upper arm angle vs horizontal, up is positive
    th3 = S3 * q_elbow        + C3    forearm angle relative to the upper arm
    yaw = q_base                      base rotation about vertical

The asymmetry matters: the shoulder motor turns twice per joint revolution
(1:2 reduction) so q_shoulder is halved, while the elbow is 1:1.

The base yaw does NOT take part in the firmware's forward kinematics --
arm_kin only models the th2/th3 planar two-link chain (arm_kin.cpp:9-48).
fk_links() therefore applies yaw as a pure rotation of the planar result,
which is exact for this geometry and is what arm_watch.py already does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------- geometry
# Measured with a tape measure, mm.  arm.h:386-388
ARM_H_S_MM = 240.0   # base yaw axis -> shoulder pitch axis, vertical
ARM_L_SE_MM = 300.0  # shoulder -> elbow
ARM_L_ET_MM = 338.0  # elbow -> end effector (fitted, tape measure said 390)

# Motor angle -> joint angle calibration.  arm.h:393-396
ARM_S2 = +1.0
ARM_C2 = +2.783074
ARM_S3 = +1.0
ARM_C3 = -2.771681

# Shoulder joint turns half as fast as the motor.  arm_kin.h:28
SHOULDER_REDUCTION = 2.0

ARM_KIN_CALIBRATED = True  # arm.h:390

# ----------------------------------------------------------------- joint limits
# Radians, in *motor/joint* angle space (the same numbers the firmware uses).
BASE_LIMIT = (-2.5, 2.5)                 # arm.h:47-48
SHOULDER_MECH_LIMIT = (-5.50, 0.612)     # arm.h:79-83
SHOULDER_SOFT_LIMIT = (-5.35, 0.462)     # arm.h:79-83, what the firmware enforces
ELBOW_LIMIT = (-2.5, 3.15)               # arm.h:88-89

# Interlock from arm_safety.cpp:19-37: when the shoulder is below this angle the
# elbow is confined, to stop the forearm folding into the shoulder.
SHOULDER_INTERLOCK_BELOW = -3.5
ELBOW_LOCKED_RANGE = (1.8, 3.1)

# safe_flags bit meanings.  Only bit 0 is confirmed, by tools/arm_watch.py:185-186.
SAFE_STALL = 1 << 0  # 顶住 - commanded vs measured error stayed large


def motor_to_joint(q_shoulder: float, q_elbow: float) -> tuple[float, float]:
    """Motor angles (rad) -> planar joint angles (th2, th3) in radians."""
    th2 = ARM_S2 * q_shoulder / SHOULDER_REDUCTION + ARM_C2
    th3 = ARM_S3 * q_elbow + ARM_C3
    return th2, th3


def joint_to_motor(th2: float, th3: float) -> tuple[float, float]:
    """Inverse of motor_to_joint.  Needed to turn a desired pose back into targets."""
    q_shoulder = (th2 - ARM_C2) * SHOULDER_REDUCTION / ARM_S2
    q_elbow = (th3 - ARM_C3) / ARM_S3
    return q_shoulder, q_elbow


def fk_r_z(th2: float, th3: float) -> tuple[float, float]:
    """Planar forward kinematics: radial distance and height of the tip, in mm.

    Matches ArmKin_Fk in the firmware and arm_watch.py:133-144.
    """
    r = ARM_L_SE_MM * math.cos(th2) + ARM_L_ET_MM * math.cos(th2 + th3)
    z = ARM_H_S_MM + ARM_L_SE_MM * math.sin(th2) + ARM_L_ET_MM * math.sin(th2 + th3)
    return r, z


def ik_r_z(r: float, z: float, elbow_up: bool = True) -> tuple[float, float] | None:
    """Planar inverse kinematics.  Returns (th2, th3) or None if out of reach.

    Two-link IK, used only for sanity checks and future Cartesian helpers.
    ``elbow_up`` selects the solution branch.
    """
    dz = z - ARM_H_S_MM
    d2 = r * r + dz * dz
    l1, l2 = ARM_L_SE_MM, ARM_L_ET_MM

    if d2 > (l1 + l2) ** 2 or d2 < (l1 - l2) ** 2:
        return None

    cos_th3 = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    th3 = math.acos(cos_th3)
    if not elbow_up:
        th3 = -th3

    th2 = math.atan2(dz, r) - math.atan2(
        l2 * math.sin(th3), l1 + l2 * math.cos(th3)
    )
    return th2, th3


@dataclass(frozen=True)
class LinkPositions:
    """End-effector chain in vehicle frame, mm.  x forward, y left, z up."""

    x: tuple[float, float, float, float]
    y: tuple[float, float, float, float]
    z: tuple[float, float, float, float]

    @property
    def tip(self) -> tuple[float, float, float]:
        return self.x[3], self.y[3], self.z[3]


def fk_links(q_base: float, q_shoulder: float, q_elbow: float) -> LinkPositions:
    """Full chain: base origin -> shoulder -> elbow -> tip, in vehicle frame."""
    th2, th3 = motor_to_joint(q_shoulder, q_elbow)
    radial = (
        0.0,
        0.0,
        ARM_L_SE_MM * math.cos(th2),
        ARM_L_SE_MM * math.cos(th2) + ARM_L_ET_MM * math.cos(th2 + th3),
    )
    z = (
        0.0,
        ARM_H_S_MM,
        ARM_H_S_MM + ARM_L_SE_MM * math.sin(th2),
        ARM_H_S_MM + ARM_L_SE_MM * math.sin(th2) + ARM_L_ET_MM * math.sin(th2 + th3),
    )
    c, s = math.cos(q_base), math.sin(q_base)
    return LinkPositions(
        x=tuple(round(r * c, 1) for r in radial),
        y=tuple(round(r * s, 1) for r in radial),
        z=tuple(round(v, 1) for v in z),
    )


def check_limits(
    q_base: float,
    q_shoulder: float,
    q_elbow: float,
    tolerance_rad: float = 0.02,
) -> list[str]:
    """Return human-readable limit violations.  Empty list means all clear.

    This mirrors what the firmware already enforces; it exists so the PC can
    refuse to command something the arm would reject anyway, and so recorded data
    can be screened afterwards.

    ``tolerance_rad`` matters because the firmware limits the *commanded target*,
    not the measurement.  A live run showed the base reaching -2.518 rad against a
    +/-2.5 rad limit - 18 mrad of tracking error and backlash, not a real
    violation.  Screening with a strict ``<=`` cried wolf on 29 of 360 frames.
    The default 0.02 rad (~1.1 deg) absorbs that without hiding a genuine
    excursion.
    """
    out: list[str] = []
    tol = tolerance_rad

    if not (BASE_LIMIT[0] - tol <= q_base <= BASE_LIMIT[1] + tol):
        out.append(f"base {q_base:+.3f} outside {BASE_LIMIT} (tol {tol})")

    if not (SHOULDER_SOFT_LIMIT[0] - tol <= q_shoulder <= SHOULDER_SOFT_LIMIT[1] + tol):
        out.append(
            f"shoulder {q_shoulder:+.3f} outside soft {SHOULDER_SOFT_LIMIT} (tol {tol})"
        )

    if not (ELBOW_LIMIT[0] - tol <= q_elbow <= ELBOW_LIMIT[1] + tol):
        out.append(f"elbow {q_elbow:+.3f} outside {ELBOW_LIMIT} (tol {tol})")

    # The interlock is a hard structural constraint, so it gets no tolerance.
    if q_shoulder < SHOULDER_INTERLOCK_BELOW and not (
        ELBOW_LOCKED_RANGE[0] <= q_elbow <= ELBOW_LOCKED_RANGE[1]
    ):
        out.append(
            f"shoulder {q_shoulder:+.3f} < {SHOULDER_INTERLOCK_BELOW} requires "
            f"elbow in {ELBOW_LOCKED_RANGE}, got {q_elbow:+.3f}"
        )

    return out
