"""PC-side support code for the 27_engineer three-axis Damiao arm + Orbbec Gemini 2.

Layers
------
arm_model       kinematic model and joint limits, mirroring the STM32 firmware
telemetry       UART7 @921600 ASCII telemetry parsing and a background reader
sync            host-clock alignment between the 20 Hz telemetry and 30 Hz camera
camera          Orbbec Gemini 2 device wrapper (color + aligned depth)

LeRobot adapters (imported lazily, because they require LeRobot to be installed)
--------------------------------------------------------------------------------
lerobot_camera  GeminiCameraConfig / GeminiCamera   -> --camera.type=gemini
lerobot_robot   ArmRobotConfig / ArmRobot           -> --robot.type=arm_27
lerobot_teleop  ArmTelemetryConfig / ArmTelemetry   -> --teleop.type=arm_telemetry

The adapters are exposed through a module-level ``__getattr__`` rather than plain
imports so that ``import arm_lerobot.telemetry`` keeps working in an interpreter
that does not have LeRobot installed - the offline probe tools rely on that.  The
names are still resolvable via ``getattr``, which is what LeRobot's
``make_device_from_device_class`` fallback uses.

Design rule for this package: the PC never owns real-time control.  The STM32H723
keeps the 1 kHz MIT servo loop, gravity feed-forward, soft limits and interlocks.
The PC reads state and, from phase 2 onward, writes target joint angles.  See
docs/REQUIREMENTS.md for the full architecture decision record.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

# name -> module that defines it.  Kept explicit so a typo fails loudly.
_LAZY_ADAPTERS: dict[str, str] = {
    "GeminiCamera": "arm_lerobot.lerobot_camera",
    "GeminiCameraConfig": "arm_lerobot.lerobot_camera",
    "ArmRobot": "arm_lerobot.lerobot_robot",
    "ArmRobotConfig": "arm_lerobot.lerobot_robot",
    "JOINT_NAMES": "arm_lerobot.lerobot_robot",
    "ArmTelemetry": "arm_lerobot.lerobot_teleop",
    "ArmTelemetryConfig": "arm_lerobot.lerobot_teleop",
}

__all__ = [
    "__version__",
    "arm_model",
    "camera",
    "sync",
    "telemetry",
    *_LAZY_ADAPTERS,
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy import of the LeRobot adapters.

    LeRobot resolves custom device classes by looking up the class name on the
    parent package, so ``getattr(arm_lerobot, "ArmRobot")`` must work - but only
    when someone actually asks for it, so that the hardware-only tools stay usable
    in an interpreter without LeRobot.
    """
    module_name = _LAZY_ADAPTERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups are direct
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
