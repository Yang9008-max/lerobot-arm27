"""LeRobot ``Camera`` adapter for the Orbbec Gemini 2.

This is the thin layer that makes the device in :mod:`arm_lerobot.camera` look
like every other LeRobot camera, so ``--robot.cameras`` and the record loop can
treat it uniformly.

Interface facts that were verified against the installed LeRobot 0.6.1 source,
not assumed:

  * ``Camera`` is an ABC in ``lerobot/cameras/camera.py``.  The abstract members
    are ``is_connected``, ``find_cameras`` (staticmethod), ``connect(warmup=True)``,
    ``read()``, ``async_read(timeout_ms=200)`` and ``disconnect()``.
  * ``async_read`` returns a **bare numpy array**, not an ``(image, timestamp)``
    tuple (``camera.py:122``).  Three of LeRobot's own camera docstrings claim
    otherwise; the code is what counts.
  * ``Camera`` has no ``config_class`` or ``name`` attribute, unlike ``Robot``.
  * ``read_latest`` has a default implementation that only warns, so overriding
    it is optional but avoids a FutureWarning spam.

Depth convention
----------------
``async_read_depth`` returns **float32 metres**, because LeRobot infers the depth
unit from the dtype (float -> metres, integer -> millimetres) when it builds the
dataset feature.  The underlying device wrapper in ``camera.py`` works in
millimetres; the conversion happens here and nowhere else.

Note that depth is deliberately *not* declared in the robot's
``observation_features``.  ``hw_to_dataset_features`` turns any ``(H, W, 1)``
tuple into ``observation.images.<name>`` with ``is_depth_map=True``, and
``dataset_to_policy_features`` classifies every image/video feature as
``FeatureType.VISUAL`` - so a declared depth stream would be fed to ACT as an
extra camera.  Depth is available here for live post-processing (success checks,
object localisation) without entering the dataset.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig

# Aliased: the device wrapper's config would otherwise shadow LeRobot's CameraConfig
# in this module, and the two mean different things.
from .camera import CameraConfig as CaptureConfig, OrbbecCamera

__all__ = ["GeminiCamera", "GeminiCameraConfig"]


@CameraConfig.register_subclass("gemini")
@dataclass
class GeminiCameraConfig(CameraConfig):
    """Configuration for the Orbbec Gemini 2.

    ``width``/``height``/``fps`` are inherited from ``CameraConfig`` and must not
    be None - ``RobotConfig.__post_init__`` rejects a robot whose cameras leave
    them unset.
    """

    # Device selection.  None means "the only attached Orbbec device".
    serial: str | None = None

    # Depth request.  The device has no native 640x480 depth mode; 640x400 is the
    # closest, and the aligned result takes the colour resolution anyway.
    depth_width: int = 640
    depth_height: int = 400
    depth_fps: int = 30

    align_to_color: bool = True
    enable_frame_sync: bool = True
    warmup_frames: int = 5
    read_timeout_ms: int = 1000

    def __post_init__(self) -> None:
        # Fill in the conventional defaults so a bare GeminiCameraConfig() works.
        if self.width is None:
            self.width = 640
        if self.height is None:
            self.height = 480
        if self.fps is None:
            self.fps = 30


class GeminiCamera(Camera):
    """Background-threaded color (+depth) capture exposing the LeRobot Camera API.

    A single pump thread owns the device.  ``read``/``async_read`` only ever
    consume from its most recent frame, which keeps the SDK calls on one thread -
    pyorbbecsdk pipelines are not safe to drive from several threads, and the
    LeRobot record loop plus any visualisation would otherwise both call in.

    ``last_frame_host_time`` exposes the host ``time.monotonic()`` at which the
    frame currently held was captured.  The Camera interface does not carry a
    timestamp, but the robot needs one to align the image against arm telemetry;
    reading it off the instance right after ``async_read`` is more accurate than
    calling ``time.monotonic()`` afterwards, because ``async_read`` may hand back
    a frame captured a few milliseconds earlier.
    """

    def __init__(self, config: GeminiCameraConfig):
        super().__init__(config)
        self.config = config

        self._device: OrbbecCamera | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

        self._lock = threading.Lock()
        self._frame = None
        self._frame_host_time: float | None = None
        self._new_frame = threading.Event()
        self._timeouts = 0
        self._failures = 0

    # ------------------------------------------------------------ properties
    @property
    def is_connected(self) -> bool:
        return self._device is not None

    @property
    def last_frame_host_time(self) -> float | None:
        """Host monotonic time of the frame most recently handed out."""
        with self._lock:
            return self._frame_host_time

    @property
    def intrinsics(self) -> tuple[float, float, float, float] | None:
        """(fx, fy, cx, cy) of the colour stream, or None if unavailable.

        Needed to turn a pixel plus a depth into a metric position.  Taken from the
        profile the *device* negotiated (this adapter does not keep its own copy)
        and cached after the first successful read.
        """
        cached = getattr(self, "_intrinsics", None)
        if cached is not None:
            return cached
        profile = getattr(self._device, "_color_profile", None) if self._device else None
        if profile is None:
            return None
        from .block import camera_intrinsics

        self._intrinsics = camera_intrinsics(profile)
        return self._intrinsics

    def last_capture(self):
        """The most recent complete capture, without waiting for a new one.

        Unlike ``async_read`` this returns the whole ``arm_lerobot.camera.Frame``,
        including depth in millimetres, and never blocks.  Used by post-processing
        (success checks) that needs geometry rather than just the image.  Returns
        None when no frame has been captured yet.
        """
        with self._lock:
            return self._frame

    @property
    def timeouts(self) -> int:
        return self._timeouts

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def error(self) -> BaseException | None:
        """Set if the pump thread died; the record loop should surface this."""
        return self._error

    # ------------------------------------------------------------- discovery
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """List attached Orbbec devices, as LeRobot's camera discovery expects."""
        try:
            return OrbbecCamera.list_devices()
        except Exception as exc:  # pragma: no cover - depends on hardware
            return [{"error": str(exc)}]

    # -------------------------------------------------------------- lifecycle
    def connect(self, warmup: bool = True) -> None:
        if self._device is not None:
            return

        capture = CaptureConfig(
            color_width=int(self.config.width or 640),
            color_height=int(self.config.height or 480),
            color_fps=int(self.config.fps or 30),
            depth_width=self.config.depth_width,
            depth_height=self.config.depth_height,
            depth_fps=self.config.depth_fps,
            align_to_color=self.config.align_to_color,
            enable_frame_sync=self.config.enable_frame_sync,
            # mm as float32; converted to metres only in async_read_depth.
            depth_as_uint16=False,
            warmup_frames=self.config.warmup_frames if warmup else 0,
        )

        device = OrbbecCamera(capture)
        device.open()
        self._device = device

        self._stop.clear()
        self._new_frame.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._pump, name=f"gemini-{self.config.serial or '0'}", daemon=True
        )
        self._thread.start()

    def _pump(self) -> None:
        assert self._device is not None
        while not self._stop.is_set():
            try:
                frame = self._device.read(self.config.read_timeout_ms)
            except Exception as exc:  # SDK failure or device unplugged
                self._error = exc
                self._failures += 1
                break
            if frame is None or frame.rgb is None:
                self._timeouts += 1
                continue
            with self._lock:
                self._frame = frame
                self._frame_host_time = frame.host_time
                self._new_frame.set()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._device is not None:
            self._device.close()
            self._device = None
        self._new_frame.clear()

    # ------------------------------------------------------------------ read
    def _wait_for_frame(self, timeout_ms: float):
        """Block until an unconsumed frame is available.  Raises TimeoutError."""
        end = time.monotonic() + max(timeout_ms, 0.0) / 1000.0
        while True:
            if self._error is not None:
                raise RuntimeError(f"Gemini 2 pump thread failed: {self._error}")
            remaining = end - time.monotonic()
            if not self._new_frame.wait(timeout=max(remaining, 0.0)):
                raise TimeoutError(
                    f"{self.__class__.__name__}: no new frame within {timeout_ms} ms "
                    f"(timeouts so far: {self._timeouts})"
                )
            with self._lock:
                self._new_frame.clear()
                frame = self._frame
            if frame is not None:
                return frame

    def read(self) -> np.ndarray:
        """Blocking read of the next frame.

        Implemented on top of the pump thread rather than as a direct SDK call, so
        that the device is only ever touched from one thread.
        """
        return self._wait_for_frame(2000).rgb

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self._wait_for_frame(timeout_ms).rgb

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        """Non-blocking peek at the newest frame; raises if it is too old."""
        with self._lock:
            frame = self._frame
            host_time = self._frame_host_time
        if frame is None or host_time is None:
            raise RuntimeError("no frame captured yet")
        age_ms = (time.monotonic() - host_time) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(f"latest frame is {age_ms:.0f} ms old (> {max_age_ms} ms)")
        return frame.rgb

    # ----------------------------------------------------------------- depth
    def async_read_depth(self, timeout_ms: float = 200) -> np.ndarray:
        """Return aligned depth in **metres** (float32).

        LeRobot infers depth units from dtype: float -> metres, integer ->
        millimetres.  Returning millimetres as float32 would silently claim a
        depth range 1000x too large.
        """
        frame = self._wait_for_frame(timeout_ms)
        if frame.depth is None:
            raise RuntimeError("depth was not requested or not available")
        return (frame.depth / 1000.0).astype(np.float32)

    def read_latest_depth(self, max_age_ms: int = 500) -> np.ndarray:
        """Non-blocking aligned depth in metres; raises if the frame is too old."""
        with self._lock:
            frame = self._frame
            host_time = self._frame_host_time
        if frame is None or frame.depth is None or host_time is None:
            raise RuntimeError("no depth frame captured yet")
        age_ms = (time.monotonic() - host_time) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(f"latest depth frame is {age_ms:.0f} ms old")
        return (frame.depth / 1000.0).astype(np.float32)

    # ------------------------------------------------------------ diagnostics
    def stats(self) -> dict[str, Any]:
        return {
            "timeouts": self._timeouts,
            "failures": self._failures,
            "error": None if self._error is None else repr(self._error),
            "frames": None if self._device is None else self._device.stats.frames,
        }
