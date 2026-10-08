"""Orbbec Gemini 2 capture for LeRobot.

Wraps pyorbbecsdk with the small surface this project actually needs: open the
color + depth streams, align depth into the color view, hand back a clean
RGB uint8 array plus a float32 depth array in millimetres.

API usage follows the official examples shipped inside the wheel
(pyorbbecsdk/examples/beginner/03_color_and_depth_aligned.py, examples/utils.py),
so it is verified rather than guessed.  Device facts measured on this machine
(Gemini 2, serial AY6V163008W, USB3.0):

    color  : 640x480@30 RGB available (also 1280x720@30, 1920x1080@30)
    depth  : 1280x800 / 640x400 / 320x200, formats Y16, Y14, RLE
             NOTE there is no native 640x480 depth mode, so alignment is
             mandatory and the aligned depth always takes the color resolution.
    rate   : 29.94 fps measured, inter-frame jitter p95-p50 = 2.4 ms
    stamps : hardware timestamps are NOT available (get_timestamp_us returns a
             constant 2**33 sentinel).  Windows requires an administrator to run
             pyorbbecsdk/shared/obsensor_metadata_win10.ps1 -op install_all, and
             to re-run it for every newly attached device.  Alignment therefore
             uses host arrival time; see docs/REQUIREMENTS.md risk R2.

Design notes
------------
* Colour is returned as **RGB**, not BGR.  LeRobot's dataset and policy code
  treats camera output as RGB.  Conversion to BGR happens only when saving a
  PNG for a human to look at.
* Frame decoding uses ``reshape`` with an explicit length check.  The official
  ``utils.frame_to_bgr_image`` uses ``np.resize``, which silently *repeats* data
  when the size does not match instead of failing.  A wrong resolution would then
  produce a plausible-looking but wrong image.  We would rather raise.
* Depth format matters: RLE is compressed and must not be reinterpreted as raw
  uint16.  Profile selection prefers uncompressed Y16, then Y14, then RLE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# pyorbbecsdk is imported lazily inside open() so that importing this module -
# and therefore the whole package - does not require a camera to be present.

# Timestamp accessors, most-preferred first.  Orbbec's SDK has renamed these
# across releases and on Windows they only return real values once the UVC
# metadata device has been registered (obsensor_metadata_win10.ps1).
_TIMESTAMP_METHODS = (
    "get_timestamp_us",
    "get_global_timestamp_us",
    "get_system_timestamp_us",
    "get_timestamp",
)


def _safe(obj: Any, name: str, default: Any = None) -> Any:
    """Call ``obj.name()`` if it exists, else return ``default``.  Never raises."""
    fn = getattr(obj, name, None)
    if not callable(fn):
        return default
    try:
        return fn()
    except Exception:
        return default


def frame_timestamp_us(frame: Any) -> tuple[str | None, int | None]:
    """Return (method_name, microseconds) for the first working accessor."""
    for name in _TIMESTAMP_METHODS:
        value = _safe(frame, name)
        if value is not None:
            return name, int(value)
    return None, None


@dataclass
class CameraConfig:
    """What to ask the device for.

    A width/height/fps of 0 means "no constraint" to the profile picker.
    The depth defaults deliberately ask for 640x400 rather than the device
    default 1280x800: the aligned result is resampled into the 640x480 color
    frame anyway, so the extra 2.56x pixels cost USB bandwidth and CPU for no
    benefit in the training data.
    """

    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30
    depth_width: int = 640
    depth_height: int = 400
    depth_fps: int = 30
    align_to_color: bool = True
    enable_frame_sync: bool = True
    warmup_frames: int = 5
    # Keep raw uint16 depth instead of float32 mm.  Saves memory and matches
    # what the device actually produces; useful when recording every frame.
    depth_as_uint16: bool = False

    def describe(self) -> str:
        def part(w: int, h: int, f: int) -> str:
            dims = f"{w or 'any'}x{h or 'any'}"
            return f"{dims}@{f or 'any'}"

        return (
            f"color={part(self.color_width, self.color_height, self.color_fps)} "
            f"depth={part(self.depth_width, self.depth_height, self.depth_fps)} "
            f"align={'color' if self.align_to_color else 'off'} "
            f"sync={'on' if self.enable_frame_sync else 'off'}"
        )


@dataclass
class Frame:
    """One synchronized, aligned sample."""

    rgb: np.ndarray | None            # (H, W, 3) uint8, RGB
    depth: np.ndarray | None          # (H, W) float32 mm, or uint16 raw if configured
    host_time: float                  # time.monotonic() when wait_for_frames returned
    wall_time: float                  # time.time(), for matching against logs/datasets
    color_timestamp_us: int | None
    depth_timestamp_us: int | None
    timestamp_source: str | None      # which accessor actually worked
    color_format: str | None
    depth_format: str | None
    depth_scale: float | None         # device units -> mm
    frame_number: int | None

    @property
    def has_rgb(self) -> bool:
        return self.rgb is not None

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    def depth_at(self, x: int, y: int) -> float | None:
        """Depth in mm at a pixel, or None if invalid/out of range."""
        if self.depth is None:
            return None
        h, w = self.depth.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        value = float(self.depth[y, x])
        return value if value > 0 else None

    def timestamps_look_real(self) -> bool:
        """False when the device hands back its unusable sentinel value.

        Observed on an unregistered Windows install (see the module docstring):

            run 1: colour = depth = 8589934592  = 2 * 2**32
            run 2: colour = depth = 12884901888 = 3 * 2**32

        Two things give it away, and BOTH must be checked because the exact value
        is not stable - an earlier version of this method compared against a
        hardcoded 2**33 and would happily have called the second value "real":

          1. colour and depth report the *same* value, although the two sensors
             are physically captured microseconds apart;
          2. the value is always a multiple of 2**32, i.e. the low 32 bits are
             always zero - a shifted or truncated counter, not a clock.

        The authoritative check is :meth:`CameraStats.timestamp_fps`, which has
        frame history and can see that the value never advances.  This per-frame
        method only catches the obvious cases.
        """
        ts = self.color_timestamp_us
        if ts is None:
            return False
        if self.depth_timestamp_us is not None and ts == self.depth_timestamp_us:
            return False
        return ts % (2 ** 32) != 0


@dataclass
class CameraStats:
    frames: int = 0
    dropped: int = 0
    timeouts: int = 0
    first_host_time: float | None = None
    last_host_time: float | None = None
    color_timestamps: list[int] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        if self.first_host_time is None or self.last_host_time is None:
            return 0.0
        return self.last_host_time - self.first_host_time

    @property
    def fps(self) -> float:
        # frames arrive spaced by 1/fps, so use frames-1 intervals
        if self.frames < 2 or self.elapsed_s <= 0:
            return 0.0
        return (self.frames - 1) / self.elapsed_s

    def timestamp_fps(self) -> float | None:
        """FPS computed from the device's own timestamps, if it provides usable ones.

        If this disagrees badly with ``fps``, the device clock and the host clock
        are not tracking each other - which is exactly the failure that would make
        camera/telemetry alignment wrong later.  Returns None when the device
        hands back a constant sentinel.
        """
        ts = self.color_timestamps
        if len(ts) < 2 or len(set(ts)) < 2:
            return None
        span_us = ts[-1] - ts[0]
        if span_us <= 0:
            return None
        return (len(ts) - 1) / (span_us / 1e6)


def convert_color_to_rgb(frame: Any) -> np.ndarray | None:
    """Decode a pyorbbecsdk video frame into an RGB uint8 array.

    Supports the formats the Gemini 2 actually emits.  Raises ValueError on a
    size mismatch instead of silently returning a scrambled image.
    """
    import cv2
    from pyorbbecsdk import OBFormat, FormatConvertFilter, OBConvertFormat

    width = _safe(frame, "get_width")
    height = _safe(frame, "get_height")
    fmt = _safe(frame, "get_format")
    if width is None or height is None or fmt is None:
        return None

    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    def expect(n: int, what: str) -> None:
        if data.size != n:
            raise ValueError(
                f"{what}: expected {n} bytes for {width}x{height}, got {data.size}"
            )

    if fmt == OBFormat.RGB:
        expect(width * height * 3, "RGB")
        return data.reshape((height, width, 3)).copy()

    if fmt == OBFormat.BGR:
        expect(width * height * 3, "BGR")
        return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.YUYV:
        expect(width * height * 2, "YUYV")
        return cv2.cvtColor(data.reshape((height, width, 2)), cv2.COLOR_YUV2RGB_YUY2)

    if fmt == OBFormat.UYVY:
        expect(width * height * 2, "UYVY")
        return cv2.cvtColor(data.reshape((height, width, 2)), cv2.COLOR_YUV2RGB_UYVY)

    if fmt == OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.NV12:
        expect(width * height * 3 // 2, "NV12")
        return cv2.cvtColor(data.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2RGB_NV12)

    if fmt == OBFormat.NV21:
        expect(width * height * 3 // 2, "NV21")
        return cv2.cvtColor(data.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2RGB_NV21)

    if fmt == OBFormat.I420:
        expect(width * height * 3 // 2, "I420")
        return cv2.cvtColor(data.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2RGB_I420)

    # Unknown format: ask the SDK to convert for us.
    mapping = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
        OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
        OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888,
    }
    target = mapping.get(fmt)
    if target is None:
        return None
    conv = FormatConvertFilter()
    conv.set_format_convert_format(target)
    converted = conv.process(frame)
    if converted is None:
        return None
    return convert_color_to_rgb(converted)


class OrbbecCamera:
    """Blocking color + depth capture from an Orbbec device.

    Not thread-safe by design: one reader, one loop.  LeRobot's camera protocol
    wants an ``async_read``; that wrapper lives in the LeRobot adapter, not here,
    so this class stays testable without LeRobot installed.

    Usage::

        with OrbbecCamera(CameraConfig()) as cam:
            frame = cam.read()
            if frame and frame.has_rgb:
                print(frame.rgb.shape, frame.depth.shape)
    """

    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self.stats = CameraStats()
        self._pipeline: Any = None
        self._align: Any = None
        self._color_profile: Any = None
        self._depth_profile: Any = None
        self._depth_scale: float | None = None

    # ------------------------------------------------------------- discovery
    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """Enumerate connected Orbbec devices with the info we care about.

        The Context must stay referenced for as long as the DeviceList is used:
        the SDK's device manager lives inside it.  Writing
        ``devices = Context().query_devices()`` lets the temporary Context be
        garbage collected immediately, and the next get_device_by_index() then
        raises ``OBError: NULL pointer passed for argument "deviceMgr"``.
        """
        from pyorbbecsdk import Context

        out: list[dict[str, Any]] = []
        context = Context()  # noqa: F841  keep alive for the whole loop
        devices = context.query_devices()
        for i in range(devices.get_count()):
            device = devices.get_device_by_index(i)
            info = device.get_device_info()
            out.append({
                "index": i,
                "name": _safe(info, "get_name", "?"),
                "vid": _safe(info, "get_vid"),
                "pid": _safe(info, "get_pid"),
                "serial": _safe(info, "get_serial_number", ""),
                "connection": _safe(info, "get_connection_type", ""),
                "uid": _safe(info, "get_uid", ""),
            })
        return out

    # ------------------------------------------------------------------ open
    def open(self) -> None:
        from pyorbbecsdk import (
            AlignFilter,
            Config,
            OBFormat,
            OBFrameAggregateOutputMode,
            OBSensorType,
            OBStreamType,
            Pipeline,
        )

        self._pipeline = Pipeline()
        config = Config()

        try:
            profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            self._color_profile = self._pick_profile(
                profiles, OBFormat.RGB, self.config.color_width,
                self.config.color_height, self.config.color_fps,
                (OBFormat.RGB, OBFormat.MJPG, OBFormat.YUYV),
            )
            if self._color_profile is None:
                raise RuntimeError("no usable RGB color profile on this device")
            config.enable_stream(self._color_profile)
        except Exception as exc:
            raise RuntimeError(f"color stream setup failed: {exc}") from exc

        try:
            profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            # Depth is raw 16-bit in Y16/Y14; RLE is compressed and must not be
            # reinterpreted as uint16, so it is the last resort.
            self._depth_profile = self._pick_profile(
                profiles, None, self.config.depth_width,
                self.config.depth_height, self.config.depth_fps,
                (OBFormat.Y16, OBFormat.Y14, OBFormat.RLE),
            )
            if self._depth_profile is None:
                raise RuntimeError("no usable depth profile on this device")
            config.enable_stream(self._depth_profile)
        except Exception as exc:
            raise RuntimeError(f"depth stream setup failed: {exc}") from exc

        # FULL_FRAME_REQUIRE makes wait_for_frames return only complete sets, so
        # color and depth we hand out always belong to the same capture.  This is
        # the difference between usable and useless training pairs.
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

        if self.config.enable_frame_sync:
            try:
                self._pipeline.enable_frame_sync()
            except Exception:
                pass  # not all devices support it; continue unsynchronized

        if self.config.align_to_color:
            self._align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        self._pipeline.start(config)

        # Let auto-exposure settle so the first recorded frame is not black.
        for _ in range(max(self.config.warmup_frames, 0)):
            self._pipeline.wait_for_frames(1000)

    @staticmethod
    def _pick_profile(
        profiles: Any,
        fmt: Any,
        width: int,
        height: int,
        fps: int,
        prefer_formats: tuple[Any, ...] = (),
    ) -> Any:
        """Best-effort profile selection with graceful fallbacks.

        The SDK's own ``get_video_stream_profile`` needs a concrete format; with
        ``fmt=None`` it cannot express "any format", so we scan the list
        ourselves in that case.  Without this, the depth_width/depth_height/
        depth_fps settings would be silently ignored and the device default
        (1280x800 RLE) would always win.

        Selection order:
          1. let the SDK resolve an exact request when the format is known,
             first at the requested fps, then at any fps;
          2. scan for profiles matching the requested width/height/fps, ranked by
             ``prefer_formats`` order and then by highest fps;
          3. fall back to the device default rather than refusing to start.
        """
        if fmt is not None:
            for f in (fps, 0):
                try:
                    profile = profiles.get_video_stream_profile(width, height, fmt, f)
                except Exception:
                    profile = None
                if profile is not None:
                    return profile

        candidates: list[tuple[tuple[int, int], Any]] = []
        for i in range(len(profiles)):
            p = profiles[i]
            if width and _safe(p, "get_width") != width:
                continue
            if height and _safe(p, "get_height") != height:
                continue
            p_fmt = _safe(p, "get_format")
            if fmt is not None and p_fmt != fmt:
                continue
            p_fps = _safe(p, "get_fps") or 0
            if fps and p_fps != fps:
                continue
            rank = (
                prefer_formats.index(p_fmt)
                if p_fmt in prefer_formats
                else len(prefer_formats)
            )
            candidates.append(((rank, -p_fps), p))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        try:
            return profiles.get_default_video_stream_profile()
        except Exception:
            return None

    # ------------------------------------------------------------- profiles
    def list_profiles(self) -> dict[str, list[str]]:
        """All available stream profiles, for diagnostics and choosing a config."""
        from pyorbbecsdk import OBSensorType

        out: dict[str, list[str]] = {}
        for label, sensor in (
            ("color", OBSensorType.COLOR_SENSOR),
            ("depth", OBSensorType.DEPTH_SENSOR),
        ):
            entries: list[str] = []
            try:
                profiles = self._pipeline.get_stream_profile_list(sensor)
                for i in range(len(profiles)):
                    p = profiles[i]
                    entries.append(
                        f"{_safe(p, 'get_width', '?')}x{_safe(p, 'get_height', '?')}"
                        f"@{_safe(p, 'get_fps', '?')} {_safe(p, 'get_format', '?')}"
                    )
            except Exception as exc:
                entries.append(f"<error: {exc}>")
            out[label] = entries
        return out

    # ------------------------------------------------------------------ read
    def read(self, timeout_ms: int = 1000) -> Frame | None:
        """Wait for one aligned color+depth pair.  None on timeout."""
        if self._pipeline is None:
            raise RuntimeError("camera not open; call open() first")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        host_time = time.monotonic()
        wall_time = time.time()

        if not frames:
            self.stats.timeouts += 1
            return None

        if self._align is not None:
            aligned = self._align.process(frames)
            if aligned is None:
                self.stats.dropped += 1
                return None
            frames = aligned

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        rgb = None
        color_format = None
        color_ts = None
        ts_source = None
        frame_number = None

        if color_frame is not None:
            color_format = str(_safe(color_frame, "get_format", "?"))
            frame_number = _safe(color_frame, "get_frame_number")
            ts_source, color_ts = frame_timestamp_us(color_frame)
            try:
                rgb = convert_color_to_rgb(color_frame)
            except ValueError:
                # A size mismatch means our format assumption is wrong; treat the
                # frame as unusable rather than shipping a scrambled image.
                self.stats.dropped += 1
                rgb = None

        depth = None
        depth_format = None
        depth_ts = None
        if depth_frame is not None:
            depth_format = str(_safe(depth_frame, "get_format", "?"))
            _, depth_ts = frame_timestamp_us(depth_frame)
            try:
                h = depth_frame.get_height()
                w = depth_frame.get_width()
                raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                if raw.size != h * w:
                    # Almost always means a compressed format (RLE) slipped
                    # through, or the reported geometry is wrong.
                    raise ValueError(
                        f"depth: expected {h * w} uint16 samples, got {raw.size} "
                        f"(format={depth_format})"
                    )
                depth = raw.reshape((h, w))
                scale = _safe(depth_frame, "get_depth_scale", 1.0)
                self._depth_scale = float(scale or 1.0)
                if not self.config.depth_as_uint16:
                    depth = depth.astype(np.float32) * self._depth_scale
            except ValueError:
                self.stats.dropped += 1
                depth = None

        if rgb is None and depth is None:
            return None

        self.stats.frames += 1
        if self.stats.first_host_time is None:
            self.stats.first_host_time = host_time
        self.stats.last_host_time = host_time
        if color_ts is not None:
            self.stats.color_timestamps.append(color_ts)

        return Frame(
            rgb=rgb,
            depth=depth,
            host_time=host_time,
            wall_time=wall_time,
            color_timestamp_us=color_ts,
            depth_timestamp_us=depth_ts,
            timestamp_source=ts_source,
            color_format=color_format,
            depth_format=depth_format,
            depth_scale=self._depth_scale,
            frame_number=frame_number,
        )

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align = None

    def __enter__(self) -> "OrbbecCamera":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
