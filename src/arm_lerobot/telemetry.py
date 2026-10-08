"""UART7 telemetry parsing for the 27_engineer three-axis Damiao arm.

Wire format
-----------
Physical layer : UART7, 921600 8N1, ASCII, "\\n"-terminated lines (arm.h:398).
Rate           : ARM_KIN_UART_HZ = 20 Hz (arm.h:402).
Producer       : kinUartTick() in project/application/arm/arm.cpp:41-96, which
                 writes TWO lines per tick:

                    kin,  <17 fields>
                    watch,<27 fields>

Scaling
-------
The firmware links nano.specs, which has no %f, so every float is sent as a
scaled integer (arm.cpp:30-39):

    angles   x 1e4   -> rad
    lengths  x 10    -> mm
    speeds   x 1e3   -> rad/s
    torques  x 1e3   -> N*m

Field order is taken verbatim from the snprintf format string at arm.cpp:57-58
and cross-checked against the existing, hardware-verified PC parser
tools/arm_watch.py:24-29.

Why both q and q_target matter
------------------------------
`qb/qa/qe`   are measured joint angles  -> this is the OBSERVATION
`qbt/qat/qet` are commanded joint angles -> this is the ACTION

Both arrive in the same line.  That is what makes phase 1 (read-only data
collection) possible with zero firmware changes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

# ------------------------------------------------------------------- scaling
Q_SCALE = 1e4    # angles, rad
MM_SCALE = 10.0  # lengths, mm
W_SCALE = 1e3    # angular velocity, rad/s
TAU_SCALE = 1e3  # torque, N*m

BAUD = 921600
TELEMETRY_HZ = 20

# --------------------------------------------------------------- field layout
# Order is load-bearing; it must match arm.cpp:57-58 exactly.
WATCH_FIELDS = (
    "t_ms", "en", "safe", "online", "chassis",
    "qb", "qa", "qe", "qbt", "qat", "qet", "r", "z",
    "wb", "wa", "we", "tb", "ta", "te",
    "sw_r", "sw_l", "sw_cr", "sw_cl", "rh", "rv", "lh", "lv",
)

KIN_FIELDS = (
    "t_ms", "en", "calibrated",
    "qb", "qa", "qe", "th2", "th3", "r", "z",
    "cart_mode", "cart_reject", "tour_state", "tour_pose", "safe",
    "pitchb", "pitchf",
)

WATCH_TOKEN_COUNT = 1 + len(WATCH_FIELDS)  # 28
KIN_TOKEN_COUNT = 1 + len(KIN_FIELDS)      # 18

SWITCH_NAMES = {0: "UP", 32: "MID", 64: "DOWN"}

# safe_flags bit 0 = stall.  See arm_model.SAFE_STALL.
SAFE_STALL = 1 << 0


def switch_name(value: int) -> str:
    return SWITCH_NAMES.get(value, str(value))


@dataclass(frozen=True)
class ArmState:
    """One parsed `watch` line: the arm's full observable state at one instant.

    Raw integers are kept as-is (``raw``) so nothing is lost; the convenient
    float properties below apply the documented scaling.
    """

    t_ms: int
    en: int
    safe: int
    online: int
    chassis: int
    qb: int
    qa: int
    qe: int
    qbt: int
    qat: int
    qet: int
    r: int
    z: int
    wb: int
    wa: int
    we: int
    tb: int
    ta: int
    te: int
    sw_r: int
    sw_l: int
    sw_cr: int
    sw_cl: int
    rh: int
    rv: int
    lh: int
    lv: int

    # ---- parsed-at property accessors (cheap, computed on demand) ----
    @property
    def enabled(self) -> bool:
        return bool(self.en)
    @property
    def remote_online(self) -> bool:
        return bool(self.online)
    @property
    def stalled(self) -> bool:
        return bool(self.safe & SAFE_STALL)
    @property
    def chassis_enabled(self) -> bool:
        return bool(self.chassis)

    @property
    def q_base(self) -> float:
        return self.qb / Q_SCALE
    @property
    def q_shoulder(self) -> float:
        return self.qa / Q_SCALE
    @property
    def q_elbow(self) -> float:
        return self.qe / Q_SCALE

    @property
    def q_target_base(self) -> float:
        return self.qbt / Q_SCALE
    @property
    def q_target_shoulder(self) -> float:
        return self.qat / Q_SCALE
    @property
    def q_target_elbow(self) -> float:
        return self.qet / Q_SCALE

    @property
    def r_mm(self) -> float:
        return self.r / MM_SCALE
    @property
    def z_mm(self) -> float:
        return self.z / MM_SCALE

    @property
    def omega(self) -> tuple[float, float, float]:
        """Joint velocities (base, shoulder, elbow), rad/s."""
        return (self.wb / W_SCALE, self.wa / W_SCALE, self.we / W_SCALE)

    @property
    def tau(self) -> tuple[float, float, float]:
        """Joint torques (base, shoulder, elbow), N*m."""
        return (self.tb / TAU_SCALE, self.ta / TAU_SCALE, self.te / TAU_SCALE)

    @property
    def q(self) -> tuple[float, float, float]:
        """Measured joint vector - the LeRobot observation."""
        return (self.q_base, self.q_shoulder, self.q_elbow)

    @property
    def q_target(self) -> tuple[float, float, float]:
        """Commanded joint vector - the LeRobot action."""
        return (self.q_target_base, self.q_target_shoulder, self.q_target_elbow)

    @property
    def switch_right(self) -> str:
        return switch_name(self.sw_r)
    @property
    def switch_left(self) -> str:
        return switch_name(self.sw_l)

    def as_dict(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in WATCH_FIELDS}

    def summary(self) -> str:
        state = "ON " if self.enabled else "off"
        link = "link" if self.remote_online else "LOST"
        stall = " STALL" if self.stalled else ""
        return (
            f"t={self.t_ms:>9d} {state} {link}{stall} swR={self.switch_right:<4} "
            f"q=({self.q_base:+.3f},{self.q_shoulder:+.3f},{self.q_elbow:+.3f}) "
            f"q* =({self.q_target_base:+.3f},{self.q_target_shoulder:+.3f},{self.q_target_elbow:+.3f}) "
            f"rz={self.r_mm:7.1f},{self.z_mm:7.1f}mm"
        )


@dataclass(frozen=True)
class KinState:
    """One parsed `kin` line: geometry status plus calibrated joint angles."""

    t_ms: int
    en: int
    calibrated: int
    qb: int
    qa: int
    qe: int
    th2: int
    th3: int
    r: int
    z: int
    cart_mode: int
    cart_reject: int
    tour_state: int
    tour_pose: int
    safe: int
    pitchb: int
    pitchf: int

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrated)
    @property
    def th2_rad(self) -> float:
        return self.th2 / Q_SCALE
    @property
    def th3_rad(self) -> float:
        return self.th3 / Q_SCALE
    @property
    def cartesian_mode(self) -> bool:
        return bool(self.cart_mode)


def parse_watch(line: str) -> ArmState | None:
    """Parse one `watch` line.  Returns None on anything malformed.

    Never raises: a corrupted line over a long serial run is normal, not an
    exception.  Callers count failures instead.
    """
    parts = line.strip().split(",")
    if len(parts) != WATCH_TOKEN_COUNT or parts[0] != "watch":
        return None
    try:
        values = [int(x) for x in parts[1:]]
    except ValueError:
        return None
    return ArmState(*values)


def parse_kin(line: str) -> KinState | None:
    """Parse one `kin` line.  Returns None on anything malformed."""
    parts = line.strip().split(",")
    if len(parts) != KIN_TOKEN_COUNT or parts[0] != "kin":
        return None
    try:
        values = [int(x) for x in parts[1:]]
    except ValueError:
        return None
    return KinState(*values)


def parse_line(line: str) -> ArmState | KinState | None:
    """Dispatch on the tag.  Unknown lines return None silently."""
    if line.startswith("watch,"):
        return parse_watch(line)
    if line.startswith("kin,"):
        return parse_kin(line)
    return None


# ------------------------------------------------------------------- reader
class TelemetryReader:
    """Background reader for the arm's UART7 telemetry.

    Threaded because a blocking read at 20 Hz must never stall a camera loop or
    a policy forward pass.  The whole point of this layer is that the STM32 owns
    real time, so the PC side is allowed to be lazy.

    Usage::

        with TelemetryReader("COM8") as telem:
            while True:
                s = telem.latest()
                if s: print(s.summary())
                time.sleep(0.2)
    """

    def __init__(
        self,
        port: str,
        baud: int = BAUD,
        history: int = 512,
        resample: bool = True,
    ):
        import serial  # imported lazily so the parser stays usable without pyserial

        self.port = port
        self.serial = serial.Serial(port, baud, timeout=0.2)
        self.serial.reset_input_buffer()

        self._lock = threading.Lock()
        self._state: ArmState | None = None
        self._kin: KinState | None = None
        self._history: deque[ArmState] = deque(maxlen=history)
        self._monotonic_at_state: float | None = None

        # Time alignment, shared by every consumer of this port.  The Robot needs
        # the arm state at each camera frame's instant and the Teleoperator needs
        # the matching target angles; both must come out of ONE resampler or the
        # recorded observation and action would drift apart.
        #
        # Imported inside the function rather than at module scope because
        # arm_lerobot.sync imports this module - a top-level import would be a
        # circular import.
        self.clock = None
        self.resampler = None
        if resample:
            from .sync import ClockMapper, TelemetryResampler

            self.clock = ClockMapper()
            self.resampler = TelemetryResampler(clock=self.clock)

        # The most recent aligned snapshot, so the Teleoperator can reuse the
        # exact instant the Robot aligned its observation to.  LeRobot's record
        # loop calls robot.get_observation() and teleop.get_action() as two
        # independent objects with no way to pass a timestamp between them; this
        # shared slot is how the two stay on one clock.
        self._last_snapshot: tuple[float, object] | None = None

        self.lines_seen = 0
        self.watch_parsed = 0
        self.kin_parsed = 0
        self.parse_failures = 0
        self.unknown_lines = 0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="arm-telemetry")
        self._thread.start()

    # ------------------------------------------------------------- internals
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self.serial.readline()
            except Exception:
                break
            if not raw:
                continue
            self.lines_seen += 1
            try:
                line = raw.decode("ascii", errors="strict").strip()
            except UnicodeDecodeError:
                self.parse_failures += 1
                continue
            if not line:
                continue

            parsed = parse_line(line)
            if parsed is None:
                if line.startswith(("watch", "kin")):
                    self.parse_failures += 1
                else:
                    self.unknown_lines += 1
                continue

            now = time.monotonic()
            with self._lock:
                if isinstance(parsed, ArmState):
                    self.watch_parsed += 1
                    self._state = parsed
                    self._monotonic_at_state = now
                    self._history.append(parsed)
                    # Feed time alignment under the same lock snapshot() takes,
                    # so the resampler is never mutated while it is being read.
                    if self.resampler is not None:
                        self.resampler.add(parsed, now)
                else:
                    self.kin_parsed += 1
                    self._kin = parsed

    # ------------------------------------------------------------------- api
    def latest(self) -> ArmState | None:
        with self._lock:
            return self._state

    # ------------------------------------------------------- aligned snapshots
    def snapshot(self, host_time: float):
        """Arm state interpolated to ``host_time``, or None if unavailable.

        Returns an ``arm_lerobot.sync.InterpolatedState``.  The result is cached as
        the most recent snapshot so a Teleoperator can reuse the exact instant the
        Robot aligned its observation to.
        """
        if self.resampler is None:
            return None
        with self._lock:
            state = self.resampler.state_at(host_time)
            if state is not None:
                self._last_snapshot = (host_time, state)
        return state

    def last_snapshot(self, max_age_s: float = 0.2):
        """Most recent successful snapshot, if not older than ``max_age_s``.

        The age is measured against the host time the snapshot was *requested* for,
        which is what the observation was aligned to - not against wall-clock now.
        """
        with self._lock:
            entry = self._last_snapshot
        if entry is None:
            return None
        host_time, state = entry
        if time.monotonic() - host_time > max_age_s:
            return None
        return state

    def state_age_s(self, host_time: float) -> float | None:
        """Age of the newest telemetry sample relative to ``host_time``."""
        if self.resampler is None:
            return None
        with self._lock:
            return self.resampler.age_s(host_time)

    def kin(self) -> KinState | None:
        with self._lock:
            return self._kin

    def age_s(self) -> float | None:
        """Seconds since the last valid `watch` line.  None if none yet."""
        with self._lock:
            if self._monotonic_at_state is None:
                return None
            return time.monotonic() - self._monotonic_at_state

    def history(self) -> list[ArmState]:
        with self._lock:
            return list(self._history)

    def stats(self) -> dict[str, int]:
        return {
            "lines_seen": self.lines_seen,
            "watch_parsed": self.watch_parsed,
            "kin_parsed": self.kin_parsed,
            "parse_failures": self.parse_failures,
            "unknown_lines": self.unknown_lines,
        }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.serial.close()
        except Exception:
            pass

    def __enter__(self) -> "TelemetryReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------- shared reader registry
# LeRobot builds the Robot and the Teleoperator as two independent objects, but
# in this project both need the *same* serial telemetry: the Robot contributes
# `q` (measured) as the observation, the Teleoperator contributes `q_target`
# (commanded) as the action.  Two TelemetryReader objects on one COM port would
# each steal roughly half of the incoming lines, so they must share one.
#
# A tiny refcounted registry keeps the port open for as long as anybody needs it
# and closes it when the last user lets go.

_shared_readers: dict[tuple[str, int], list] = {}
_shared_lock = threading.Lock()


def acquire_reader(port: str, baud: int = BAUD) -> "TelemetryReader":
    """Return a process-wide shared reader for ``port``, opening it if needed.

    Every successful acquire must be paired with exactly one
    :func:`release_reader` call.
    """
    key = (port.upper(), baud)
    with _shared_lock:
        entry = _shared_readers.get(key)
        if entry is None:
            entry = [TelemetryReader(port, baud), 0]
            _shared_readers[key] = entry
        entry[1] += 1
        return entry[0]


def release_reader(port: str, baud: int = BAUD) -> None:
    """Drop one reference; closes the port when the count reaches zero."""
    key = (port.upper(), baud)
    with _shared_lock:
        entry = _shared_readers.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            entry[0].close()
            del _shared_readers[key]


def shared_reader_count() -> int:
    """How many distinct ports are currently held open.  For tests/diagnostics."""
    with _shared_lock:
        return len(_shared_readers)


# ----------------------------------------------------------------- autodetect
def list_serial_ports() -> list[tuple[str, str]]:
    """Return [(device, description), ...] for every visible COM port."""
    from serial.tools import list_ports

    return [(p.device, p.description or "") for p in list_ports.comports()]


def autodetect_port(timeout_s: float = 3.0, probe_s: float = 1.5) -> str | None:
    """Scan COM ports and return the first one that emits `watch,`/`kin,` lines.

    This machine enumerates a lot of CH340 adapters, so guessing by name is
    hopeless; identify by actual telemetry instead.
    """
    import serial  # noqa: F401  (import here keeps list_serial_ports import-light)

    for device, description in list_serial_ports():
        try:
            with serial.Serial(device, BAUD, timeout=probe_s) as ser:
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    raw = ser.readline()
                    if not raw:
                        continue
                    text = raw.decode("ascii", errors="ignore").strip()
                    if text.startswith(("watch,", "kin,")):
                        return device
        except Exception:
            continue
    return None
