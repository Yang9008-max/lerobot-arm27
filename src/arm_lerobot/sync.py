"""Time alignment between the STM32 telemetry stream and the camera stream.

Why this module exists
----------------------
The arm telemetry arrives at 20 Hz over USB serial, the camera at 30 Hz over USB3.
Neither stream is timestamped on a clock the other shares:

    STM32   emits its own millisecond counter, ``t_ms``  (arm.cpp:59)
    Gemini2 hardware timestamps are unusable on this machine (constant 2**33)
    Windows does not give us a shared clock either

So both are observed on the host's ``time.monotonic()`` at arrival.  Naively
pairing each camera frame with "whatever telemetry arrived last" is wrong twice
over:

  1. USB scheduling jitter makes the *arrival* time a noisy observation of the
     instant the firmware actually sampled.
  2. At 20 Hz versus 30 Hz the sample you happen to hold can be up to 50 ms old,
     and a moving arm covers real distance in 50 ms.

The fix is to stop treating arrival time as truth:

  * ``ClockMapper`` fits ``host_time = slope * t_ms + intercept`` over a sliding
    window.  The firmware's ``t_ms`` is clean and monotonic, so the fit recovers
    a de-jittered mapping from firmware time to host time.
  * ``TelemetryResampler`` then interpolates joint state *in t_ms space* at the
    exact firmware time corresponding to a camera frame.

That turns a 50 ms worst-case pairing error into a sub-millisecond interpolated
estimate, which is what ACT training data actually needs.

Interpolation and liveness are separate concerns
------------------------------------------------
``TelemetryResampler.state_at()`` is pure interpolation over whatever history is
buffered.  It answers "what was the arm doing at this instant", including for
times well in the past - which is exactly what offline re-alignment of a recorded
log needs.

It deliberately does *not* decide whether the data is fresh.  A live recording
loop must ask that separately via ``age_s()``, because "stale" only has meaning
relative to the caller's notion of now.  Conflating the two made earlier versions
refuse valid historical queries.

Known precision limit
---------------------
The firmware sends ``(long)(now_ms + 0.5f)`` where ``now_ms`` is a float32
(arm.cpp:59).  A float32 mantissa is 24 bits, so above 2**24 ms (~4.66 hours) the
millisecond value itself starts quantising:

    < 16.7 min : 0.0625 ms resolution
    ~ 4.66 h   : 1 ms
    ~ 9.3  h   : 2 ms

Perception datasets are collected in minutes, so this is a documented caveat
rather than a blocker.  Restart the firmware before a long session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .telemetry import Q_SCALE, TAU_SCALE, W_SCALE, ArmState

# Largest acceptable spacing between the two telemetry samples that bracket a
# query.  At the firmware's 20 Hz that is 50 ms; 250 ms means we tolerated five
# consecutive misses, which should be treated as a data hole, not interpolated.
DEFAULT_MAX_GAP_S = 0.25

# How far past the newest buffered sample a query may reach.
#
# Telemetry nominally arrives every 50 ms, but it is measured, not assumed: a
# first live run at 30 fps camera / 20 Hz telemetry dropped 4 frames out of 244
# with "age=0.063s" - i.e. the newest sample was 63 ms old, just past a 60 ms
# limit, purely because USB scheduling stretched one interval.  90 ms covers one
# full interval plus the observed jitter, and the extrapolation error stays in the
# second-order term (see the comment in TelemetryResampler.state_at).
DEFAULT_LOOKAHEAD_MS = 90.0


class ClockMapper:
    """Map firmware ``t_ms`` to host ``time.monotonic()``.

    The slope is NOT estimated from the data.  It is physically fixed: one
    millisecond of firmware time is one millisecond of host time, to within the
    ppm-level difference between two crystals, which over the 6.4 s fit window is
    far below a millisecond.  Only the offset is estimated, as the MEDIAN of
    ``host - 1e-3 * t_ms``.

    Both of those choices were forced by hardware.  The first implementation did a
    least-squares fit of BOTH parameters, and a single stale telemetry line left in
    the serial buffer at startup - its ``t_ms`` about 50 s behind everything else -
    tilted the fitted slope to 5.75e-5 instead of 1e-3.  That put every subsequent
    query ~40 000 ms into the "future", past the lookahead, so the alignment layer
    refused to answer at all and recording died before the first frame.

    A median offset cannot be dragged by a minority of bad samples, and freezing
    the slope removes the parameter that was being corrupted.  Slow clock drift is
    still tracked, because the window slides and the offset is re-estimated
    locally.  ``residual_ms`` reports the robust spread; a jump in it means the
    streams stopped tracking each other.
    """

    SLOPE = 1e-3  # seconds of host time per millisecond of firmware time

    def __init__(
        self,
        window: int = 128,
        min_samples: int = 8,
        reset_jump_ms: float = 1000.0,
    ):
        self.window = window
        self.min_samples = min_samples
        self.reset_jump_ms = reset_jump_ms

        self._samples: deque[tuple[float, float]] = deque(maxlen=window)
        self._slope: float | None = None
        self._intercept: float | None = None
        self._residual_ms: float | None = None
        self.resets = 0

    # ------------------------------------------------------------------ input
    def add(self, t_ms: float, host_time: float) -> None:
        """Record one observation.  Detects stream discontinuities automatically."""
        if self._samples:
            delta = t_ms - self._samples[-1][0]
            # A discontinuity in EITHER direction starts a new epoch.
            #
            # Backwards means the board rebooted and its counter restarted.
            #
            # Forwards by more than a second means the lines read so far were a
            # STALE SERIAL BACKLOG.  The firmware streams at 20 Hz whether or not
            # anyone is listening, so the first lines available after opening the
            # port can be a minute old.  Measured live: the first 24 lines were
            # 56 s behind the rest, and they drained in a burst (50 ms of firmware
            # time arriving in 4 ms of host time).
            #
            # That matters because the offset is a median: with 24 stale lines
            # against 8 fresh ones the stale mode wins, the offset lands 56 s out,
            # and every subsequent query falls in the hole between the stale and
            # fresh regions.  state_at() then correctly refuses to interpolate
            # across it, so recording died with "no usable arm telemetry" at the
            # exact moment it started.  Dropping the prefix is the honest fix.
            if delta < -self.reset_jump_ms or delta > self.reset_jump_ms:
                self._samples.clear()
                self._slope = self._intercept = self._residual_ms = None
                self.resets += 1

        self._samples.append((float(t_ms), float(host_time)))
        if len(self._samples) >= self.min_samples:
            self._refit()

    # ------------------------------------------------------------------- fit
    def _refit(self) -> None:
        offsets = [host - self.SLOPE * t for t, host in self._samples]
        median = _median(offsets)
        deviations = [abs(o - median) for o in offsets]
        # 1.4826 * MAD is the consistent estimator of sigma for normal data.
        self._residual_ms = 1000.0 * 1.4826 * _median(deviations)
        self._slope = self.SLOPE
        self._intercept = median

    # ------------------------------------------------------------------ query
    @property
    def ready(self) -> bool:
        return self._slope is not None and self._intercept is not None

    @property
    def slope(self) -> float | None:
        """Seconds of host time per millisecond of firmware time (~1e-3)."""
        return self._slope

    @property
    def intercept(self) -> float | None:
        return self._intercept

    @property
    def residual_ms(self) -> float | None:
        """RMS fit residual in milliseconds.  A few ms is normal USB jitter."""
        return self._residual_ms

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def host_at(self, t_ms: float) -> float | None:
        if self._slope is None or self._intercept is None:
            return None
        return self._slope * t_ms + self._intercept

    def t_ms_at(self, host_time: float) -> float | None:
        """Invert the fit: which firmware time corresponds to this host time?"""
        if self._slope is None or self._intercept is None or self._slope == 0.0:
            return None
        return (host_time - self._intercept) / self._slope

    def describe(self) -> str:
        if not self.ready:
            return f"ClockMapper(not ready, {len(self._samples)} samples)"
        return (
            f"ClockMapper(n={len(self._samples)} slope={self._slope:.6f} "
            f"intercept={self._intercept:+.4f}s residual={self._residual_ms:.2f}ms "
            f"resets={self.resets})"
        )


@dataclass(frozen=True)
class InterpolatedState:
    """Arm state resampled to an exact instant, in firmware time."""

    t_ms: float
    q: tuple[float, float, float]          # measured joint angles, rad
    q_target: tuple[float, float, float]   # commanded joint angles, rad
    omega: tuple[float, float, float]      # rad/s
    tau: tuple[float, float, float]        # N*m
    enabled: bool
    remote_online: bool
    stalled: bool
    gap_s: float                           # spacing of the bracketing samples
    clamped: bool                          # query was past the newest sample
    host_time: float

    @property
    def tracking_error(self) -> tuple[float, float, float]:
        """q_target - q.  Large values mean the arm is not where it was told to be."""
        return tuple(a - b for a, b in zip(self.q_target, self.q))


class TelemetryResampler:
    """Interpolate arm state at arbitrary host times.

    Samples are keyed by firmware ``t_ms`` (clean, monotonic) and queries are
    converted into that space through a :class:`ClockMapper`.  Interpolation is
    linear, which is correct for position over a 50 ms window: at the arm's
    fastest joint rate the curvature over 50 ms is far below the encoder noise.

    Refuses to answer rather than guessing when:
      * fewer than two samples are buffered, or the clock fit is not ready;
      * the query lies before the oldest buffered sample;
      * the query lies more than ``lookahead_ms`` past the newest sample;
      * the bracketing samples are more than ``max_gap_s`` apart (a data hole).

    Liveness is the caller's business - see ``age_s()``.
    """

    def __init__(
        self,
        clock: ClockMapper | None = None,
        max_gap_s: float = DEFAULT_MAX_GAP_S,
        lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
        buffer: int = 8192,
    ):
        self.clock = clock or ClockMapper()
        self.max_gap_s = max_gap_s
        self.lookahead_ms = lookahead_ms
        self._samples: deque[tuple[float, ArmState]] = deque(maxlen=buffer)

        self.rejected_gap = 0
        self.rejected_range = 0
        self.rejected_future = 0
        self.clamped_count = 0

    # ------------------------------------------------------------------ input
    def add(self, state: ArmState, host_time: float) -> None:
        self.clock.add(state.t_ms, host_time)
        self._samples.append((float(state.t_ms), state))

    # -------------------------------------------------------------- liveness
    def newest_t_ms(self) -> float | None:
        return self._samples[-1][0] if self._samples else None

    def oldest_t_ms(self) -> float | None:
        return self._samples[0][0] if self._samples else None

    def newest_host_time(self) -> float | None:
        """Host time (by the fitted clock) of the most recent telemetry sample."""
        newest = self.newest_t_ms()
        return None if newest is None else self.clock.host_at(newest)

    def age_s(self, host_time: float) -> float | None:
        """How old the newest telemetry is relative to ``host_time``.

        Positive means telemetry is behind (normal); negative means the query is
        ahead of the newest sample.  A live recording loop should refuse to
        record frames when this exceeds its staleness budget.
        """
        newest_host = self.newest_host_time()
        return None if newest_host is None else host_time - newest_host

    # ------------------------------------------------------------------ query
    def state_at(self, host_time: float) -> InterpolatedState | None:
        if len(self._samples) < 2 or not self.clock.ready:
            return None

        t_query = self.clock.t_ms_at(host_time)
        if t_query is None:
            return None

        newest = self._samples[-1][0]
        oldest = self._samples[0][0]

        if t_query > newest + self.lookahead_ms:
            self.rejected_future += 1
            return None
        if t_query < oldest:
            self.rejected_range += 1
            return None

        lo, hi = self._bracket(t_query)
        if lo is None or hi is None:
            self.rejected_range += 1
            return None

        t0, s0 = self._samples[lo]
        t1, s1 = self._samples[hi]
        gap_s = (t1 - t0) / 1000.0
        if gap_s > self.max_gap_s:
            self.rejected_gap += 1
            return None

        if t1 == t0:
            alpha = 0.0
            clamped = False
        else:
            alpha = (t_query - t0) / (t1 - t0)
            # alpha > 1 means the query sits past the newest telemetry sample.
            # That is the COMMON case here, not an edge case: the camera runs at
            # 30 Hz against 20 Hz telemetry, so a frame almost always lands
            # between two lines, and USB jitter decides whether the line it needs
            # has arrived yet.
            #
            # Holding the newest value (alpha clamped to 1.0) is wrong by
            # joint_velocity * gap.  Measured on this arm: 2.35e-2 rad for a 50 ms
            # gap, i.e. 235 encoder ticks - catastrophic for training data, and a
            # pure timing artefact rather than a real measurement error.
            #
            # So extrapolate along the last interval instead.  The cost is only
            # the second-order term (~5e-4 rad) because the arm cannot accelerate
            # meaningfully in 50 ms.  The overshoot is already bounded above by
            # lookahead_ms in the range check earlier in this method.
            clamped = alpha > 1.0

        if alpha < 0.0:
            alpha = 0.0
        if clamped:
            self.clamped_count += 1

        def lerp(a: int, b: int) -> float:
            return (a + (b - a) * alpha) / Q_SCALE

        def lerp_scaled(a: int, b: int, scale: float) -> float:
            return (a + (b - a) * alpha) / scale

        return InterpolatedState(
            t_ms=t_query,
            q=(lerp(s0.qb, s1.qb), lerp(s0.qa, s1.qa), lerp(s0.qe, s1.qe)),
            q_target=(
                lerp(s0.qbt, s1.qbt),
                lerp(s0.qat, s1.qat),
                lerp(s0.qet, s1.qet),
            ),
            omega=(
                lerp_scaled(s0.wb, s1.wb, W_SCALE),
                lerp_scaled(s0.wa, s1.wa, W_SCALE),
                lerp_scaled(s0.we, s1.we, W_SCALE),
            ),
            tau=(
                lerp_scaled(s0.tb, s1.tb, TAU_SCALE),
                lerp_scaled(s0.ta, s1.ta, TAU_SCALE),
                lerp_scaled(s0.te, s1.te, TAU_SCALE),
            ),
            enabled=bool(s0.en and s1.en),
            remote_online=bool(s0.online and s1.online),
            stalled=bool(s0.stalled and s1.stalled),
            gap_s=gap_s,
            clamped=clamped,
            host_time=host_time,
        )

    def _bracket(self, t_query: float) -> tuple[int | None, int | None]:
        """Indices of the samples immediately before and after t_query.

        The deque is append-ordered and t_ms is monotonic within a firmware run,
        so a linear scan from the right is both correct and fast for the small
        lookback this needs.
        """
        n = len(self._samples)
        hi = None
        for i in range(n - 1, -1, -1):
            if self._samples[i][0] <= t_query:
                hi = i
                break
        if hi is None:
            return None, None
        if hi == n - 1:
            # Query is past the newest sample: use the final interval, and
            # state_at() will clamp alpha so we never invent future motion.
            return n - 2, n - 1
        return hi, hi + 1


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])
