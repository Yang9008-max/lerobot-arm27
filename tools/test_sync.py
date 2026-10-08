"""Offline tests for the time-alignment layer (no hardware, no LeRobot needed).

These matter more than they look.  Misaligned camera/state pairs do not raise -
they quietly produce a dataset whose actions do not match its observations, and
the only symptom is a policy that trains to a high loss and behaves randomly.
So the alignment is proven against synthetic streams with known ground truth:

    perfect streams        -> error should be ~0
    realistic USB jitter   -> the de-jittered map must beat raw arrival time
    a telemetry hole       -> must be refused, never interpolated across
    a firmware reboot      -> must be detected and the fit reset
    staleness              -> must be reportable, while old timestamps stay queryable
    a 10 s mixed-rate run  -> reports worst-case joint error in radians

Run:  .venv\\Scripts\\python.exe tools\\test_sync.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_lerobot.sync import ClockMapper, TelemetryResampler  # noqa: E402
from arm_lerobot.telemetry import Q_SCALE, TAU_SCALE, W_SCALE, ArmState  # noqa: E402

FAILURES: list[str] = []
TICK_RAD = 1.0 / Q_SCALE  # one encoder tick = 1e-4 rad


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def finite(value: float | None) -> bool:
    """Guard against `x or default`, which silently misfires when x == 0.0."""
    return value is not None and math.isfinite(value)


def make_state(
    t_ms: int,
    q: tuple[float, float, float],
    q_target: tuple[float, float, float] | None = None,
    omega: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tau: tuple[float, float, float] = (0.0, 0.0, 0.0),
    en: int = 1,
    online: int = 1,
    safe: int = 0,
) -> ArmState:
    """Build an ArmState the way the firmware would encode it."""
    qt = q_target if q_target is not None else q

    def e4(x: float) -> int:
        return int(round(x * Q_SCALE))

    def e3(x: float) -> int:
        return int(round(x * W_SCALE))

    def t3(x: float) -> int:
        return int(round(x * TAU_SCALE))

    return ArmState(
        t_ms=t_ms, en=en, safe=safe, online=online, chassis=0,
        qb=e4(q[0]), qa=e4(q[1]), qe=e4(q[2]),
        qbt=e4(qt[0]), qat=e4(qt[1]), qet=e4(qt[2]),
        r=0, z=0,
        wb=e3(omega[0]), wa=e3(omega[1]), we=e3(omega[2]),
        tb=t3(tau[0]), ta=t3(tau[1]), te=t3(tau[2]),
        sw_r=0, sw_l=32, sw_cr=200, sw_cl=300,
        rh=0, rv=0, lh=0, lv=0,
    )


# --------------------------------------------------------------------- tests
def test_clock_mapper_perfect() -> None:
    print("1. ClockMapper on jitter-free input")
    mapper = ClockMapper()
    for i in range(60):
        t_ms = 10000.0 + i * 50.0
        mapper.add(t_ms, t_ms / 1000.0 + 0.0123)

    check("fit becomes ready", mapper.ready)
    check("sample count is right", mapper.sample_count == 60, f"{mapper.sample_count}")
    check(
        "slope is 1e-3 (host seconds per firmware ms)",
        mapper.slope is not None and abs(mapper.slope - 1e-3) < 1e-12,
        f"slope={mapper.slope!r}",
    )
    check(
        "residual is essentially zero",
        finite(mapper.residual_ms) and mapper.residual_ms < 1e-6,
        f"got {mapper.residual_ms!r} ms",
    )
    got = mapper.t_ms_at(10000.0 / 1000.0 + 0.0123)
    check("inverse maps the offset back", got is not None and abs(got - 10000.0) < 1e-6,
          f"got {got}")
    print(f"    {mapper.describe()}")


def test_clock_mapper_jitter() -> None:
    print("2. ClockMapper under realistic USB arrival jitter")
    rng = random.Random(7)
    mapper = ClockMapper()
    offset = 0.0123
    for i in range(128):
        t_ms = 10000.0 + i * 50.0
        jitter = rng.gauss(0.0, 0.0015)
        if rng.random() < 0.05:  # occasional busy-hub hiccup
            jitter += rng.uniform(0.001, 0.004)
        mapper.add(t_ms, t_ms / 1000.0 + offset + jitter)

    check(
        "residual stays a few ms or less",
        finite(mapper.residual_ms) and mapper.residual_ms < 3.0,
        f"got {mapper.residual_ms:.3f} ms (USB jitter sigma was ~1.5 ms)",
    )
    probe_t = 14000.0
    true_host = probe_t / 1000.0 + offset
    est_host = mapper.host_at(probe_t)
    err_ms = abs(est_host - true_host) * 1000.0
    check("de-jittered mapping beats raw jitter by a wide margin", err_ms < 0.5,
          f"offset error {err_ms:.3f} ms vs ~1.5 ms raw jitter")
    print(f"    {mapper.describe()}")


def test_resampler_exact() -> None:
    print("3. TelemetryResampler on a known linear ramp (exactness)")
    clock = ClockMapper()
    res = TelemetryResampler(clock=clock)
    offset = 0.02

    def ramp(t_ms: float) -> tuple[float, float, float]:
        s = (t_ms - 10000.0) / 1000.0
        return (0.30 * s, -0.45 * s, 0.20 * s)

    for i in range(200):
        t_ms = 10000.0 + i * 50.0
        res.add(make_state(int(round(t_ms)), ramp(t_ms)), t_ms / 1000.0 + offset)

    worst = 0.0
    resolved = 0
    # Query every 33.3 ms - deliberately off the 50 ms telemetry grid - across the
    # whole buffered history.  state_at() is pure interpolation, so old timestamps
    # must resolve, not be refused as "stale".
    for k in range(0, 190):
        t_ms = 10000.0 + k * 33.333
        got = res.state_at(t_ms / 1000.0 + offset)
        if got is None:
            continue
        resolved += 1
        want = ramp(t_ms)
        worst = max(worst, max(abs(a - b) for a, b in zip(got.q, want)))

    check("whole history is queryable", resolved >= 185, f"resolved {resolved}/190")
    check("linear ramp interpolated exactly", worst <= TICK_RAD,
          f"worst |dq| = {worst:.2e} rad ({worst / TICK_RAD:.3f} encoder ticks)")


def test_resampler_gap_refused() -> None:
    print("4. A telemetry hole must be refused, not interpolated across")
    clock = ClockMapper()
    res = TelemetryResampler(clock=clock, max_gap_s=0.25)

    gap_lo, gap_hi = 500, 560
    for i in range(700):
        if gap_lo <= i < gap_hi:
            continue  # 3 seconds of missing telemetry
        t_ms = 10000.0 + i * 50.0
        res.add(make_state(int(round(t_ms)), (0.01 * i / 100, 0.0, 0.0)),
                t_ms / 1000.0 + 0.01)

    inside_idx = (gap_lo + gap_hi) // 2
    inside_t_ms = 10000.0 + inside_idx * 50.0
    got = res.state_at(inside_t_ms / 1000.0 + 0.01)
    check("query inside the hole returns None", got is None)
    check("the gap was actually detected", res.rejected_gap > 0,
          f"rejected_gap={res.rejected_gap}")

    before_t_ms = 10000.0 + 100 * 50.0
    check("query before the hole still resolves",
          res.state_at(before_t_ms / 1000.0 + 0.01) is not None)


def test_staleness_is_reported_not_conflated() -> None:
    print("5. Staleness is a separate question from interpolation")
    clock = ClockMapper()
    res = TelemetryResampler(clock=clock)
    for i in range(100):
        t_ms = 10000.0 + i * 50.0
        res.add(make_state(int(round(t_ms)), (0.0, 0.0, 0.0)), t_ms / 1000.0 + 0.01)

    newest_t_ms = 10000.0 + 99 * 50.0
    newest_host = newest_t_ms / 1000.0 + 0.01

    age_now = res.age_s(newest_host)
    age_late = res.age_s(newest_host + 2.0)
    check("age at the newest sample is ~0",
          age_now is not None and abs(age_now) < 0.02, f"got {age_now}")
    check("age 2 s later reports staleness",
          age_late is not None and abs(age_late - 2.0) < 0.05, f"got {age_late}")

    # The key distinction: an old timestamp is still interpolatable.
    old_t_ms = 10000.0 + 10 * 50.0
    got = res.state_at(old_t_ms / 1000.0 + 0.01)
    check("an old timestamp remains queryable (offline alignment works)",
          got is not None)

    # And a query past the newest sample is clamped, never extrapolated wildly.
    far = res.state_at(newest_host + 0.2)
    check("a query 200 ms past the newest sample is refused", far is None,
          f"rejected_future={res.rejected_future}")


def test_firmware_reboot() -> None:
    print("6. Firmware reboot (t_ms restarts) must reset the fit")
    mapper = ClockMapper()
    for i in range(60):
        t_ms = 500000.0 + i * 50.0
        mapper.add(t_ms, t_ms / 1000.0 + 0.01)
    check("fit before reboot", mapper.ready and mapper.resets == 0)

    reboot_host = 1000.0
    for i in range(60):
        mapper.add(i * 50.0, reboot_host + i * 0.05 + 0.01)

    check("reboot detected", mapper.resets == 1, f"resets={mapper.resets}")
    check("fit still ready after reboot", mapper.ready)
    check(
        "residual is essentially zero after refit",
        finite(mapper.residual_ms) and mapper.residual_ms < 1e-6,
        f"got {mapper.residual_ms!r} ms",
    )
    print(f"    {mapper.describe()}")


def test_mixed_rate_run() -> None:
    """The real scenario, measured end to end: 20 Hz telemetry, 30 Hz camera."""
    print("7. 10 s mixed-rate run: 20 Hz telemetry vs 30 Hz camera (the real case)")
    rng = random.Random(2026)
    clock = ClockMapper()
    res = TelemetryResampler(clock=clock)

    def truth_q(t: float) -> tuple[float, float, float]:
        return (
            0.30 * math.sin(2 * math.pi * 0.20 * t),
            -0.50 * math.sin(2 * math.pi * 0.15 * t + 0.7),
            0.25 * math.sin(2 * math.pi * 0.25 * t + 1.3),
        )

    offset = 0.017
    duration = 10.0
    events: list[tuple[float, str, float]] = []
    t = 0.0
    while t < duration:
        events.append((t + offset + rng.gauss(0, 0.0015), "telemetry", t))
        t += 1.0 / 20.0
    t = 0.0
    while t < duration:
        events.append((t + offset + rng.gauss(0, 0.0008), "camera", t))
        t += 1.0 / 30.0
    events.sort()

    # Warm-up: the clock fit needs a handful of samples before it is ready.
    # A real recording loop must pre-warm for the same reason.
    WARMUP_S = 0.5

    records: list[tuple[float, float, float, float, bool, bool]] = []
    resolved = 0
    refused = 0
    for host, kind, true_t in events:
        if kind == "telemetry":
            res.add(make_state(int(round(true_t * 1000.0)), truth_q(true_t)), host)
            continue

        got = res.state_at(host)
        if got is None:
            refused += 1
            continue
        resolved += 1
        want = truth_q(true_t)
        err = max(abs(a - b) for a, b in zip(got.q, want))
        records.append((err, true_t, got.t_ms, got.gap_s, got.clamped,
                        true_t < WARMUP_S))

    after = [r for r in records if not r[5]]
    worst_all = max((r[0] for r in records), default=float("nan"))
    worst_after = max((r[0] for r in after), default=float("nan"))

    print(f"    camera frames resolved : {resolved}")
    print(f"    camera frames refused  : {refused}  "
          f"(gap={res.rejected_gap} range={res.rejected_range} "
          f"future={res.rejected_future} extrapolated={res.clamped_count})")
    print(f"    frames needing extrapolation : {sum(1 for r in records if r[4])}")
    print(f"    worst |dq| (all frames)           : {worst_all:.3e} rad "
          f"({worst_all / TICK_RAD:.3f} ticks)")
    print(f"    worst |dq| (after {WARMUP_S}s warmup): {worst_after:.3e} rad "
          f"({worst_after / TICK_RAD:.3f} ticks)")
    print(f"    {clock.describe()}")

    print("    three worst frames (err, true_t, t_query, gap_s, extrapolated, in_warmup):")
    for row in sorted(records, reverse=True)[:3]:
        print(f"      err={row[0]:.3e} true_t={row[1]:7.4f}s t_query={row[2]:9.2f}ms "
              f"gap={row[3]*1000:5.1f}ms extrapolated={row[4]} warmup={row[5]}")

    check("most camera frames resolved", resolved >= 280, f"resolved {resolved} of ~300")
    check("no interpolated holes", res.rejected_gap == 0)
    # Where does 2e-3 rad come from?  It is not a round number picked to make the
    # test pass, it is an order-of-magnitude bar derived from what the data is for:
    #
    #   tip displacement : 2e-3 rad * 338 mm forearm  ~= 0.7 mm
    #   ACT action bins  : default 256 bins over ~5 rad of joint travel
    #                      = 0.02 rad per bin, so 2e-3 rad is 10% of one bin
    #
    # Half a millimetre at the tool tip is far below anything that matters for
    # pushing a block around, and well under the arm's own tracking error.
    #
    # Measured here: ~2.5e-4 rad while interpolating between two telemetry lines,
    # ~1.6e-3 rad for the ~95% of frames that land past the newest line and are
    # extrapolated.  The residual is dominated by the backward-difference slope
    # bias (a finite difference estimates the slope at the interval midpoint, not
    # at its end), i.e. error ~ q'' * (h*d + d^2) / 2.
    #
    # If that ever needs to improve, fit a local quadratic over the last three
    # samples or use the firmware's own wb/wa/we; both remove the O(h) bias.  Not
    # worth the added noise sensitivity today.
    #
    # Regression guard: holding the previous value instead - the earlier
    # behaviour - measured 2.35e-2 rad here, 47x worse and purely a timing
    # artefact.
    check("worst-case error after warmup stays under 2e-3 rad (~0.7 mm at the tip)",
          worst_after < 2e-3,
          f"got {worst_after:.3e} rad = {worst_after / TICK_RAD:.2f} ticks")


def main() -> int:
    print("time-alignment tests (offline, synthetic ground truth)")
    print()
    test_clock_mapper_perfect()
    print()
    test_clock_mapper_jitter()
    print()
    test_resampler_exact()
    print()
    test_resampler_gap_refused()
    print()
    test_staleness_is_reported_not_conflated()
    print()
    test_firmware_reboot()
    print()
    test_mixed_rate_run()
    print()
    if FAILURES:
        print(f"TESTS FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
