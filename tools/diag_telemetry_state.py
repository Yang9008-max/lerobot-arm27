"""Diagnose why an aligned snapshot is not available (no arm motion needed).

Prints, once per second, every quantity that can make
``TelemetryResampler.state_at()`` return None:

    sample count      < 2                      -> not enough history
    clock.ready       False                    -> ClockMapper fit not established
    age               > lookahead_ms past newest -> query is "in the future"
    age               > max_age_s              -> stale
    bracket/gap                                -> data hole

It does not need the arm enabled, only powered and streaming.

Run::

    .venv\\Scripts\\python.exe tools\\diag_telemetry_state.py --seconds 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO / ".cache" / "lerobot"))
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--hz", type=float, default=1.0)
    args = ap.parse_args()

    from arm_lerobot.telemetry import acquire_reader, release_reader

    reader = acquire_reader(args.port, args.baud)
    print(f"reader acquired on {args.port} @ {args.baud}")
    print(f"  resampler present : {reader.resampler is not None}")
    print(f"  clock present     : {reader.clock is not None}")
    if reader.resampler is not None:
        print(f"  lookahead_ms      : {reader.resampler.lookahead_ms}")
        print(f"  max_gap_s         : {reader.resampler.max_gap_s}")

    deadline = time.monotonic() + args.seconds
    period = 1.0 / max(args.hz, 0.1)
    n = 0
    try:
        while time.monotonic() < deadline:
            n += 1
            now = time.monotonic()
            res = reader.resampler
            clock = reader.clock

            samples = len(res._samples) if res is not None else 0
            newest = res.newest_t_ms() if res is not None else None
            oldest = res.oldest_t_ms() if res is not None else None
            ready = clock.ready if clock is not None else False
            resid = clock.residual_ms if clock is not None else None
            age = reader.state_age_s(now)
            t_q = clock.t_ms_at(now) if clock is not None else None
            latest = reader.latest()

            print()
            print(f"--- sample {n} ---")
            print(f"  reader stats      : {reader.stats()}")
            print(f"  resampler samples : {samples}")
            print(f"  t_ms oldest/newest: {oldest} / {newest}")
            print(f"  clock.ready       : {ready}   residual={resid}")
            print(f"  t_ms_at(now)      : {t_q}")
            if t_q is not None and newest is not None:
                print(f"  query - newest    : {t_q - newest:+.1f} ms "
                      f"(lookahead {res.lookahead_ms} ms)")
            print(f"  age_s(now)        : {age}")
            print(f"  latest() enabled  : "
                  f"{'n/a' if latest is None else latest.enabled}")
            snap = reader.snapshot(now)
            print(f"  snapshot(now)     : "
                  f"{'None  <-- this is the failure' if snap is None else 'ok'}")
            if snap is not None:
                print(f"    q  = ({snap.q[0]:+.4f}, {snap.q[1]:+.4f}, {snap.q[2]:+.4f})")
                print(f"    q* = ({snap.q_target[0]:+.4f}, {snap.q_target[1]:+.4f}, "
                      f"{snap.q_target[2]:+.4f})")
            print(f"  last_snapshot(0.2): "
                  f"{'None' if reader.last_snapshot(0.2) is None else 'ok'}")

            # The decisive numbers: do t_ms and the host clock advance together,
            # and is the offset stable?  A stale line left in the serial buffer at
            # startup shows up as one sample whose offset differs by tens of
            # seconds - which, with a fitted slope, used to corrupt everything.
            #
            # NB the resampler stores (t_ms, ArmState); only the clock mapper holds
            # the (t_ms, host_time) pairs needed here.
            if clock is not None and len(clock._samples) >= 2:
                pairs = list(clock._samples)
                fmt = lambda t, h: f"(t={t:.0f} host={h:.3f} off={h - 0.001 * t:+.3f})"  # noqa: E731
                print("  first 3 samples   : " + " ".join(fmt(t, h) for t, h in pairs[:3]))
                print("  last 3 samples    : " + " ".join(fmt(t, h) for t, h in pairs[-3:]))
                dt_ms = pairs[-1][0] - pairs[0][0]
                dh_s = pairs[-1][1] - pairs[0][1]
                ratio = (dh_s * 1000.0) / dt_ms if dt_ms else float("nan")
                print(f"  t_ms span         : {dt_ms:.0f} ms of firmware time over "
                      f"{dh_s:.3f} s of host time")
                print(f"  advance ratio     : {ratio:.4f}   "
                      f"({'OK, want ~1.0' if 0.9 < ratio < 1.1 else 'BAD - stale or duplicated samples'})")
                offs = [h - 0.001 * t for t, h in pairs]
                print(f"  offset spread     : min={min(offs):+.3f} max={max(offs):+.3f} "
                      f"({(max(offs) - min(offs)) * 1000:.1f} ms)")

            time.sleep(period)
    finally:
        release_reader(args.port, args.baud)
        print()
        print("reader released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
