"""Probe the arm's UART7 telemetry link (phase 1a verification tool).

Purpose: prove the PC can read the arm's state, and prove the parser agrees with
the firmware, BEFORE any LeRobot code is involved.  Everything downstream
depends on this working.

    python tools/probe_telemetry.py --list
    python tools/probe_telemetry.py --autodetect
    python tools/probe_telemetry.py COM8
    python tools/probe_telemetry.py COM8 --raw --seconds 10
    python tools/probe_telemetry.py --self-test      # no hardware needed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arm_lerobot import arm_model, telemetry  # noqa: E402


def cmd_self_test() -> int:
    """Validate the parser against hand-built lines.  No serial port needed."""
    print("self-test: parser and model, no hardware required")
    print()

    # Build a synthetic but byte-accurate `watch` line from the documented schema.
    fields = {
        "t_ms": 123456, "en": 1, "safe": 0, "online": 1, "chassis": 0,
        "qb": 2500, "qa": -18000, "qe": 12000,
        "qbt": 2600, "qat": -17900, "qet": 12100,
        "r": 4200, "z": 3800,
        "wb": 12, "wa": -340, "we": 210,
        "tb": 150, "ta": 2100, "te": -1800,
        "sw_r": 0, "sw_l": 32, "sw_cr": 200, "sw_cl": 300,
        "rh": 0, "rv": -500, "lh": 120, "lv": 0,
    }
    line = "watch," + ",".join(str(fields[f]) for f in telemetry.WATCH_FIELDS)
    print(f"synthetic line has {len(line.split(','))} tokens "
          f"(schema says {telemetry.WATCH_TOKEN_COUNT})")

    state = telemetry.parse_watch(line)
    if state is None:
        print("FAIL: parser rejected a well-formed line")
        return 1

    checks: list[tuple[str, object, object]] = [
        ("q_base rad", round(state.q_base, 4), 0.25),
        ("q_shoulder rad", round(state.q_shoulder, 4), -1.8),
        ("q_elbow rad", round(state.q_elbow, 4), 1.2),
        ("target shoulder", round(state.q_target_shoulder, 4), -1.79),
        ("r_mm", state.r_mm, 420.0),
        ("z_mm", state.z_mm, 380.0),
        ("torque shoulder", state.tau[1], 2.1),
        ("switch_right", state.switch_right, "UP"),
        ("switch_left", state.switch_left, "MID"),
        ("enabled", state.enabled, True),
        ("remote_online", state.remote_online, True),
        ("stalled", state.stalled, False),
    ]
    failed = 0
    for name, got, want in checks:
        ok = got == want
        failed += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name:22s} got={got!r:>10} want={want!r}")

    print()
    print(state.summary())
    print()

    # Boundary case: exactly len(WATCH_FIELDS) numeric fields + the tag IS valid.
    n = len(telemetry.WATCH_FIELDS)
    boundary = "watch," + ",".join(["0"] * n)
    if telemetry.parse_watch(boundary) is None:
        print(f"FAIL: parser rejected a valid {n + 1}-token line")
        failed += 1
    else:
        print(f"  OK   accepted boundary case: {n} fields + tag = {n + 1} tokens")

    # Malformed input must be rejected, never raise.
    bad = [
        ("empty string", ""),
        ("tag only", "watch"),
        ("few tokens", "watch,1,2,3"),
        ("one token short", "watch," + ",".join(["0"] * (n - 1))),
        ("one token long", "watch," + ",".join(["0"] * (n + 1))),
        ("non-numeric field", "watch," + ",".join(["0"] * (n - 1)) + ",x"),
        ("float instead of int", "watch," + ",".join(["0"] * (n - 1)) + ",1.5"),
        ("unrelated text", "garbage"),
    ]
    for label, b in bad:
        if telemetry.parse_watch(b) is not None:
            print(f"FAIL: parser accepted malformed input ({label}): {b[:60]!r}")
            failed += 1
    print(f"  OK   rejected {len(bad)} malformed inputs without raising")

    print()
    print("--- kinematics cross-check against firmware constants ---")
    th2, th3 = arm_model.motor_to_joint(-1.8, 1.2)
    r, z = arm_model.fk_r_z(th2, th3)
    print(f"  motor_to_joint(-1.8, 1.2) -> th2={th2:+.4f} th3={th3:+.4f}")
    print(f"  fk_r_z -> r={r:7.1f} mm  z={z:7.1f} mm")
    print(f"  telemetry said r={state.r_mm:.1f} z={state.z_mm:.1f} (different "
          f"angles, so values are not expected to match)")

    # Round-trip: IK must reproduce the FK input.
    th2b, th3b = arm_model.ik_r_z(r, z, elbow_up=(th3 >= 0))
    rt_r, rt_z = arm_model.fk_r_z(th2b, th3b)
    print(f"  IK round-trip: r {r:.3f} -> {rt_r:.3f}, z {z:.3f} -> {rt_z:.3f}")
    if abs(rt_r - r) > 1e-6 or abs(rt_z - z) > 1e-6:
        print("FAIL: IK/FK round-trip is not self-consistent")
        failed += 1
    else:
        print("  OK   IK/FK round-trip consistent")

    print()
    print("--- limit checker ---")
    for qb, qs, qe in [(0.0, -1.0, 1.0), (0.0, -1.0, 9.9), (9.9, 0.0, 0.0),
                       (0.0, -4.0, 0.5)]:
        v = arm_model.check_limits(qb, qs, qe)
        print(f"  ({qb:+.1f},{qs:+.1f},{qe:+.1f}) -> {v if v else 'clear'}")

    print()
    print(f"SELF-TEST {'PASSED' if failed == 0 else f'FAILED ({failed})'}")
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the arm UART7 telemetry link.")
    ap.add_argument("port", nargs="?", help="COM port, e.g. COM8")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--autodetect", action="store_true",
                    help="scan ports for arm telemetry and exit")
    ap.add_argument("--raw", action="store_true", help="also dump raw lines")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to watch")
    ap.add_argument("--hz", type=float, default=4.0, help="print rate")
    ap.add_argument("--self-test", action="store_true", help="offline parser test")
    args = ap.parse_args()

    if args.self_test:
        return cmd_self_test()

    if args.list:
        ports = telemetry.list_serial_ports()
        if not ports:
            print("no serial ports found")
        for dev, desc in ports:
            print(f"  {dev:8s} {desc}")
        return 0

    if args.autodetect:
        print(f"scanning for arm telemetry at {telemetry.BAUD} baud ...")
        found = telemetry.autodetect_port()
        print(f"  detected: {found}" if found else "  nothing found")
        return 0 if found else 1

    port = args.port
    if not port:
        print("no port given; trying autodetect ...")
        port = telemetry.autodetect_port()
        if not port:
            print("autodetect failed. Try --list, or pass the port explicitly.")
            return 1
        print(f"  detected: {port}")

    print(f"opening {port} @ {telemetry.BAUD} 8N1")
    try:
        reader = telemetry.TelemetryReader(port)
    except Exception as exc:
        print(f"FAILED to open {port}: {exc}")
        return 1

    period = 1.0 / max(args.hz, 0.1)
    deadline = time.monotonic() + args.seconds
    last_print = 0.0
    last_stats = reader.stats()
    last_time = time.monotonic()

    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_print >= period:
                last_print = now
                state = reader.latest()
                if state is None:
                    print("  ... no valid watch line yet")
                else:
                    print(state.summary())
                    violations = arm_model.check_limits(
                        state.q_base, state.q_shoulder, state.q_elbow
                    )
                    if violations:
                        for v in violations:
                            print(f"      LIMIT: {v}")
                if args.raw:
                    pass  # raw lines are surfaced through stats below
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stats = reader.stats()
        elapsed = time.monotonic() - last_time
        print()
        print("--- stats ---")
        for k, v in stats.items():
            print(f"  {k:16s} {v}")
        rate = (stats["watch_parsed"] - last_stats["watch_parsed"]) / max(elapsed, 1e-9)
        print(f"  watch rate       {rate:.1f} Hz  (firmware emits {telemetry.TELEMETRY_HZ} Hz)")
        print(f"  age of last line {reader.age_s():.3f} s")
        kin = reader.kin()
        if kin:
            print(f"  kin: calibrated={kin.is_calibrated} "
                  f"cart_mode={kin.cartesian_mode} th2={kin.th2_rad:+.4f} "
                  f"th3={kin.th3_rad:+.4f}")
        reader.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
