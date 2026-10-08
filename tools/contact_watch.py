"""Live, camera-based contact monitor for the yellow block.

Why this exists
---------------
The contact check in ``record_episode.py`` only runs AFTER an episode is saved,
so the operator gets no feedback while pushing.  Measured across four attempts:

    episode  shoulder_min   block moved   result
       0        -3.518         7.1 mm     CONTACT
       1        -2.522         0.8 mm     no contact
       2        -2.394         0.6 mm     no contact
       3        -2.279         0.3 mm     no contact

The gap between a hit and a miss was about 1 radian of shoulder travel - roughly
53 degrees, which cannot be judged from the transmitter.  Every attempt stopped
short and the operator had no way to know.

This tool closes that loop using the depth camera only:

    * segment the yellow block (colour), measure its distance (depth), project it
      into the camera frame using the colour intrinsics
    * hold that as the baseline
    * then report, several times a second, how far the block has moved in
      millimetres, and latch CONTACT the moment it passes the threshold

Nothing here is inferred from occlusion.  Occlusion only happens when the arm
passes between the camera and the block; the arm can touch the block from the far
side without occluding anything.  The only signal used is the one that is
actually true by definition: the block moved.

Run it, push with the transmitter, and stop when it says CONTACT.

    .venv\\Scripts\\python.exe tools\\contact_watch.py
    .venv\\Scripts\\python.exe tools\\contact_watch.py --threshold-mm 4 --baseline-s 2
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

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Live camera-based contact monitor.")
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--threshold-mm", type=float, default=5.0,
                    help="block displacement that counts as contact (noise floor is ~1 mm)")
    ap.add_argument("--baseline-s", type=float, default=1.5,
                    help="how long to average the block's resting position")
    ap.add_argument("--hz", type=float, default=2.0, help="readout rate")
    args = ap.parse_args()

    from arm_lerobot.block import BlockObservation, camera_intrinsics, measure_block
    from arm_lerobot.camera import CameraConfig, OrbbecCamera
    from arm_lerobot.telemetry import acquire_reader, release_reader

    reader = acquire_reader(args.port)
    cam = OrbbecCamera(CameraConfig(
        color_width=args.width, color_height=args.height, color_fps=30
    ))
    cam.open()
    intrinsics = camera_intrinsics(cam._color_profile)
    print(f"camera intrinsics (fx, fy, cx, cy) = {intrinsics}")
    print(f"telemetry on {args.port}: "
          f"{'streaming' if reader.latest() is not None else 'NO DATA YET'}")
    print()

    # ---------------------------------------------------------------- baseline
    print(f"measuring the block's resting position for {args.baseline_s:.1f}s "
          f"- keep the arm clear of it ...")
    deadline = time.monotonic() + args.baseline_s
    samples: list[tuple[float, float, float, int]] = []
    while time.monotonic() < deadline:
        frame = cam.read()
        if frame is None:
            continue
        obs = measure_block(frame.rgb, frame.depth, intrinsics)
        if obs.found and obs.xyz_mm is not None:
            samples.append((obs.xyz_mm[0], obs.xyz_mm[1], obs.xyz_mm[2], obs.area_px))
        time.sleep(0.02)

    if len(samples) < 5:
        print(f"FAILED: only {len(samples)} usable measurements. Is the yellow block "
              f"in view and unobstructed? Try tools/probe_depth_coverage.py.")
        cam.close()
        release_reader(args.port)
        return 1

    arr = np.array(samples)
    base_xyz = np.median(arr[:, :3], axis=0)
    base_area = int(np.median(arr[:, 3]))

    # The noise floor is NOT the +/-1 mm seen over five frames; over tens of
    # seconds it is several millimetres.  A 1 px wobble in the segmented centroid
    # projects to depth/fx = 1385/517 = 2.7 mm at this distance, and the blob edge
    # shifts by about that much frame to frame from antialiasing and video
    # compression.  Measured live with the arm completely still: up to 5.6 mm.
    #
    # So calibrate the threshold from the data instead of trusting a constant.
    residual = np.linalg.norm(arr[:, :3] - base_xyz, axis=1)
    noise_mm = float(np.median(residual))
    threshold = max(args.threshold_mm, 5.0 * noise_mm)

    print(f"baseline: xyz=({base_xyz[0]:+.0f}, {base_xyz[1]:+.0f}, {base_xyz[2]:+.0f}) mm  "
          f"area={base_area} px  ({len(samples)} samples)")
    print(f"measurement noise: median {noise_mm:.1f} mm over the baseline window")
    print(f"contact threshold: {threshold:.1f} mm "
          f"(max of --threshold-mm={args.threshold_mm:.0f} and 5x noise)")
    print()
    print("Push with the transmitter. Stop as soon as it says CONTACT.")
    print("If the block is hidden behind the arm it will say 'hidden' - keep going.")
    print()

    # -------------------------------------------------------------------- live
    from collections import deque

    period = 1.0 / max(args.hz, 0.1)
    latched = False
    best_mm = 0.0
    hidden = 0
    # ~1.5 s of history: the decision is taken on the median of these, so that
    # symmetric measurement jitter cannot masquerade as a sustained push.
    recent: deque[float] = deque(maxlen=max(3, int(args.hz * 1.5)))
    try:
        while True:
            t0 = time.monotonic()
            frame = cam.read()
            if frame is None:
                continue

            obs = measure_block(
                frame.rgb, frame.depth, intrinsics, baseline_area_px=base_area
            )
            state = reader.latest()
            sh = f"{state.q_shoulder:+.3f}" if state is not None else "  n/a "
            en = "ON " if (state is not None and state.enabled) else "off"

            if obs.occluded:
                hidden += 1
                print(f"  sh={sh}  {en}  block hidden behind the arm "
                      f"({hidden} consecutive) - keep pushing", flush=True)
                continue
            if not obs.found:
                print(f"  sh={sh}  {en}  block not visible ({obs.reason})", flush=True)
                continue
            hidden = 0

            if obs.xyz_mm is None:
                print(f"  sh={sh}  {en}  block at ({obs.u:.0f},{obs.v:.0f}) "
                      f"no depth - cannot measure", flush=True)
                continue

            d = float(np.linalg.norm(np.array(obs.xyz_mm) - base_xyz))
            best_mm = max(best_mm, d)

            # Decide on a ROLLING MEDIAN, never on the instantaneous value.
            # Jitter is symmetric, a real push is a sustained step, so the median
            # over ~1.5 s separates them where a single sample cannot: measured
            # live, instantaneous noise alone reached 5.6 mm with the arm still.
            recent.append(d)
            sustained = float(np.median(recent)) if len(recent) >= 3 else d

            if sustained >= threshold and not latched:
                latched = True
                print()
                print("  " + "=" * 62)
                print(f"  *** CONTACT ***  block moved {sustained:.1f} mm sustained "
                      f"(threshold {threshold:.1f} mm)")
                print(f"  shoulder is at {sh} - this is the pose to aim for")
                print("  " + "=" * 62)
                print()
            tag = "CONTACT" if latched else ("approaching" if sustained > 2.0 else "no contact yet")
            bar = "#" * min(30, int(sustained / max(threshold, 0.1) * 10))
            print(f"  sh={sh}  {en}  moved={d:5.1f} sustained={sustained:5.1f} mm  "
                  f"{bar:<30s} {tag}", flush=True)

            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print()
        print(f"stopped. largest block displacement seen: {best_mm:.1f} mm")
        print("CONTACT reached" if latched else "CONTACT was never reached")
    finally:
        cam.close()
        release_reader(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
