"""Probe the Orbbec Gemini 2 (phase 1a verification tool).

Purpose: prove the camera streams, measure its real frame rate, and find out
whether its hardware timestamps are usable.  All three matter downstream:

  * frame rate        decides the dataset fps we can honestly claim
  * timestamp source  decides whether camera and arm telemetry can be aligned
                      on a shared clock, or only by host arrival time
  * depth validity    decides whether depth is worth recording at all

Runs against the *system* Python 3.13, which already has pyorbbecsdk2 installed,
so it does not need to wait for the project venv.

    python tools/probe_camera.py --list
    python tools/probe_camera.py --profiles
    python tools/probe_camera.py --seconds 5 --save
    python tools/probe_camera.py --frames 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from arm_lerobot.camera import CameraConfig, OrbbecCamera  # noqa: E402


def cmd_list() -> int:
    devices = OrbbecCamera.list_devices()
    if not devices:
        print("no Orbbec device found")
        return 1
    print(f"{len(devices)} device(s):")
    for d in devices:
        vid = d["vid"]
        pid = d["pid"]
        vid_s = f"0x{vid:04X}" if isinstance(vid, int) else str(vid)
        pid_s = f"0x{pid:04X}" if isinstance(pid, int) else str(pid)
        print(f"  [{d['index']}] {d['name']}")
        print(f"       VID={vid_s} PID={pid_s} connection={d['connection']}")
        print(f"       serial={d['serial']}  uid={d['uid']}")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    cam = OrbbecCamera(CameraConfig(
        color_width=args.width, color_height=args.height, color_fps=args.fps,
        align_to_color=not args.no_align,
        enable_frame_sync=not args.no_sync,
    ))
    cam.open()
    try:
        profiles = cam.list_profiles()
        for kind, entries in profiles.items():
            print(f"--- {kind} ({len(entries)}) ---")
            for e in entries:
                print(f"    {e}")
        print()
        print(f"chosen color: {cam._color_profile}")
        print(f"chosen depth: {cam._depth_profile}")
    finally:
        cam.close()
    return 0


def _summarize_depth(depth: np.ndarray, is_uint16: bool) -> str:
    valid = depth > 0
    ratio = 100.0 * float(valid.mean()) if depth.size else 0.0
    if not valid.any():
        return f"shape={depth.shape} dtype={depth.dtype}  NO VALID PIXELS"
    vals = depth[valid]
    unit = "raw" if is_uint16 else "mm"
    return (
        f"shape={depth.shape} dtype={depth.dtype} valid={ratio:5.1f}% "
        f"min={float(vals.min()):8.1f}{unit} "
        f"p50={float(np.median(vals)):8.1f}{unit} "
        f"max={float(vals.max()):8.1f}{unit}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the Orbbec Gemini 2.")
    ap.add_argument("--list", action="store_true", help="list devices and exit")
    ap.add_argument("--profiles", action="store_true", help="dump stream profiles and exit")
    ap.add_argument("--seconds", type=float, default=0.0, help="capture duration")
    ap.add_argument("--frames", type=int, default=0, help="capture frame count")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-align", action="store_true", help="skip depth->color alignment")
    ap.add_argument("--no-sync", action="store_true", help="skip hardware frame sync")
    ap.add_argument("--raw-depth", action="store_true", help="keep uint16 depth, not mm")
    ap.add_argument("--save", action="store_true", help="write sample images to outputs/")
    args = ap.parse_args()

    if args.list:
        return cmd_list()
    if args.profiles:
        return cmd_profiles(args)

    seconds = args.seconds or (0.0 if args.frames else 5.0)
    target_frames = args.frames

    cfg = CameraConfig(
        color_width=args.width, color_height=args.height, color_fps=args.fps,
        align_to_color=not args.no_align,
        enable_frame_sync=not args.no_sync,
        warmup_frames=5,
        depth_as_uint16=args.raw_depth,
    )

    print(f"configuration: {cfg.describe()}")

    try:
        cam = OrbbecCamera(cfg)
        cam.open()
    except Exception as exc:
        print(f"FAILED to open camera: {exc}")
        return 1

    info = OrbbecCamera.list_devices()
    if info:
        d = info[0]
        print(f"device       : {d['name']}  serial={d['serial']}  "
              f"connection={d['connection']}")

    # Report what the device actually granted, which may differ from the request.
    cp, dp = cam._color_profile, cam._depth_profile
    if cp is not None:
        print(f"color profile: {cp}")
    if dp is not None:
        print(f"depth profile: {dp}")
    print(f"depth scale  : {cam._depth_scale}")
    print()

    deadline = time.monotonic() + seconds if seconds else None
    captured = 0
    first_frame = None
    depth_report_done = False
    interval_stats: list[float] = []
    last_t = None

    print("capturing ... (Ctrl-C to stop)")
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if target_frames and captured >= target_frames:
                break

            frame = cam.read(timeout_ms=1000)
            if frame is None:
                continue

            if first_frame is None:
                first_frame = frame
                print(f"first frame  : format={frame.color_format} "
                      f"frame_number={frame.frame_number}")
                print(f"timestamp API: {frame.timestamp_source or 'NONE AVAILABLE'}")
                if frame.color_timestamp_us is not None:
                    print(f"color ts     : {frame.color_timestamp_us} us")
                if frame.depth_timestamp_us is not None:
                    print(f"depth ts     : {frame.depth_timestamp_us} us")
                if frame.rgb is not None:
                    print(f"rgb          : shape={frame.rgb.shape} "
                          f"dtype={frame.rgb.dtype} "
                          f"mean={frame.rgb.mean():.1f}")
                print()

            if frame.depth is not None and not depth_report_done and captured > 5:
                print(f"depth        : {_summarize_depth(frame.depth, args.raw_depth)}")
                depth_report_done = True

            now = time.monotonic()
            if last_t is not None:
                interval_stats.append(now - last_t)
            last_t = now
            captured += 1

            if captured % 30 == 0:
                print(f"  {captured:5d} frames  host_fps={cam.stats.fps:5.1f}")
    except KeyboardInterrupt:
        print("\ninterrupted")

    print()
    print("--- results ---")
    st = cam.stats
    print(f"frames captured   : {st.frames}")
    print(f"timeouts          : {st.timeouts}")
    print(f"dropped (bad size): {st.dropped}")
    print(f"elapsed           : {st.elapsed_s:.2f} s")
    print(f"host FPS          : {st.fps:.2f}")
    ts_fps = st.timestamp_fps()
    if ts_fps is None:
        print("device-timestamp FPS: unavailable (no usable hardware timestamps)")
    else:
        print(f"device-timestamp FPS: {ts_fps:.2f}")
        if st.fps > 0:
            drift = 100.0 * (ts_fps - st.fps) / st.fps
            print(f"  host vs device drift: {drift:+.2f}%")
            if abs(drift) > 5.0:
                print("  WARNING: device clock and host clock disagree by >5%. "
                      "Alignment must interpolate on the host clock, not trust "
                      "device timestamps directly.")
    if interval_stats:
        arr = np.array(interval_stats)
        print(f"inter-frame time  : mean={arr.mean()*1000:.1f}ms "
              f"p50={np.median(arr)*1000:.1f}ms "
              f"p95={np.percentile(arr, 95)*1000:.1f}ms "
              f"max={arr.max()*1000:.1f}ms")
        print(f"  jitter (p95-p50): {(np.percentile(arr,95)-np.median(arr))*1000:.1f}ms")

    if args.save and first_frame is not None:
        import cv2

        outdir = REPO / "outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        if first_frame.rgb is not None:
            p = outdir / f"camera_rgb_{stamp}.png"
            cv2.imwrite(str(p), cv2.cvtColor(first_frame.rgb, cv2.COLOR_RGB2BGR))
            print(f"saved {p}")

        if first_frame.depth is not None:
            depth = first_frame.depth.astype(np.float32)
            valid = depth > 0
            norm = np.zeros_like(depth, dtype=np.uint8)
            if valid.any():
                lo, hi = float(depth[valid].min()), float(depth[valid].max())
                if hi > lo:
                    norm = np.clip((depth - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                norm[~valid] = 0
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            p = outdir / f"camera_depth_{stamp}.png"
            cv2.imwrite(str(p), colored)
            print(f"saved {p}")
            p = outdir / f"camera_depth_raw_{stamp}.npy"
            np.save(p, first_frame.depth)
            print(f"saved {p} (raw, for exact value checks)")

    cam.close()

    print()
    if st.fps > 0:
        print(f"VERDICT: camera streams at {st.fps:.1f} FPS "
              f"({cfg.color_width}x{cfg.color_height}) with "
              f"{'usable' if ts_fps else 'NO'} hardware timestamps.")
    return 0 if st.frames > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
