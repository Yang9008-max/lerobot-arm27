"""Map which objects in view actually have usable depth.

Stereo depth (Gemini 2) fails on smooth, textureless, or reflective surfaces, so
"the camera streams depth" does not imply "every object in front of the vehicle
has depth".  This tool answers the practical question per object:

  * overall valid pixel ratio, and the ratio inside the central region
  * connected blobs of INVALID depth, each with its bounding box, pixel area and
    MEAN RGB - the colour is what identifies which object failed
  * depth validity and distance inside any yellow region (HSV threshold), because
    the task is to push a yellow block

It writes two images so a human can see the answer:

  outputs/depth_covered_<stamp>.png   RGB with invalid-depth pixels tinted red
  outputs/depth_color_<stamp>.png     the depth map colourised, invalid = black

Run::

    .venv\\Scripts\\python.exe tools\\probe_depth_coverage.py
    .venv\\Scripts\\python.exe tools\\probe_depth_coverage.py --seconds 3 --ymin-mm 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402


def analyse(depth_mm: np.ndarray, rgb: np.ndarray, min_mm: float, max_mm: float):
    """Return the invalid mask and a summary, using cv2 for blob analysis."""
    import cv2

    valid = (depth_mm >= min_mm) & (depth_mm <= max_mm)
    h, w = valid.shape
    invalid = ~valid

    # A few speckled holes are normal; what matters is a whole object missing.
    # Open with a small kernel so single-pixel dropouts are not reported as blobs.
    mask = invalid.astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.002 * h * w:  # ignore anything under 0.2% of the frame
            continue
        region = labels[y:y + bh, x:x + bw] == i
        mean_rgb = rgb[y:y + bh, x:x + bw][region].mean(axis=0)
        blobs.append({
            "area_px": int(area),
            "area_pct": 100.0 * area / (h * w),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "mean_rgb": tuple(round(float(v), 1) for v in mean_rgb),
        })
    blobs.sort(key=lambda b: -b["area_px"])

    # Central region: where the arm and the block actually are.
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    center = valid[cy0:cy1, cx0:cx1]

    return valid, invalid, mask, blobs, float(center.mean())


def yellow_region(rgb: np.ndarray):
    """Largest saturated-yellow blob, or an all-False mask.

    A loose threshold also catches the beige floor tiles and the tan furniture:
    an earlier run reported a "yellow" bounding box 598 px wide spanning 1.4 m to
    8 m of depth, which cannot be one block.  Requiring real saturation and then
    keeping only the largest connected component makes the measurement mean
    something.
    """
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # OpenCV hue is 0..179; saturated yellow sits around 20-35.
    mask = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([35, 255, 255]))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros(rgb.shape[:2], dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def describe_colour(rgb_mean: tuple[float, ...]) -> str:
    r, g, b = rgb_mean
    mx, mn = max(rgb_mean), min(rgb_mean)
    if mx < 60:
        return "very dark / black"
    if mn > 170 and mx - mn < 30:
        return "bright and near-white (likely blown out)"
    if mx - mn < 25:
        return "flat grey (low texture)"
    if r > g > b and r - b > 60:
        return "warm / yellowish-red"
    if b > r and b - r > 25:
        return "bluish"
    if g >= r and g >= b:
        return "greenish"
    return "saturated colour (low texture for stereo)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Which objects have usable depth?")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--min-mm", type=float, default=200.0)
    ap.add_argument("--max-mm", type=float, default=8000.0)
    ap.add_argument("--save", action="store_true", default=True)
    args = ap.parse_args()

    from arm_lerobot.camera import CameraConfig, OrbbecCamera

    cfg = CameraConfig(color_width=args.width, color_height=args.height, color_fps=30)
    cam = OrbbecCamera(cfg)
    cam.open()

    print(f"capturing {args.seconds:.1f}s, keeping the frame with the most valid depth ...")
    best = None
    best_valid = -1.0
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        frame = cam.read()
        if frame is None or frame.depth is None or frame.rgb is None:
            continue
        v = float(((frame.depth >= args.min_mm) & (frame.depth <= args.max_mm)).mean())
        if v > best_valid:
            best_valid = v
            best = frame
        time.sleep(0.05)
    cam.close()

    if best is None:
        print("FAILED: no frame with depth captured")
        return 1

    depth = best.depth
    rgb = best.rgb
    print(f"kept the frame with {100 * best_valid:.1f}% valid depth")
    print()

    valid, invalid, mask, blobs, center_valid = analyse(
        depth, rgb, args.min_mm, args.max_mm
    )
    h, w = valid.shape

    print(f"image              : {w}x{h}")
    print(f"valid depth        : {100 * valid.mean():5.1f}% overall, "
          f"{100 * center_valid:5.1f}% in the central region")
    vals = depth[valid]
    if vals.size:
        print(f"depth range        : min={vals.min():.0f} p25={np.percentile(vals,25):.0f} "
              f"p50={np.median(vals):.0f} p75={np.percentile(vals,75):.0f} "
              f"max={vals.max():.0f} mm")
    print()

    # ---- yellow block ----
    ymask = yellow_region(rgb)
    nyellow = int(ymask.sum())
    print(f"yellow-ish pixels  : {nyellow} ({100.0 * nyellow / (h * w):.2f}% of frame)")
    if nyellow >= 50:
        yvalid = valid[ymask]
        ydepth = depth[ymask]
        yv = ydepth[yvalid]
        print(f"  depth at yellow  : {100.0 * yvalid.mean():5.1f}% valid")
        if yv.size:
            print(f"  distance         : p50={np.median(yv):.0f} mm "
                  f"(range {yv.min():.0f}..{yv.max():.0f})")
            print(f"  -> the block is {'MEASURABLE' if yvalid.mean() > 0.5 else 'POORLY MEASURABLE'}"
                  f" by depth alone")
        ys = np.argwhere(ymask)
        if ys.size:
            y0, x0 = ys.min(axis=0)
            y1, x1 = ys.max(axis=0)
            print(f"  bbox             : x {x0}..{x1} ({x1 - x0 + 1} px)  "
                  f"y {y0}..{y1} ({y1 - y0 + 1} px)")
    else:
        print("  nothing yellow in view")
    print()

    # ---- the objects with NO usable depth ----
    print(f"invalid-depth blobs bigger than 0.2% of the frame: {len(blobs)}")
    if not blobs:
        print("  none - every object in view has depth")
    for i, b in enumerate(blobs[:8], 1):
        x, y, bw, bh = b["bbox"]
        print(f"  {i}. {b['area_pct']:5.2f}% of frame  ({b['area_px']:6d} px)  "
              f"bbox x{x} y{y} {bw}x{bh}")
        print(f"     mean RGB {b['mean_rgb']}  ->  {describe_colour(b['mean_rgb'])}")

    if args.save:
        try:
            import cv2
        except ImportError:
            print("\n(cv2 not available, skipping images)")
            return 0
        outdir = REPO / "outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        overlay = rgb.copy()
        # Grow the invalid mask so thin failures stay visible after scaling.
        grown = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        overlay[grown] = (0.35 * overlay[grown] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
        p1 = outdir / f"depth_covered_{stamp}.png"
        cv2.imwrite(str(p1), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        norm = np.zeros_like(depth, dtype=np.uint8)
        if vals.size:
            lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
            if hi > lo:
                norm = np.clip((depth - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            norm[~valid] = 0
        p2 = outdir / f"depth_color_{stamp}.png"
        cv2.imwrite(str(p2), cv2.applyColorMap(norm, cv2.COLORMAP_JET))

        print()
        print(f"RED = no depth   -> {p1}")
        print(f"depth colormap   -> {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
