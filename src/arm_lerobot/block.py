"""Locate the yellow block and convert it to camera-frame millimetres.

Why colour first, then depth
----------------------------
The block is a saturated yellow cube.  Colour segmentation is robust and cheap;
stereo depth is what fails on flat, low-texture surfaces.  Using colour to find
the block and depth only to measure the distance at that location plays to each
sensor's strength, and it also survives the case where the block's own surface is
poorly matched by stereo.

Success criterion without hand-eye calibration
----------------------------------------------
We never calibrate camera-to-vehicle.  Instead the block's pixel centroid plus its
depth is projected into the camera frame with the colour intrinsics, giving a
metric position.  Displacement between the start and the end of an episode is then
a plain Euclidean distance in millimetres, which does not care how the camera is
aimed.

Measured on this setup (640x480, block at rest):
    bbox 74 x 76 px, 99.4% of its pixels have valid depth,
    distance 1507 mm with only a 130 mm spread across the blob.

Occlusion is the real hazard: the arm passes in front of the block while pushing.
A shrunken blob is therefore reported as ``occluded`` rather than as a small
block, so the caller can refuse to judge success instead of silently measuring
the arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Saturated yellow.  A loose threshold also catches beige floor tiles and tan
# furniture - an earlier version reported a 598 px wide "yellow" region spanning
# 1.4 m to 8 m of depth, which cannot be one block.
HUE_LO, HUE_HI = 20, 35
SAT_MIN, VAL_MIN = 120, 120

DEFAULT_MIN_AREA_PX = 200


@dataclass(frozen=True)
class BlockObservation:
    """Where the block is, or why we cannot say."""

    found: bool
    reason: str
    u: float = float("nan")
    v: float = float("nan")
    area_px: int = 0
    depth_mm: float | None = None
    valid_ratio: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    xyz_mm: tuple[float, float, float] | None = None
    occluded: bool = False

    def describe(self) -> str:
        if not self.found:
            return f"block NOT found ({self.reason})"
        xyz = "" if self.xyz_mm is None else (
            f"  cam xyz=({self.xyz_mm[0]:+.0f}, {self.xyz_mm[1]:+.0f}, "
            f"{self.xyz_mm[2]:+.0f}) mm"
        )
        return (
            f"block at ({self.u:.0f}, {self.v:.0f}) px  area={self.area_px} px  "
            f"depth={self.depth_mm:.0f} mm  valid={100 * self.valid_ratio:.0f}%"
            f"{xyz}"
        )


def camera_intrinsics(profile) -> tuple[float, float, float, float] | None:
    """(fx, fy, cx, cy) from a stream profile, or None if unavailable."""
    try:
        intr = profile.get_intrinsic()
    except Exception:
        return None
    fx = getattr(intr, "fx", None)
    fy = getattr(intr, "fy", None)
    cx = getattr(intr, "cx", None)
    cy = getattr(intr, "cy", None)
    if None in (fx, fy, cx, cy):
        return None
    return float(fx), float(fy), float(cx), float(cy)


def project_to_camera_frame(
    u: float, v: float, depth_mm: float, intrinsics: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    """Pixel + depth -> millimetres in the camera frame (Z forward)."""
    fx, fy, cx, cy = intrinsics
    z = depth_mm
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def measure_block(
    rgb: np.ndarray,
    depth_mm: np.ndarray | None = None,
    intrinsics: tuple[float, float, float, float] | None = None,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    baseline_area_px: int | None = None,
    occlusion_ratio: float = 0.4,
    min_depth_mm: float = 150.0,
    max_depth_mm: float = 6000.0,
) -> BlockObservation:
    """Find the largest saturated-yellow blob and measure it.

    Args:
        rgb: (H, W, 3) uint8, RGB order.
        depth_mm: (H, W) float32 millimetres, 0 = invalid.  May be None, in which
            case geometry is not attempted.
        intrinsics: from :func:`camera_intrinsics`.
        baseline_area_px: blob area measured when the block was definitely visible.
            If given and the current area falls below ``occlusion_ratio`` of it,
            the result is flagged ``occluded`` instead of being reported as a
            moved block.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        return BlockObservation(False, f"cv2 unavailable: {exc}")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([HUE_LO, SAT_MIN, VAL_MIN]),
        np.array([HUE_HI, 255, 255]),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return BlockObservation(False, "no yellow pixels")

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < min_area_px:
        return BlockObservation(False, f"largest yellow blob only {area} px")

    x, y, bw, bh = (int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP]),
                    int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT]))
    blob = labels == idx

    if baseline_area_px and area < occlusion_ratio * baseline_area_px:
        return BlockObservation(
            False, f"area {area} px is {area / baseline_area_px:.0%} of baseline "
                   f"({baseline_area_px} px) - the arm is probably in front of it",
            area_px=area, bbox=(x, y, bw, bh), occluded=True,
        )

    # Centroid from the mask itself, not from the bounding box.
    ys, xs = np.nonzero(blob)
    u, v = float(xs.mean()), float(ys.mean())

    depth_mm_out = None
    valid_ratio = 0.0
    xyz = None
    if depth_mm is not None:
        vals = depth_mm[blob]
        ok = (vals >= min_depth_mm) & (vals <= max_depth_mm)
        valid_ratio = float(ok.mean()) if ok.size else 0.0
        if ok.any():
            # Median, not mean: depth edges bleed and a mean would be dragged.
            depth_mm_out = float(np.median(vals[ok]))
            if intrinsics is not None:
                xyz = project_to_camera_frame(u, v, depth_mm_out, intrinsics)

    return BlockObservation(
        found=True,
        reason="ok",
        u=u, v=v, area_px=area,
        depth_mm=depth_mm_out,
        valid_ratio=valid_ratio,
        bbox=(x, y, bw, bh),
        xyz_mm=xyz,
    )


def displacement_mm(a: BlockObservation, b: BlockObservation) -> float | None:
    """Euclidean distance the block moved, in mm, or None if not measurable."""
    if not a.found or not b.found or a.xyz_mm is None or b.xyz_mm is None:
        return None
    return float(np.linalg.norm(np.array(b.xyz_mm) - np.array(a.xyz_mm)))


def pushed_forward(before: BlockObservation, after: BlockObservation, min_mm: float) -> tuple[bool | None, str]:
    """Did the block move at least ``min_mm``?

    Returns (verdict, explanation).  ``None`` means "cannot tell", which is a
    different answer from "no" and must stay distinct: an occluded block is not a
    failed push.
    """
    if before.occluded or after.occluded:
        return None, "block occluded by the arm - cannot judge"
    if not before.found:
        return None, f"no baseline measurement ({before.reason})"
    if not after.found:
        return None, f"no final measurement ({after.reason})"
    d = displacement_mm(before, after)
    if d is None:
        return None, "depth unavailable at one end, cannot measure displacement"
    if d >= min_mm:
        return True, f"block moved {d:.0f} mm (>= {min_mm:.0f} mm)"
    return False, f"block moved only {d:.0f} mm (< {min_mm:.0f} mm)"
