"""Per-episode trajectory analysis for an ``arm27_push``-style dataset.

``inspect_dataset.py`` answers "is this dataset well formed".  This answers a
different and more urgent question: **what does each episode actually show, and
does the set of them agree with each other enough for a policy to learn
anything?**

It reads the parquet files directly (no video decode), so it is fast, and prints
one row per episode plus the cross-episode spread.  The columns are chosen around
the failure this project actually hit - a 1 radian shortfall in the shoulder that
the operator cannot see from the transmitter - plus the firmware interlock that
lives at exactly that shoulder angle:

    SHOULDER_INTERLOCK_BELOW = -3.5  ->  elbow forced into [1.8, 3.1]

because if the demonstrate-able elbow range collapses to a narrow band at the
moment of contact, a policy cannot learn a smooth approach from these episodes.

Run::

    .venv\\Scripts\\python.exe tools\\analyze_episodes.py
    .venv\\Scripts\\python.exe tools\\analyze_episodes.py --root data/arm27_push
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

JOINTS = ("q_base", "q_shoulder", "q_elbow")


def column(df: pd.DataFrame, name: str) -> np.ndarray:
    """LeRobot parquet stores fixed-size vectors as object columns of arrays."""
    col = df[name].to_numpy()
    return np.stack([np.asarray(v, dtype=np.float64).ravel() for v in col])


def summarize(path: Path, fps: float) -> dict:
    df = pd.read_parquet(path)
    state = column(df, "observation.state")
    action = column(df, "action")
    n = len(df)

    sh = state[:, 1]
    i_min = int(np.argmin(sh))

    # Where the shoulder passes the interlock threshold, if it ever does.
    below = np.nonzero(sh < -3.5)[0]
    if below.size:
        el = state[below, 2]
        interlock_band = (float(el.min()), float(el.max()))
        interlock_frames = int(below.size)
    else:
        interlock_band = (float("nan"), float("nan"))
        interlock_frames = 0

    return {
        "file": path.name,
        "frames": n,
        "seconds": n / fps,
        "sh_min": float(sh.min()),
        "sh_min_at_pct": 100.0 * i_min / max(n - 1, 1),
        "sh_max": float(sh.max()),
        "el_at_sh_min": float(state[i_min, 2]),
        "el_min": float(state[:, 2].min()),
        "el_max": float(state[:, 2].max()),
        "sh_travel": float(np.ptp(sh)),
        "el_travel": float(np.ptp(state[:, 2])),
        "base_travel": float(np.ptp(state[:, 0])),
        "interlock_frames": interlock_frames,
        "interlock_elbow": interlock_band,
        "track_err": float(np.abs(action - state).max()),
        "state": state,
        "action": action,
    }


def measure_block_in_video(path: Path, depth_mm: float, fx: float) -> dict:
    """Track the yellow block's pixel centroid through one episode video.

    The dataset deliberately carries no depth (see README design note 3), so the
    displacement is measured in pixels and converted with the known working
    distance and fx.  That conversion is exact for motion perpendicular to the
    optical axis and an underestimate for motion along it, which is the safe
    direction: it can miss a marginal push, it cannot invent one.

    Centroid precision is not the limit - a ~3900 px blob localises to well under
    0.1 px.  The limit is whether the whole scene is stable, so the reported
    noise floor is the spread *before* the arm starts moving.
    """
    import av
    from arm_lerobot.block import measure_block

    mm_per_px = depth_mm / fx
    centroids: list[tuple[float, float]] = []
    areas: list[int] = []
    missing = 0

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            obs = measure_block(rgb)  # no depth: segmentation only
            if obs.found:
                centroids.append((obs.u, obs.v))
                areas.append(obs.area_px)
            else:
                centroids.append((float("nan"), float("nan")))
                areas.append(0)
                missing += 1

    uv = np.array(centroids)
    ar = np.array(areas)
    n = len(uv)
    ok = ~np.isnan(uv[:, 0])

    # Baseline: the largest stretch of consecutive good frames at the start.
    head = min(25, n)
    good_head = ok[:head]
    if good_head.sum() < 5:
        return {"n": n, "error": "block not visible at the start"}
    base = np.median(uv[:head][good_head], axis=0)
    base_area = int(np.median(ar[:head][good_head]))

    # Noise floor while the block is still at rest: use the first 10% of frames,
    # before the operator can plausibly have reached it.
    rest = int(max(10, 0.10 * n))
    rest_ok = ok[:rest]
    rest_res = np.linalg.norm(uv[:rest][rest_ok] - base, axis=1)
    noise_px = float(np.median(rest_res)) if rest_ok.sum() else float("nan")

    dist = np.full(n, np.nan)
    dist[ok] = np.linalg.norm(uv[ok] - base, axis=1)
    i_max = int(np.nanargmax(dist)) if ok.any() else -1

    return {
        "n": n,
        "missing": missing,
        "base_area": base_area,
        "noise_px": noise_px,
        "max_px": float(dist[i_max]) if i_max >= 0 else float("nan"),
        "max_mm": float(dist[i_max]) * mm_per_px if i_max >= 0 else float("nan"),
        "max_at": i_max,
        "max_at_pct": 100.0 * i_max / max(n - 1, 1) if i_max >= 0 else float("nan"),
        "min_area_after_rest": int(ar[rest:].min()) if n > rest else base_area,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-episode trajectory analysis.")
    ap.add_argument("--root", default=str(REPO / "data" / "arm27_push"))
    ap.add_argument("--measure-block", action="store_true",
                    help="also decode the videos and measure how far the yellow "
                         "block moved (the only definitionally true success test)")
    ap.add_argument("--block-depth-mm", type=float, default=1500.0,
                    help="assumed block distance, to convert px to mm; the dataset "
                         "carries no depth by design")
    args = ap.parse_args()

    import json

    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        print(f"no parquet under {root}")
        return 1

    print(f"dataset : {root}")
    print(f"fps     : {fps:g}   episodes: {len(files)}   "
          f"frames: {sum(pd.read_parquet(f, columns=['index']).shape[0] for f in files)}")
    print()

    rows = [summarize(f, fps) for f in files]

    hdr = (f"{'ep':>3} {'frames':>6} {'sec':>5} {'sh_min':>8} {'@%':>5} {'el@min':>8} "
           f"{'sh_trav':>8} {'el_trav':>8} {'base_tv':>8} {'|a-s|':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        print(f"{i:>3} {r['frames']:>6} {r['seconds']:>5.1f} {r['sh_min']:>+8.3f} "
              f"{r['sh_min_at_pct']:>5.0f} {r['el_at_sh_min']:>+8.3f} "
              f"{r['sh_travel']:>8.3f} {r['el_travel']:>8.3f} {r['base_travel']:>8.3f} "
              f"{r['track_err']:>8.4f}")
    print()

    # ------------------------------------------------------- interlock screen
    print(f"interlock (firmware: shoulder < -3.5 forces elbow into "
          f"[{1.8}, {3.1}]):")
    for i, r in enumerate(rows):
        if r["interlock_frames"] == 0:
            print(f"  ep{i}: shoulder never went below -3.5 "
                  f"(deepest {r['sh_min']:+.3f}, {r['sh_min'] - (-3.5):+.3f} short)")
        else:
            lo, hi = r["interlock_elbow"]
            print(f"  ep{i}: {r['interlock_frames']:>4} frames below -3.5, "
                  f"elbow there spans [{lo:+.3f}, {hi:+.3f}]")
    print()

    # --------------------------------------------------- the actual success test
    if args.measure_block:
        cam_key = next((k for k in info["features"] if k.startswith("observation.images.")), None)
        if cam_key is None:
            print("no image feature - cannot measure the block")
            return 0
        intr = info["features"][cam_key].get("info", {})
        fx = 517.19  # measured on this camera, README
        print(f"block displacement, measured from the videos "
              f"({args.block_depth_mm:.0f} mm assumed distance, fx={fx}, "
              f"{args.block_depth_mm / fx:.2f} mm/px):")
        bs = []
        for i, f in enumerate(files):
            vid = root / "videos" / cam_key / "chunk-000" / f"{f.stem}.mp4"
            if not vid.exists():
                print(f"  ep{i}: missing {vid}")
                bs.append(None)
                continue
            b = measure_block_in_video(vid, args.block_depth_mm, fx)
            bs.append(b)
            if "error" in b:
                print(f"  ep{i}: {b['error']}")
                continue
            verdict = "CONTACT" if b["max_mm"] >= 5.0 else "no contact"
            print(f"  ep{i}: {b['n']:>4} frames  baseline area {b['base_area']:>5} px  "
                  f"rest noise {b['noise_px']:.2f} px  "
                  f"max move {b['max_px']:5.2f} px = {b['max_mm']:5.1f} mm "
                  f"at {b['max_at_pct']:3.0f}%  min area {b['min_area_after_rest']:>5} px  "
                  f"-> {verdict}")
        print()
        print("  (rest noise is the block's own jitter before the arm moves; a move "
              "smaller than ~5x it is not evidence)")
        print()

    # ------------------------------------------------------ cross-episode spread
    print("cross-episode consistency (this is what a policy has to average over):")
    for j, name in enumerate(JOINTS):
        starts = np.array([r["state"][0, j] for r in rows])
        ends = np.array([r["state"][-1, j] for r in rows])
        mins = np.array([r["state"][:, j].min() for r in rows])
        print(f"  {name:11s} start spread {np.ptp(starts):.3f} rad "
              f"({np.degrees(np.ptp(starts)):5.1f} deg)   "
              f"end spread {np.ptp(ends):.3f}   min spread {np.ptp(mins):.3f}")
    print()

    # --------------------------------------------------- same-state/action check
    print("does the same observation imply the same action? (the core of imitation)")
    # Bin the shoulder angle and look at the spread of the commanded shoulder
    # within each bin, across all episodes.  A small spread means the dataset is
    # self-consistent; a large one means the episodes disagree about what to do.
    all_s = np.concatenate([r["state"] for r in rows])
    all_a = np.concatenate([r["action"] for r in rows])
    sh_bins = np.arange(-5.5, 0.75, 0.25)
    for j, name in enumerate(JOINTS):
        spreads = []
        for lo, hi in zip(sh_bins[:-1], sh_bins[1:]):
            m = (all_s[:, 1] >= lo) & (all_s[:, 1] < hi)
            if m.sum() >= 20:
                spreads.append(np.ptp(all_a[m, j]))
        if spreads:
            print(f"  {name:11s} action spread within a 0.25 rad shoulder bin: "
                  f"median {np.median(spreads):.3f}  max {np.max(spreads):.3f} rad")
        else:
            print(f"  {name:11s} not enough samples per bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
