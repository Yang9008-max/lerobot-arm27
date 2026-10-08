"""Zoomed, side-by-side view of the block at three moments of every episode.

Why this exists
---------------
Two measurements of the same episodes disagreed:

  * the live depth measurement in ``record_episode.py`` said episodes 1-3 barely
    moved the block (0.8 / 0.6 / 0.3 mm)
  * an offline colour-blob measurement said every episode moved it by 15-107 mm

The disagreement was resolved by looking at the pixels: the colour blob merges
with a passer-by or with a shadowed patch of floor, so its centroid slides even
though the block does not.  Colour segmentation alone cannot settle this.

What this tool does instead is show the operator the truth: for each episode it
picks three frames (start, the moment the shoulder is deepest - which is when
contact would have happened - and the end) and crops a fixed window around the
block, magnified, with a reference line drawn at the block's starting column.

If the block moved, its left edge visibly crosses that line.  If it did not, the
three tiles are identical.  No threshold, no verdict - just the evidence.

Run::

    .venv\\Scripts\\python.exe tools\\compare_block_crops.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "arm27_push"))
    ap.add_argument("--camera-key", default="observation.images.front")
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--u0", type=int, default=10, help="crop window left")
    ap.add_argument("--u1", type=int, default=200, help="crop window right")
    ap.add_argument("--v0", type=int, default=100)
    ap.add_argument("--v1", type=int, default=210)
    args = ap.parse_args()

    import av
    import cv2
    import pandas as pd

    from arm_lerobot.arm_model import SHOULDER_INTERLOCK_BELOW

    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])

    parquets = sorted((root / "data").rglob("*.parquet"))
    rows = []
    for f in parquets:
        df = pd.read_parquet(f, columns=["observation.state"])
        st = np.stack([np.asarray(v, dtype=float).ravel() for v in df["observation.state"].to_numpy()])
        sh = st[:, 1]
        rows.append((f.stem, int(np.argmin(sh)), float(sh.min()), st[:, 2][int(np.argmin(sh))]))

    print(f"{'ep':>4} {'sh_min':>8} {'el@min':>8} {'interlock?':>11}")
    for name, i_min, sh_min, el in rows:
        il = "ENGAGED" if sh_min < SHOULDER_INTERLOCK_BELOW else ""
        print(f"{name:>4} {sh_min:>+8.3f} {el:>+8.3f} {il:>11}")
    print()

    all_rows = []
    for name, i_min, sh_min, el in rows:
        vid = root / "videos" / args.camera_key / "chunk-000" / f"{name}.mp4"
        if not vid.exists():
            print(f"missing {vid}")
            continue
        frames = []
        with av.open(str(vid)) as container:
            for fr in container.decode(video=0):
                frames.append(fr.to_ndarray(format="rgb24"))
        n = len(frames)
        picks = [0, i_min, n - 1]

        tiles = []
        # Reference column = the block's left edge at frame 0.
        for k, i in enumerate(picks):
            img = frames[i][args.v0:args.v1, args.u0:args.u1].copy()
            # Column of the leftmost strongly-yellow pixel in THIS tile's frame 0,
            # drawn on every tile of the episode so drift is visible.
            if k == 0:
                hsv = cv2.cvtColor(frames[0], cv2.COLOR_RGB2HSV)
                m = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([35, 255, 255]))
                ys, xs = np.nonzero(m)
                ref_u = int(xs.min()) if xs.size else args.u0
                ref_u = max(args.u0, min(ref_u, args.u1 - 1))
            vis = img
            for uu in (ref_u,):
                x = uu - args.u0
                cv2.line(vis, (x, 0), (x, vis.shape[0] - 1), (255, 0, 255), 1)
            vis = cv2.resize(vis, None, fx=args.zoom, fy=args.zoom,
                             interpolation=cv2.INTER_NEAREST)
            label = {0: f"{name} f0 (0%)",
                     1: f"f{i_min} sh={sh_min:+.3f} DEEPEST",
                     2: f"f{n - 1} (100%)"}[k]
            cv2.putText(vis, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        all_rows.append(np.hstack(tiles))

    sheet = np.vstack(all_rows)
    outdir = REPO / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "block_movement_all_episodes.png"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("magenta line = the block's left edge at frame 0 of that episode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
