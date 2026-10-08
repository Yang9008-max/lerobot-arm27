"""Visual confirmation that one episode's block really moved, tag included.

``track_block_tag.py`` reports numbers; this shows the same evidence as pixels so
the two can be checked against each other.  For one episode it prints and draws

    frame 0        the reference
    deepest        the frame where the shoulder is lowest (where contact happens)
    final          the last frame

marking the R-tag position found by the same template match in every tile, and
adds a zoom of the tag at the start and at the end so a translation is obvious.

Run::

    .venv\\Scripts\\python.exe tools/verify_block_move.py --episode 4
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--episode", type=int, default=4)
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--half", type=int, default=110, help="zoom half-window, px")
    args = ap.parse_args()

    import av
    import cv2
    import pandas as pd
    from arm_lerobot.block import measure_block

    root = Path(args.root)
    pq = root / "data" / "chunk-000" / f"file-{args.episode:03d}.parquet"
    st = np.stack([np.asarray(v, dtype=float).ravel()
                   for v in pd.read_parquet(pq, columns=["observation.state"])
                   ["observation.state"].to_numpy()])
    sh = st[:, 1]
    i_min = int(np.argmin(sh))

    vid = root / "videos" / args.camera_key / "chunk-000" / f"file-{args.episode:03d}.mp4"
    frames = []
    with av.open(str(vid)) as container:
        for fr in container.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    n = min(len(frames), len(sh))

    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    obs = measure_block(frames[0])
    x, y, w, h = obs.bbox
    ix, iy = int(x + 0.25 * w), int(y + 0.25 * h)
    iw, ih = int(0.5 * w), int(0.5 * h)
    templ0 = gray0[iy:iy + ih, ix:ix + iw]

    def find(i):
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(g, templ0, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        return ml[0] + iw / 2, ml[1] + ih / 2, mx

    picks = [0, i_min, n - 1]
    print(f"episode {args.episode}: n={n} frames, shoulder deepest at f{i_min} "
          f"({100 * i_min / (n - 1):.0f}%) sh={sh[i_min]:+.3f}")
    print(f"tag template from bbox=({x},{y},{w},{h}) inset -> {ix},{iy} {iw}x{ih}")
    print()
    pos = {}
    for i in picks:
        cx, cy, sc = find(i)
        pos[i] = (cx, cy, sc)
        d = np.hypot(cx - pos[0][0], cy - pos[0][1])
        print(f"  f{i:>4} ({100 * i / (n - 1):3.0f}%) sh={sh[i]:+.3f} "
              f"tag=({cx:6.1f},{cy:6.1f}) score={sc:.3f} "
              f"moved {d:5.2f} px = {d * 2.90:5.1f} mm")

    # ---- tiles ----
    tiles = []
    for i in picks:
        vis = frames[i].copy()
        cx, cy, sc = pos[i]
        for j in picks:
            px, py, _ = pos[j]
            colour = (255, 0, 0) if j == 0 else (0, 255, 0)
            cv2.circle(vis, (int(px), int(py)), 4, colour, -1)
        cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (255, 0, 255), 1)
        label = {0: f"ep{args.episode} f0  REFERENCE",
                 i_min: f"f{i_min} sh={sh[i_min]:+.3f} DEEPEST",
                 n - 1: f"f{n - 1} END  moved "
                        f"{np.hypot(cx - pos[0][0], cy - pos[0][1]):.1f}px"}[i]
        cv2.putText(vis, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # tag zoom at start vs end, each centred on its own match
    zooms = []
    for i in (0, n - 1):
        cx, cy, _ = pos[i]
        c = int(round(cx)); r = int(round(cy))
        y0, y1 = max(0, r - args.half), min(frames[i].shape[0], r + args.half)
        x0, x1 = max(0, c - args.half), min(frames[i].shape[1], c + args.half)
        z = frames[i][y0:y1, x0:x1].copy()
        z = cv2.resize(z, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
        cv2.putText(z, f"tag zoom {'start' if i == 0 else 'end'}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(z, f"tag zoom {'start' if i == 0 else 'end'}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        zooms.append(cv2.cvtColor(z, cv2.COLOR_RGB2BGR))

    top = np.hstack(tiles)
    zz = np.hstack([cv2.copyMakeBorder(z, 0, max(0, top.shape[0] - z.shape[0]),
                                       0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                    for z in zooms])
    if zz.shape[1] < top.shape[1]:
        zz = cv2.copyMakeBorder(zz, 0, 0, 0, top.shape[1] - zz.shape[1],
                                cv2.BORDER_CONSTANT, value=(40, 40, 40))
    sheet = np.vstack([top, zz[:, :top.shape[1]]])

    outdir = REPO / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"verify_episode{args.episode:03d}.png"
    cv2.imwrite(str(out), sheet)
    print(f"\nwrote {out} ({sheet.shape[1]}x{sheet.shape[0]})")
    print("circles: blue = tag position at f0 (reference), green = tag position in that tile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
