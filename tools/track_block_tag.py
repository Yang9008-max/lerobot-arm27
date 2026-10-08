"""Track the printed R-tag on the block to decide, per frame, whether it moved.

Why the tag and not the colour blob
-----------------------------------
The colour blob is not a reliable witness.  Measured directly: in episode 1 the
blob's bounding box jumps from 88x67 px to 98x73 px at frame 59 while its *area*
falls slightly, and its centroid slides 10 px left.  That is the mask merging with
a yellow-ish sliver of something else, not the block moving.  It produced a
phantom 31 mm "push" in an episode that never touched the block.

The tag is high-contrast, has internal structure, and sits on the block's face, so
normalised cross-correlation finds it to well under a pixel and cannot be fooled
by background colour.  Matching is done over a rotation sweep as well, because a
pushed block can be *rotated* rather than translated, and a rotation is just as
real a contact event.

Reported per episode: the tag's peak position and match score at ten points
through the episode, the largest displacement seen, and - the column that matters
most - the displacement at the END.  A block that was pushed stays pushed; a blob
that merely wobbled comes back.

Run::

    .venv\\Scripts\\python.exe tools/track_block_tag.py
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
    ap.add_argument("--angles", type=int, default=13, help="rotation sweep, -30..30 deg")
    ap.add_argument("--px-per-mm", type=float, default=517.19 / 1500.0)
    args = ap.parse_args()

    import av
    import cv2
    import pandas as pd
    from arm_lerobot.block import measure_block

    root = Path(args.root)
    parquets = sorted((root / "data").rglob("*.parquet"))

    angle_list = np.linspace(-30, 30, args.angles)

    def match(frame_gray, templ, templs):
        """Best peak over the rotation sweep.  Returns (x, y, score, angle)."""
        best = (-2.0, 0, 0, 0.0)
        for ang, tmpl in templs:
            res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(res)
            if mx > best[0]:
                h, w = tmpl.shape
                best = (mx, ml[0] + w / 2, ml[1] + h / 2, ang)
        return best[1], best[2], best[0], best[3]

    print(f"{'ep':>4} {'tag pos f0':>14} {'max move':>10} {'at %':>5} "
          f"{'sh there':>9} {'END move':>9} {'sh_end':>8} {'score':>6} {'rot':>6}")
    print("-" * 92)

    for f in parquets:
        df = pd.read_parquet(f, columns=["observation.state"])
        st = np.stack([np.asarray(v, dtype=float).ravel()
                       for v in df["observation.state"].to_numpy()])
        sh = st[:, 1]

        vid = root / "videos" / args.camera_key / "chunk-000" / f"{f.stem}.mp4"
        if not vid.exists():
            continue
        frames = []
        with av.open(str(vid)) as container:
            for fr in container.decode(video=0):
                frames.append(fr.to_ndarray(format="rgb24"))
        n = min(len(frames), len(sh))

        rgb0 = frames[0]
        gray0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2GRAY)
        obs = measure_block(rgb0)
        if not obs.found or obs.bbox is None:
            print(f"{f.stem[-3:]:>4}  block not found in frame 0 ({obs.reason})")
            continue
        x, y, w, h = obs.bbox
        # Inset the block bbox by 25% to land on the tag and exclude the block's
        # outer edge, which is what changes as the block rotates.
        ix, iy = int(x + 0.25 * w), int(y + 0.25 * h)
        iw, ih = int(0.5 * w), int(0.5 * h)
        templ0 = gray0[iy:iy + ih, ix:ix + iw]
        if templ0.size == 0:
            print(f"{f.stem[-3:]:>4}  empty template")
            continue

        templs = []
        for ang in angle_list:
            M = cv2.getRotationMatrix2D((iw / 2, ih / 2), float(ang), 1.0)
            rot = cv2.warpAffine(templ0, M, (iw, ih), borderMode=cv2.BORDER_REPLICATE)
            templs.append((float(ang), rot))

        samples = np.unique(np.linspace(0, n - 1, 11).astype(int))
        pts, scores, angs = [], [], []
        for i in range(n):
            g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            cx, cy, sc, ang = match(g, templ0, templs)
            pts.append((cx, cy))
            scores.append(sc)
            angs.append(ang)
        pts = np.array(pts)

        p0 = pts[0]
        d = np.linalg.norm(pts - p0, axis=1)
        i_max = int(np.argmax(d))
        d_end = float(d[n - 1])

        print(f"{f.stem[-3:]:>4} ({p0[0]:6.1f},{p0[1]:6.1f}) "
              f"{d[i_max]:>8.2f}px {100 * i_max / max(n - 1, 1):>4.0f}% {sh[i_max]:>+9.3f} "
              f"{d_end:>7.2f}px {sh[n - 1]:>+8.3f} {scores[i_max]:>6.2f} "
              f"{angs[i_max]:>+6.0f}")
        trace = " ".join(f"{100 * i / (n - 1):.0f}%:{d[i]:.1f}" for i in samples)
        print(f"      |tag move| px by episode progress:  {trace}")
        print(f"      |match score at end {scores[-1]:.2f} (low score = block "
              f"rotated or occluded, not a reliable position)")
        print()

    print(f"1 px = {1 / args.px_per_mm:.2f} mm at the assumed 1500 mm block distance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
