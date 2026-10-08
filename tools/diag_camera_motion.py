"""Measure how much the CAMERA itself moved during each episode.

Discovery that motivated this
-----------------------------
Cropping a fixed pixel window around the block and comparing the start, the
deepest-shoulder moment and the end showed the *background* changing completely
between those frames - a person at the start, dark machinery mid-episode, a desk
at the end.  A fixed pixel window can only show a changing background if the
camera moved, so the camera is not static during an episode.

That matters far more than whether the block was nudged:

  * every image observation in the dataset comes from a different viewpoint, so
    the visual part of the observation is partly a function of the arm's own pose
  * any metric displacement measured in the camera frame (the live depth check in
    ``record_episode.py`` included) is only meaningful at moments when the camera
    is back where it started

Method
------
Phase-correlate a background patch - the top strip of the image, chosen to
contain neither the arm nor the block - between frame 0 and every later frame.
Phase correlation returns the sub-pixel translation between two images, so it
measures camera motion regardless of lighting.  It is then correlated against the
shoulder angle to test the obvious hypothesis: the camera is carried by the arm.

Run::

    .venv\\Scripts\\python.exe tools/diag_camera_motion.py
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
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, default=60, help="background strip, top of frame")
    args = ap.parse_args()

    import av
    import cv2
    import pandas as pd

    root = Path(args.root)
    parquets = sorted((root / "data").rglob("*.parquet"))

    print(f"background strip = rows {args.y0}..{args.y1} (no arm, no block)")
    print()
    hdr = (f"{'ep':>4} {'cam shift':>10} {'at %':>5} {'sh there':>9} "
           f"{'corr(shift,|sh-sh0|)':>21} {'sh_min':>8}")
    print(hdr)
    print("-" * len(hdr))

    for f in parquets:
        df = pd.read_parquet(f, columns=["observation.state"])
        st = np.stack([np.asarray(v, dtype=float).ravel()
                       for v in df["observation.state"].to_numpy()])
        sh = st[:, 1]

        vid = root / "videos" / args.camera_key / "chunk-000" / f"{f.stem}.mp4"
        if not vid.exists():
            continue
        gray = []
        with av.open(str(vid)) as container:
            for fr in container.decode(video=0):
                rgb = fr.to_ndarray(format="rgb24")
                gray.append(cv2.cvtColor(rgb[args.y0:args.y1], cv2.COLOR_RGB2GRAY)
                            .astype(np.float32))
        n = min(len(gray), len(sh))
        ref = gray[0]

        shift = np.zeros(n)
        for i in range(1, n):
            (dx, dy), _ = cv2.phaseCorrelate(ref, gray[i])
            shift[i] = float(np.hypot(dx, dy))

        i_max = int(np.argmax(shift))
        # Does the camera motion track the shoulder?
        a = np.abs(sh[:n] - sh[0])
        c = float(np.corrcoef(shift[1:], a[1:])[0, 1]) if n > 3 else float("nan")
        print(f"{f.stem[-3:]:>4} {shift[i_max]:>9.2f}px {100 * i_max / max(n - 1, 1):>4.0f}% "
              f"{sh[i_max]:>+9.3f} {c:>21.2f} {sh[:n].min():>+8.3f}")

    print()
    print("A shift of several pixels in the top strip is the camera moving. "
          "A correlation near +1 with |shoulder-shoulder0| means the camera rides "
          "the arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
