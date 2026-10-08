"""Dump a contact sheet of one episode's video, with the tracked blob marked.

Used to settle a disagreement between two measurements of the same episode:
the live depth measurement in ``record_episode.py`` and the offline pixel
measurement in ``analyze_episodes.py``.

Run::

    .venv\\Scripts\\python.exe tools\\dump_video_frames.py --episode 1
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
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--picks", type=int, default=6)
    args = ap.parse_args()

    import av
    import cv2
    from arm_lerobot.block import measure_block

    root = Path(args.root)
    vid = (root / "videos" / args.camera_key / "chunk-000"
           / f"file-{args.episode:03d}.mp4")
    if not vid.exists():
        print(f"missing {vid}")
        return 1

    frames = []
    with av.open(str(vid)) as container:
        for f in container.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
    n = len(frames)
    print(f"{vid.name}: {n} frames")

    idxs = np.linspace(0, n - 1, args.picks).astype(int)
    tiles = []
    for i in idxs:
        rgb = frames[i]
        obs = measure_block(rgb)
        vis = rgb.copy()
        note = f"f{i} {100 * i / max(n - 1, 1):.0f}%"
        if obs.found:
            cv2.rectangle(vis, (obs.bbox[0], obs.bbox[1]),
                          (obs.bbox[0] + obs.bbox[2], obs.bbox[1] + obs.bbox[3]),
                          (255, 0, 0), 2)
            cv2.circle(vis, (int(obs.u), int(obs.v)), 4, (0, 255, 0), -1)
            note += f" area={obs.area_px} uv=({obs.u:.0f},{obs.v:.0f})"
            print(f"  f{i:>4} ({100 * i / max(n - 1, 1):3.0f}%) found area={obs.area_px:>5} "
                  f"uv=({obs.u:6.1f},{obs.v:6.1f}) bbox={obs.bbox}")
        else:
            note += f" NOT FOUND ({obs.reason})"
            print(f"  f{i:>4} ({100 * i / max(n - 1, 1):3.0f}%) NOT FOUND: {obs.reason}")
        cv2.putText(vis, note, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, note, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    sheet = np.vstack(rows)
    outdir = REPO / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"episode{args.episode:03d}_contact_sheet.png"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
