"""Inspect a recorded LeRobotDataset: schema, timing, values, and decoded video.

Purpose is to catch the failures that a successful ``save_episode()`` hides:

  * video that encodes and decodes to something degenerate (all black / all flat)
  * state and action that are identical, which would mean the action source is
    wired to the measurement instead of the commanded target
  * non-uniform frame timing, which a skipped frame produces silently because
    LeRobot derives ``timestamp`` from ``frame_index / fps``
  * joint values outside the arm's own soft limits

It also writes a few decoded frames as PNGs so a human can look at what was
actually recorded.

Run::

    .venv\\Scripts\\python.exe tools\\inspect_dataset.py
    .venv\\Scripts\\python.exe tools\\inspect_dataset.py --root data/arm27_episode --frames 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO / ".cache" / "lerobot"))

sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

JOINTS = ("q_base", "q_shoulder", "q_elbow")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a recorded dataset.")
    ap.add_argument("--root", default=str(REPO / "data" / "arm27_episode"))
    ap.add_argument("--repo-id", default="local/arm27_episode")
    ap.add_argument("--frames", type=int, default=3, help="how many frames to dump as PNG")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root)
    ds = LeRobotDataset(args.repo_id, root=root)
    print(f"dataset      : {root}")
    print(f"frames       : {len(ds)}")
    print(f"fps          : {ds.fps}")
    print(f"episodes     : {len(ds.episodes) if hasattr(ds, 'episodes') else 'n/a'}")
    print(f"robot_type   : {ds.meta.robot_type}")
    print()
    print("features:")
    for key, spec in ds.meta.features.items():
        print(f"  {key:32s} dtype={spec['dtype']:8s} shape={spec['shape']}")
    print()

    cam_keys = [k for k in ds.meta.features if k.startswith("observation.images.")]
    if not cam_keys:
        print("WARNING: no image features in this dataset")
    problems: list[str] = []

    # ---- timing ----
    n = len(ds)
    items = [ds[i] for i in range(n)]
    ts = np.array([float(it["timestamp"]) for it in items])
    fi = np.array([int(it["frame_index"]) for it in items])
    deltas = np.diff(ts)
    print("timing:")
    print(f"  timestamp range : {ts[0]:.4f} .. {ts[-1]:.4f} s")
    print(f"  frame_index     : {fi[0]} .. {fi[-1]}  (contiguous: {bool(np.all(np.diff(fi) == 1))})")
    print(f"  dt              : mean={deltas.mean()*1000:.3f}ms "
          f"min={deltas.min()*1000:.3f}ms max={deltas.max()*1000:.3f}ms")
    expected_dt = 1.0 / ds.fps
    if not np.allclose(deltas, expected_dt, atol=1e-6):
        problems.append(f"timestamps are not uniform at {ds.fps} fps")
    if not np.all(np.diff(fi) == 1):
        problems.append("frame_index is not contiguous")
    print()

    # ---- state / action ----
    states = np.stack([np.asarray(it["observation.state"]).ravel() for it in items])
    actions = np.stack([np.asarray(it["action"]).ravel() for it in items])
    diff = actions - states
    print("state vs action (rad):")
    for i, name in enumerate(JOINTS):
        s, a = states[:, i], actions[:, i]
        print(f"  {name:11s} state [{s.min():+.4f},{s.max():+.4f}] "
              f"action [{a.min():+.4f},{a.max():+.4f}] "
              f"|a-s| max={np.abs(a - s).max():.5f}")
    print(f"  state range spans : {np.ptp(states, axis=0)}")
    print(f"  identical?        : {np.allclose(states, actions)}")

    if np.allclose(states, actions, atol=1e-9):
        problems.append("state and action are identical - the action is the "
                        "measurement, not the commanded target")
    if np.all(np.ptp(states, axis=0) < 1e-3):
        print("  NOTE: the arm barely moved, so this episode contains no motion. "
              "The pipeline is proven but the data is not yet useful for training.")

    # ---- limit screen, using the firmware's own numbers ----
    from arm_lerobot.arm_model import check_limits

    violations = 0
    for row in states:
        if check_limits(float(row[0]), float(row[1]), float(row[2])):
            violations += 1
    print(f"  limit violations  : {violations} / {n} frames")
    if violations:
        problems.append(f"{violations} frames violate the firmware's own soft limits")
    print()

    # ---- images: decode and check they are not degenerate ----
    for key in cam_keys:
        imgs = np.stack([np.asarray(it[key]) for it in items])
        # LeRobot stores channels-first float32 in [0,1]
        per_frame_std = imgs.reshape(len(imgs), -1).std(axis=1)
        print(f"images {key}:")
        print(f"  shape/dtype     : {imgs.shape} {imgs.dtype}")
        print(f"  value range     : {imgs.min():.4f} .. {imgs.max():.4f}")
        print(f"  per-frame std   : min={per_frame_std.min():.4f} "
              f"mean={per_frame_std.mean():.4f}")
        print(f"  frames distinct : {len(np.unique(imgs.reshape(len(imgs), -1), axis=0))} / {len(imgs)}")
        if per_frame_std.min() < 1e-4:
            problems.append(f"{key} contains a degenerate (flat) frame")
        if len(np.unique(imgs.reshape(len(imgs), -1), axis=0)) == 1 and len(imgs) > 1:
            problems.append(f"{key} is the same image in every frame")

        # Dump a few frames so a human can confirm what was recorded.
        try:
            import cv2
        except ImportError:
            continue
        outdir = REPO / "outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        picks = sorted({0, len(items) // 2, len(items) - 1})[: args.frames]
        for idx in picks:
            img = np.asarray(items[idx][key])
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            path = outdir / f"dataset_{key.split('.')[-1]}_frame{idx:04d}.png"
            cv2.imwrite(str(path), img)
            print(f"  wrote {path}")
    print()

    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("DATASET LOOKS GOOD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
