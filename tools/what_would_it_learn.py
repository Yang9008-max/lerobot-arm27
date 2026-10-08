"""What would a policy actually learn from this dataset?

The question "train it and see" is normally answered by training, but for a set
this small and this inconsistent the answer is already determined by the data, and
it is worth stating precisely before any GPU time is spent.

Three things decide it:

1. **How many episodes contain the thing we want imitated.**
   Success is defined the only way that is true by definition - the block moved and
   stayed moved.  Measured from the videos with the R-tag template tracker
   (``track_block_tag.py``), because the colour blob is not a reliable witness.

2. **What a conditional-mean policy would command.**
   ACT is a conditional generative model, but at inference LeRobot runs it with the
   latent at its prior mean, so what it emits is essentially E[action | observation].
   With few demonstrations that mean is dominated by the majority behaviour.  If the
   majority of episodes never reach the block, the mean never reaches the block
   either - regardless of how long it trains.

3. **Whether the observation can disambiguate.**
   A conditional mean is only wrong when the same observation has different
   answers.  So the dataset is split by proprioceptive state and the spread of the
   commanded action inside each bin is measured: that spread is the irreducible
   error of any deterministic policy, and it is also the amount of "go left or go
   right?" ambiguity the images would have to resolve.

Run::

    .venv\\Scripts\\python.exe tools/what_would_it_learn.py
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
import pandas as pd  # noqa: E402

JOINTS = ("q_base", "q_shoulder", "q_elbow")

# Contact threshold, from the two episodes that provably moved the block: their
# shoulder minima were -3.518 and -3.391, and both were within about 3 degrees of
# the block.  -3.45 rad is the midpoint, matching the number already recorded in
# docs/HANDOFF.md.  It is a threshold, not a law: it depends on where the block is.
CONTACT_SHOULDER = -3.45


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "arm27_push"))
    ap.add_argument("--success", default="0,4",
                    help="episodes that provably moved the block (from track_block_tag.py)")
    ap.add_argument("--bins", type=int, default=24)
    args = ap.parse_args()

    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    parquets = sorted((root / "data").rglob("*.parquet"))
    success = {int(s) for s in args.success.split(",") if s.strip()}

    eps = []
    for i, f in enumerate(parquets):
        df = pd.read_parquet(f)
        st = np.stack([np.asarray(v, dtype=float).ravel()
                       for v in df["observation.state"].to_numpy()])
        ac = np.stack([np.asarray(v, dtype=float).ravel()
                       for v in df["action"].to_numpy()])
        eps.append({"i": i, "state": st, "action": ac, "n": len(st),
                    "ok": i in success})

    print("=" * 78)
    print("1. WHAT IS IN THE DATASET")
    print("=" * 78)
    print(f"{'ep':>3} {'frames':>7} {'sec':>5} {'sh_start':>9} {'sh_min':>8} "
          f"{'reaches?':>9}  {'outcome':>8}")
    for e in eps:
        sh = e["state"][:, 1]
        reaches = sh.min() <= CONTACT_SHOULDER
        print(f"{e['i']:>3} {e['n']:>7} {e['n'] / fps:>5.1f} {sh[0]:>+9.3f} "
              f"{sh.min():>+8.3f} {'YES' if reaches else 'no':>9}  "
              f"{'CONTACT' if e['ok'] else 'miss':>8}")
    n_ok = sum(e["ok"] for e in eps)
    print()
    print(f"  episodes that moved the block : {n_ok} / {len(eps)}")
    print(f"  frames showing contact        : "
          f"{sum(e['state'][:, 1].min() <= CONTACT_SHOULDER for e in eps)} episodes"
          f" dip below {CONTACT_SHOULDER} rad")

    print()
    print("=" * 78)
    print("2. WHAT A CONDITIONAL-MEAN POLICY WOULD COMMAND")
    print("=" * 78)

    # Resample every episode to the same length, then average.  This is the
    # crudest possible conditional mean - "average the demonstrations" - and it is
    # the right thing to look at, because with a handful of episodes and a network
    # that has not had time to resolve the conditioning, this is what comes out.
    m = 200
    grid = np.linspace(0, 1, m)
    aligned_s = np.stack([
        np.stack([np.interp(grid, np.linspace(0, 1, e["n"]), e["state"][:, j])
                  for j in range(3)], axis=1)
        for e in eps
    ])
    mean_s = aligned_s.mean(axis=0)

    print("  all 5 episodes averaged, shoulder training curve:")
    for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        k = int(q * (m - 1))
        print(f"    progress {q:>4.0%}   shoulder {mean_s[k, 1]:>+7.3f}"
              + ("   <-- deepest" if k == int(np.argmin(mean_s[:, 1])) else ""))
    print()
    print(f"  mean-policy shoulder minimum : {mean_s[:, 1].min():+.3f} rad")
    print(f"  contact needs                : {CONTACT_SHOULDER:+.3f} rad")
    print(f"  short by                     : {mean_s[:, 1].min() - CONTACT_SHOULDER:+.3f} rad"
          f"  ({np.degrees(mean_s[:, 1].min() - CONTACT_SHOULDER):+.1f} deg)")
    verdict = ("NEVER TOUCHES THE BLOCK" if mean_s[:, 1].min() > CONTACT_SHOULDER
               else "reaches the block")
    print(f"  => an averaged policy {verdict}")
    print()

    ok_s = aligned_s[[e["ok"] for e in eps]].mean(axis=0)
    print(f"  for reference, averaging only the {n_ok} successful episodes:")
    print(f"    shoulder minimum {ok_s[:, 1].min():+.3f} rad "
          f"({ok_s[:, 1].min() - CONTACT_SHOULDER:+.3f} vs contact) "
          f"- reaches, because every episode in it does")

    print()
    print("=" * 78)
    print("3. CAN THE OBSERVATION RESOLVE THE AMBIGUITY?")
    print("=" * 78)
    all_s = np.concatenate([e["state"] for e in eps])
    all_a = np.concatenate([e["action"] for e in eps])
    edges = np.linspace(all_s[:, 1].min(), all_s[:, 1].max(), args.bins + 1)
    print(f"{'shoulder bin':>18} {'frames':>7} {'episodes':>9}  "
          f"{'action spread (rad)':>34}")
    print(f"{'':>18} {'':>7} {'':>9}  {'base':>10} {'shoulder':>10} {'elbow':>10}")
    worst = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (all_s[:, 1] >= lo) & (all_s[:, 1] < hi)
        if msk.sum() < 15:
            continue
        who = np.concatenate([[e["i"]] * e["n"] for e in eps])[msk]
        n_ep = len(np.unique(who))
        sp = [float(np.ptp(all_a[msk, j])) for j in range(3)]
        worst.append((sp[2], lo, hi, n_ep, int(msk.sum())))
        print(f"{lo:>+8.2f}..{hi:>+6.2f} {int(msk.sum()):>7} {n_ep:>9}  "
              f"{sp[0]:>10.3f} {sp[1]:>10.3f} {sp[2]:>10.3f}")
    print()
    print("  A spread of several tenths of a radian inside one proprioceptive bin "
          "means the")
    print("  demonstrations disagree about what to do at that state.  The network has "
          "to")
    print("  read the answer out of the image instead, and only the episodes where the")
    print("  block is actually reachable can teach it that.")

    print()
    print("=" * 78)
    print("4. VISUAL CONSISTENCY (where the block is at the start of each episode)")
    print("=" * 78)
    # Reuse the tag tracker's frame-0 detections: block colour centroid is unstable,
    # so instead report the block's pixel position measured by the tracker, which
    # ``track_block_tag.py`` prints.  Here we only quantify the spread from the
    # colour blob at t=0, which is stable enough at rest (the drift appears later).
    import cv2
    import av
    from arm_lerobot.block import measure_block

    u0 = []
    for e, f in zip(eps, parquets):
        vid = root / "videos" / "observation.images.front" / "chunk-000" / f"{f.stem}.mp4"
        with av.open(str(vid)) as c:
            fr = next(c.decode(video=0))
        rgb = fr.to_ndarray(format="rgb24")
        obs = measure_block(rgb)
        u0.append(obs.u if obs.found else float("nan"))
    u0 = np.array(u0)
    for e, u in zip(eps, u0):
        print(f"  ep{e['i']}: block at u={u:6.1f} px  ({'MUST REACH' if e['ok'] else 'never reached'})")
    print(f"  spread across episodes: {np.nanmax(u0) - np.nanmin(u0):.1f} px "
          f"= {(np.nanmax(u0) - np.nanmin(u0)) * 2.90:.0f} mm at 1500 mm")
    print("  the block is in a different place in every episode, so the image is the")
    print("  only cue that could tell the policy where to go - and with "
          f"{n_ok} episodes that")
    print("  actually get there, there is nothing to learn that cue from.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
