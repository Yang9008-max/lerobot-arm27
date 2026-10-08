"""Headless single-episode recorder for the arm + Gemini 2 (phase 1b).

Why not just use ``lerobot-record``?
------------------------------------
``lerobot-record`` is interactive: it waits for ENTER between episodes and expects
a human at the keyboard.  That is right for real data collection, but it cannot be
scripted, and it also has no notion of "do not record while the arm is disabled".

This driver is the scriptable counterpart.  It uses LeRobot's own
``hw_to_dataset_features`` / ``build_dataset_frame`` / ``LeRobotDataset`` so the
output is byte-for-byte the same format, and it adds two guards that come from
observed firmware behaviour:

  1. The arm must be **enabled** before recording starts.  While disabled the
     firmware forces the commanded targets to (0, 0, 0), so any action recorded
     in that state is meaningless.
  2. Enabling the arm makes it slew to its target (observed: the base moved
     0.239 rad at ~0.95 rad/s on enable).  A settle delay after enable keeps that
     transient out of the dataset.

Run::

    .venv\\Scripts\\python.exe tools\\record_episode.py --seconds 8
    .venv\\Scripts\\python.exe tools\\record_episode.py --seconds 8 --no-camera
    .venv\\Scripts\\python.exe tools\\record_episode.py --seconds 8 --camera front
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Caches must be redirected before LeRobot is imported - it resolves its data
# directories at import time and the default lands on a nearly-full C:.
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO / ".cache" / "lerobot"))

sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402


def build(port: str, camera: str | None, camera_size: tuple[int, int], fps: int):
    from arm_lerobot.lerobot_camera import GeminiCameraConfig
    from arm_lerobot.lerobot_robot import ArmRobot, ArmRobotConfig
    from arm_lerobot.lerobot_teleop import ArmTelemetry, ArmTelemetryConfig

    cameras = {}
    if camera:
        cameras[camera] = GeminiCameraConfig(
            width=camera_size[0], height=camera_size[1], fps=fps
        )

    robot = ArmRobot(ArmRobotConfig(port=port, cameras=cameras))
    teleop = ArmTelemetry(ArmTelemetryConfig(port=port))
    return robot, teleop


def wait_for_enabled(robot, teleop, timeout_s: float) -> bool:
    """Block until the operator enables the arm (right 3-position switch UP)."""
    print(f"waiting for the arm to be enabled (right switch UP) - up to {timeout_s:.0f}s")
    deadline = time.monotonic() + timeout_s
    last_note = 0.0
    while time.monotonic() < deadline:
        state = robot._reader.latest()
        if state is not None and state.enabled:
            print("  arm is ENABLED")
            return True
        now = time.monotonic()
        if now - last_note > 2.0:
            last_note = now
            if state is None:
                print("  ... no telemetry yet")
            else:
                print(f"  ... disabled (swR={state.switch_right}, link="
                      f"{'up' if state.remote_online else 'DOWN'})")
        time.sleep(0.1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Record one episode headlessly.")
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--camera", default="front", help="camera name, or empty for none")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="wait after enabling before recording, to skip the enable transient")
    ap.add_argument("--wait-enabled-s", type=float, default=60.0)
    ap.add_argument("--task", default="push the block into the target area")
    ap.add_argument("--repo-id", default="local/arm27_episode")
    ap.add_argument("--root", default=str(REPO / "data" / "arm27_episode"))
    ap.add_argument("--no-wait-enabled", action="store_true",
                    help="record even if the arm is disabled (produces garbage actions)")
    ap.add_argument("--wait-motion-s", type=float, default=90.0,
                    help="how long to wait for the operator to start moving the arm")
    ap.add_argument("--motion-threshold", type=float, default=0.05,
                    help="radians of joint change that count as 'started moving'")
    ap.add_argument("--no-wait-motion", action="store_true",
                    help="start recording immediately after settling")
    args = ap.parse_args()

    camera = None if args.no_camera else (args.camera or None)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    robot, teleop = build(args.port, camera, (args.width, args.height), args.fps)

    print(f"connecting: port={args.port} camera={camera!r} fps={args.fps}")
    t0 = time.monotonic()
    robot.connect()
    teleop.connect()
    print(f"  connected in {time.monotonic() - t0:.2f}s")

    try:
        if not args.no_wait_enabled:
            if not wait_for_enabled(robot, teleop, args.wait_enabled_s):
                print("FAILED: the arm was never enabled. Nothing recorded.")
                print("        Flip the right 3-position switch UP, then re-run.")
                return 1
            if args.settle_s > 0:
                print(f"settling {args.settle_s:.1f}s (skipping the enable transient)")
                time.sleep(args.settle_s)

            # ---- baseline measurement of the block, taken while the arm is still
            # at its initial pose so nothing occludes it.  Done BEFORE the motion
            # trigger, because by the time motion starts the block may already
            # have been touched.
            block_before = None
            if camera:
                from arm_lerobot.block import measure_block

                cam = robot.cameras[camera]
                frame = cam.last_capture()
                if frame is not None and frame.rgb is not None:
                    block_before = measure_block(
                        frame.rgb, frame.depth, cam.intrinsics
                    )
                    if block_before.found:
                        print(f"  block baseline: {block_before.describe()}")
                    else:
                        print(f"  block baseline: {block_before.reason}")

            if not args.no_wait_motion:
                # Start timing from the first real movement rather than from the
                # moment the arm was enabled.  Otherwise every episode opens with
                # however long the operator took to reach for the sticks, and the
                # recorder has no way to know that dead time was not the task.
                # Retry rather than dying.  The resampler needs a few telemetry
                # lines before it can answer, and it deliberately restarts when the
                # stream has a discontinuity - which opening the port mid-stream
                # produces, because the serial buffer holds a stale backlog.  An
                # earlier version aborted the whole run one second before it would
                # have worked.
                base = None
                ready_deadline = time.monotonic() + 10.0
                while time.monotonic() < ready_deadline:
                    base = robot.arm_frame()
                    if base is not None:
                        break
                    print(f"  ... alignment not ready yet "
                          f"(last line age="
                          f"{robot._reader.age_s() if robot._reader.age_s() is None else round(robot._reader.age_s(), 3)}s, "
                          f"reader={robot._reader.stats()})")
                    time.sleep(0.5)
                if base is None:
                    print("FAILED: no aligned telemetry after 10 s. Is the H7 powered "
                          "and streaming? "
                          f"Reader stats: {robot._reader.stats()}")
                    return 1
                baseline = base.q
                print(f"ready. MOVE THE ARM NOW with the transmitter "
                      f"(trigger: any joint moving > {args.motion_threshold} rad, "
                      f"waiting up to {args.wait_motion_s:.0f}s)")
                print(f"  baseline q = ({baseline[0]:+.4f}, {baseline[1]:+.4f}, "
                      f"{baseline[2]:+.4f})")
                deadline = time.monotonic() + args.wait_motion_s
                moved = False
                last_note = 0.0
                while time.monotonic() < deadline:
                    frame = robot.arm_frame()
                    if frame is not None:
                        dev = max(abs(a - b) for a, b in zip(frame.q, baseline))
                        if dev > args.motion_threshold:
                            print(f"  motion detected (max joint change {dev:.4f} rad) "
                                  f"- recording starts now")
                            moved = True
                            break
                        now = time.monotonic()
                        if now - last_note > 5.0:
                            last_note = now
                            print(f"  ... still waiting (largest change so far "
                                  f"{dev:.4f} rad)")
                    time.sleep(0.02)
                if not moved:
                    print(f"FAILED: the arm never moved more than "
                          f"{args.motion_threshold} rad in {args.wait_motion_s:.0f}s. "
                          f"Nothing recorded. Use --no-wait-motion to record anyway.")
                    return 1

        # Dataset schema, generated by LeRobot from the robot's own feature spec.
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        act_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
        features = {**obs_features, **act_features}

        root = Path(args.root)
        existing = root.exists() and any(root.iterdir())
        if existing:
            # APPEND, never wipe.  The normal workflow is "run this once per
            # episode", and an earlier version called shutil.rmtree(root) here -
            # which silently destroyed every episode recorded so far, so the
            # dataset could never hold more than one.
            dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=root)
            print(f"appending to dataset at {root} "
                  f"({dataset.num_episodes} episode(s) already, {dataset.fps} fps)")
            if int(dataset.fps) != int(args.fps):
                print(f"FAILED: existing dataset is {dataset.fps} fps but --fps is "
                      f"{args.fps}. Timestamps would be inconsistent, so this is "
                      f"refused. Use a different --root, or the matching --fps.")
                return 1
        else:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id, fps=args.fps, features=features,
                root=root, robot_type=robot.name, use_videos=True,
            )
            print(f"created new dataset at {root}")

        period = 1.0 / args.fps
        n_target = int(args.seconds * args.fps)
        written = 0
        skipped = 0
        stale = 0
        intervals: list[float] = []
        next_t = time.monotonic()

        print(f"recording {args.seconds:.1f}s ({n_target} frames at {args.fps} fps) ...")
        while written < n_target:
            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

            frame_start = time.monotonic()
            try:
                obs = robot.get_observation()
                action = teleop.get_action()
            except TimeoutError as exc:
                stale += 1
                if stale <= 3 or stale % 20 == 0:
                    print(f"  skip (telemetry): {exc}")
                skipped += 1
                if stale > 90:
                    print("FAILED: telemetry lost for too long; aborting.")
                    break
                continue

            robot.send_action(action)  # phase 1: a documented no-op
            dataset.add_frame({
                **build_dataset_frame(features, obs, prefix="observation"),
                **build_dataset_frame(features, action, prefix="action"),
                "task": args.task,
            })
            written += 1
            intervals.append(frame_start)

            if written % args.fps == 0:
                pf = robot.arm_frame()
                el = ""
                if pf is not None:
                    el = (f" q=({pf.q[0]:+.3f},{pf.q[1]:+.3f},{pf.q[2]:+.3f})"
                          f" q*=({pf.q_target[0]:+.3f},{pf.q_target[1]:+.3f},{pf.q_target[2]:+.3f})"
                          f" enabled={pf.enabled}")
                print(f"  {written:4d}/{n_target} frames{el}")

        if written == 0:
            print("FAILED: no frames recorded.")
            return 1

        print(f"saving episode ({written} frames) ...")
        dataset.save_episode()
        if hasattr(dataset, "finalize"):
            dataset.finalize()

        # ---- did the push succeed? ----
        # Measured at two moments when the arm is NOT in front of the block: before
        # the motion trigger, and now.  During the push the arm occludes the block,
        # so anything measured mid-episode would be measuring the arm.
        push_verdict: bool | None = None
        if camera and block_before is not None and block_before.found:
            from arm_lerobot.block import measure_block, pushed_forward

            # The task is CONTACT, not displacement: the arm is a 3-DOF positioner
            # with no gripper, and at full extension it has almost no pushing
            # authority.  So the criterion is "the block was disturbed at all".
            #
            # 5 mm is grounded in the measurement, not guessed.  The block's
            # camera-frame position repeats to +/-1 mm across frames (measured over
            # five frames at rest), so 5 mm is 5 sigma: a real nudge rather than
            # noise.  A successful touch measured 8.2 mm in practice.
            #
            # Caveat: a perfectly tangential touch that leaves the block exactly
            # where it was will not register.  If that turns out to happen, the fix
            # is a contact detector on the joint torques (the `watch` line carries
            # ta/te) rather than a smaller threshold, which would just chase noise.
            CONTACT_THRESHOLD_MM = 5.0
            BLOCK_CHECK_S = 12.0

            cam = robot.cameras[camera]
            print()
            print(f"waiting up to {BLOCK_CHECK_S:.0f}s for a clear view of the block "
                  f"- retract the arm so it is not in the way ...")
            deadline = time.monotonic() + BLOCK_CHECK_S
            block_after = None
            last_seen = None
            while time.monotonic() < deadline:
                frame = cam.last_capture()
                if frame is not None and frame.rgb is not None:
                    obs = measure_block(
                        frame.rgb, frame.depth, cam.intrinsics,
                        baseline_area_px=block_before.area_px,
                    )
                    if obs.found:
                        block_after = obs
                        break
                    last_seen = obs
                time.sleep(0.1)

            print()
            if block_after is None:
                why = last_seen.reason if last_seen is not None else "no frame captured"
                print(f"contact verdict   : UNKNOWN - could not get a clear view ({why})")
            else:
                push_verdict, why = pushed_forward(
                    block_before, block_after, CONTACT_THRESHOLD_MM
                )
                label = {True: "CONTACT", False: "NO CONTACT", None: "UNKNOWN"}[push_verdict]
                print(f"block before      : {block_before.describe()}")
                print(f"block after       : {block_after.describe()}")
                print(f"contact verdict   : {label} - {why}")

        # ---- report ----
        print()
        print("--- results ---")
        print(f"frames written    : {written} of {n_target} requested")
        print(f"frames skipped    : {skipped} (telemetry unavailable: {stale})")

        # LeRobot derives `timestamp = frame_index / fps`, so the DECLARED fps must
        # match what the loop actually achieved.  Dividing the frame count by the
        # *requested* duration hides a shortfall: a real 1280x720 run on this laptop
        # sustained only 19.6 fps while the dataset claimed 30, which would have
        # told the policy the motion happened 1.53x faster than it did.
        achieved_fps = 0.0
        span = 0.0
        if len(intervals) > 1:
            deltas = np.diff(intervals)
            span = intervals[-1] - intervals[0]
            achieved_fps = (len(intervals) - 1) / span if span > 0 else 0.0
            print(f"loop period       : mean={deltas.mean()*1000:.1f}ms "
                  f"p95={np.percentile(deltas, 95)*1000:.1f}ms "
                  f"max={deltas.max()*1000:.1f}ms")
            print(f"wall clock        : {span:.2f} s for {len(intervals)} frames")
        print(f"DECLARED fps      : {args.fps}")
        print(f"ACHIEVED fps      : {achieved_fps:.2f}")

        rate_ok = achieved_fps > 0 and achieved_fps >= 0.95 * args.fps
        if achieved_fps > 0 and not rate_ok:
            shortfall = 100.0 * (1.0 - achieved_fps / args.fps)
            print()
            print("  " + "*" * 68)
            print(f"  *** the loop ran {shortfall:.0f}% slower than the declared rate ***")
            print("  " + "*" * 68)
            print(f"  The episode WAS saved, but its timestamps are wrong: LeRobot writes")
            print(f"  timestamp = frame_index / fps, so the dataset clock is"
                  f" {args.fps / achieved_fps:.2f}x faster than reality.")
            print(f"  Do not use this episode for training. Fixes, best first:")
            print(f"    1. record at a rate you can actually hold:  --fps {int(achieved_fps)}")
            print(f"    2. drop to 640x480 (4x fewer pixels; this laptop held 30 fps there)")
            print(f"    3. lower the camera resolution or frame rate")

        print(f"teleop snapshot   : {teleop.stats()}")
        print(f"reader            : {robot._reader.stats()}")
        if camera:
            cam = robot.cameras[camera]
            print(f"camera            : {cam.stats() if hasattr(cam, 'stats') else 'n/a'}")
        print(f"limit violations  : {robot.limit_violations}")
        print(f"stale observations: {robot.stale_observations}")

        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        print(f"on disk           : {size/1e6:.2f} MB")
        print(f"path              : {root}")

        # ---- read back to prove the data is usable ----
        print()
        print("--- read-back check ---")
        reopened = LeRobotDataset(args.repo_id, root=root)
        print(f"frames in dataset : {len(reopened)}")
        item = reopened[0]
        state = np.asarray(item["observation.state"]).ravel()
        action = np.asarray(item["action"]).ravel()
        print(f"frame 0 state     : {state}")
        print(f"frame 0 action    : {action}")
        print(f"state != action   : {not np.allclose(state, action)} "
              f"(they must differ, otherwise the action source is wrong)")
        print(f"image present     : {'observation.images.' + camera in item if camera else 'n/a'}")
        if camera:
            img = np.asarray(item[f"observation.images.{camera}"])
            print(f"image shape/dtype : {img.shape} {img.dtype}")
        print()
        print("Visual check:  .venv\\Scripts\\python.exe -m lerobot.scripts.lerobot_dataset_viz "
              f"--repo-id {args.repo_id} --root {root}")
        return 0
    finally:
        robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
