"""Offline end-to-end smoke test for the LeRobot adapter layer.

Runs with no arm and no camera attached, and answers the questions that actually
decide whether phase 1 will work:

  1. Do the adapter classes satisfy LeRobot's abstract interfaces?  Checked with
     ``inspect.isabstract`` so LeRobot itself is the judge, not a hand-written list.
  2. Do ``observation_features`` and ``get_observation()`` agree key for key?
     LeRobot requires exact equality, and a mismatch only shows up as a KeyError
     deep inside the record loop.
  3. Do the teleoperator's action keys match the robot's ``action_features``?
     ``build_dataset_frame`` looks actions up by name.
  4. Does LeRobot accept our feature spec and does a dataset round-trip -
     create, add synthetic frames, save, re-open, read back?

Cache redirection happens before LeRobot is imported, because LeRobot resolves its
data directories at import time.

Run:  .venv\\Scripts\\python.exe tools\\test_pipeline_smoke.py
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Must happen before `import lerobot`: lerobot/utils/constants.py computes
# HF_LEROBOT_HOME at import time, and the default lands on C: which has ~3 GB free.
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO / ".cache" / "lerobot"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


# ------------------------------------------------------------------- fakes
class FakeCamera:
    """Minimal stand-in for GeminiCamera: no SDK, deterministic pixels."""

    def __init__(self, height: int = 480, width: int = 640, fps: int = 30):
        self.height = height
        self.width = width
        self.fps = fps
        self.last_frame_host_time: float | None = None
        self.frames = 0
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, warmup: bool = True) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        self.last_frame_host_time = time.monotonic()
        self.frames += 1
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[..., 0] = (np.arange(self.width) % 256).astype(np.uint8)
        img[..., 1] = (np.arange(self.height)[:, None] % 256).astype(np.uint8)
        img[..., 2] = 128
        return img


class FakeReader:
    """Stands in for the shared TelemetryReader, backed by a real resampler.

    This exercises the genuine alignment code path, so a bug in the interpolation
    wiring shows up here rather than on the robot.
    """

    def __init__(self):
        from arm_lerobot.sync import ClockMapper, TelemetryResampler

        self.clock = ClockMapper()
        self.resampler = TelemetryResampler(clock=self.clock)
        self._last: tuple[float, object] | None = None

    def feed(self, state, host_time: float) -> None:
        self.resampler.add(state, host_time)

    def snapshot(self, host_time: float):
        state = self.resampler.state_at(host_time)
        if state is not None:
            self._last = (host_time, state)
        return state

    def last_snapshot(self, max_age_s: float = 0.2):
        if self._last is None:
            return None
        host_time, state = self._last
        if time.monotonic() - host_time > max_age_s:
            return None
        return state

    def state_age_s(self, host_time: float):
        return self.resampler.age_s(host_time)

    def stats(self) -> dict:
        return {"fake": True}


def make_state(t_ms: int, q, q_target=None):
    from arm_lerobot.telemetry import Q_SCALE, ArmState

    qt = q_target if q_target is not None else q
    e4 = lambda x: int(round(x * Q_SCALE))  # noqa: E731
    return ArmState(
        t_ms=t_ms, en=1, safe=0, online=1, chassis=0,
        qb=e4(q[0]), qa=e4(q[1]), qe=e4(q[2]),
        qbt=e4(qt[0]), qat=e4(qt[1]), qet=e4(qt[2]),
        r=0, z=0, wb=0, wa=0, we=0, tb=0, ta=0, te=0,
        sw_r=0, sw_l=32, sw_cr=200, sw_cl=300,
        rh=0, rv=0, lh=0, lv=0,
    )


def feed_recent_telemetry(reader: FakeReader, seconds: float = 1.0) -> None:
    """Populate the resampler with a smooth motion ending at 'now'."""
    now = time.monotonic()
    n = int(seconds * 20)
    for i in range(n):
        t_host = now - (n - 1 - i) / 20.0
        phase = i / 20.0
        q = (
            0.20 * np.sin(phase),
            -0.60 + 0.10 * np.sin(phase * 1.3),
            0.90 + 0.08 * np.sin(phase * 0.7),
        )
        q_target = (q[0] + 0.005, q[1] + 0.004, q[2] + 0.003)
        reader.feed(make_state(i * 50, q, q_target), t_host)


# -------------------------------------------------------------------- tests
def test_abstract_interfaces() -> None:
    print("1. Do the adapters satisfy LeRobot's abstract interfaces?")
    from arm_lerobot.camera import OrbbecCamera  # noqa: F401
    from arm_lerobot.lerobot_camera import GeminiCamera, GeminiCameraConfig
    from arm_lerobot.lerobot_robot import ArmRobot, ArmRobotConfig
    from arm_lerobot.lerobot_teleop import ArmTelemetry, ArmTelemetryConfig

    for cls in (GeminiCamera, ArmRobot, ArmTelemetry):
        missing = sorted(getattr(cls, "__abstractmethods__", frozenset()))
        check(
            f"{cls.__name__} implements every abstract member",
            not inspect.isabstract(cls),
            f"still abstract: {missing}" if missing else "",
        )

    # LeRobot resolves custom devices by stripping "Config" from the config class
    # name and looking the result up on the parent package.
    pairs = [
        (GeminiCameraConfig, GeminiCamera),
        (ArmRobotConfig, ArmRobot),
        (ArmTelemetryConfig, ArmTelemetry),
    ]
    import arm_lerobot

    for cfg_cls, dev_cls in pairs:
        expected = cfg_cls.__name__.removesuffix("Config")
        check(
            f"{cfg_cls.__name__} -> {expected} resolvable on the package",
            expected == dev_cls.__name__ and getattr(arm_lerobot, expected, None) is dev_cls,
        )

    # Registration is what makes --robot.type=arm_27 work.  Check it the way
    # LeRobot itself does - through the real factories - rather than by poking at
    # draccus internals.
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    try:
        cams = make_cameras_from_configs({"cam": GeminiCameraConfig()})
        check("camera factory resolves 'gemini'",
              cams.get("cam").__class__ is GeminiCamera, f"got {cams.get('cam')!r}")
    except Exception as exc:
        check("camera factory resolves 'gemini'", False, f"{type(exc).__name__}: {exc}")

    try:
        built = make_robot_from_config(
            ArmRobotConfig(port="COM_FAKE", cameras={"cam": GeminiCameraConfig()})
        )
        check("robot factory resolves 'arm_27'", isinstance(built, ArmRobot),
              f"got {type(built).__name__}")
    except Exception as exc:
        check("robot factory resolves 'arm_27'", False, f"{type(exc).__name__}: {exc}")

    try:
        built = make_teleoperator_from_config(ArmTelemetryConfig(port="COM_FAKE"))
        check("teleop factory resolves 'arm_telemetry'", isinstance(built, ArmTelemetry),
              f"got {type(built).__name__}")
    except Exception as exc:
        check("teleop factory resolves 'arm_telemetry'", False, f"{type(exc).__name__}: {exc}")


def test_feature_consistency() -> None:
    print("2. Feature declarations and observations must agree exactly")
    from arm_lerobot.lerobot_camera import GeminiCameraConfig
    from arm_lerobot.lerobot_robot import ArmRobot, ArmRobotConfig, JOINT_NAMES
    from arm_lerobot.lerobot_teleop import ArmTelemetry, ArmTelemetryConfig

    robot_cfg = ArmRobotConfig(port="COM_FAKE", cameras={"cam": GeminiCameraConfig()})
    robot = ArmRobot(robot_cfg)

    feats = robot.observation_features
    check("observation_features covers the 3 joints", all(n in feats for n in JOINT_NAMES),
          f"keys={sorted(feats)}")
    check("camera declared as (H, W, 3)",
          feats.get("cam") == (480, 640, 3), f"got {feats.get('cam')}")

    # Now with a fake reader/camera, the returned observation keys must match.
    fake = FakeReader()
    feed_recent_telemetry(fake)
    robot._reader = fake
    robot.cameras = {"cam": FakeCamera()}
    obs = robot.get_observation()

    check("observation keys == observation_features keys",
          set(obs) == set(feats),
          f"missing={sorted(set(feats) - set(obs))} extra={sorted(set(obs) - set(feats))}")
    check("joint values are plain floats",
          all(isinstance(obs[n], float) for n in JOINT_NAMES))
    check("camera value is an (H, W, 3) uint8 array",
          isinstance(obs["cam"], np.ndarray) and obs["cam"].shape == (480, 640, 3)
          and obs["cam"].dtype == np.uint8,
          f"got {type(obs['cam']).__name__} "
          f"{getattr(obs['cam'], 'shape', None)} {getattr(obs['cam'], 'dtype', None)}")
    check("joint values look like the fed motion",
          -1.0 < obs["q_shoulder"] < 0.0 and 0.5 < obs["q_elbow"] < 1.3,
          f"q={[round(obs[n], 4) for n in JOINT_NAMES]}")

    # Teleoperator must produce exactly the robot's action keys.
    teleop_cfg = ArmTelemetryConfig(port="COM_FAKE")
    teleop = ArmTelemetry(teleop_cfg)
    teleop._reader = fake
    check("teleop action_features == robot action_features",
          teleop.action_features == robot.action_features,
          f"{teleop.action_features} vs {robot.action_features}")

    action = teleop.get_action()
    check("get_action keys match action_features",
          set(action) == set(robot.action_features),
          f"got {sorted(action)}")
    check("teleop reused the robot's snapshot (same instant)",
          teleop.reused_snapshots == 1 and teleop.fresh_snapshots == 0,
          f"reused={teleop.reused_snapshots} fresh={teleop.fresh_snapshots}")
    check("action is the commanded target, offset from the measurement",
          abs(action["q_shoulder"] - obs["q_shoulder"] - 0.004) < 2e-3,
          f"target-measured={action['q_shoulder'] - obs['q_shoulder']:+.5f} (fed +0.004)")

    # Stale telemetry must be refused rather than silently recorded.
    stale = FakeReader()
    feed_recent_telemetry(stale)
    time.sleep(1.2)
    robot._reader = stale
    try:
        robot.get_observation()
        check("stale telemetry raises instead of recording garbage", False,
              "no exception raised")
    except TimeoutError:
        check("stale telemetry raises instead of recording garbage", True)


def test_dataset_roundtrip() -> None:
    print("3. LeRobot must accept the schema and round-trip a dataset")
    from lerobot.utils.feature_utils import hw_to_dataset_features

    from arm_lerobot.lerobot_camera import GeminiCameraConfig
    from arm_lerobot.lerobot_robot import ArmRobot, ArmRobotConfig, JOINT_NAMES

    robot = ArmRobot(ArmRobotConfig(port="COM_FAKE", cameras={"cam": GeminiCameraConfig()}))

    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    act_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    features = {**obs_features, **act_features}

    print("    resulting dataset schema:")
    for key, spec in features.items():
        print(f"      {key:32s} dtype={spec['dtype']:6s} shape={spec['shape']} "
              f"names={spec.get('names')}")

    check("observation.state has 3 names",
          features["observation.state"]["names"] == list(JOINT_NAMES))
    check("action has 3 names", features["action"]["names"] == list(JOINT_NAMES))
    check("image became a video feature with an RGB shape",
          features["observation.images.cam"]["shape"] == (480, 640, 3)
          and features["observation.images.cam"]["dtype"] == "video")
    check("image is NOT flagged as a depth map",
          features["observation.images.cam"]["info"]["is_depth_map"] is False)

    # Now actually write and read back a tiny dataset.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = REPO / "outputs" / "smoke_dataset"
    if root.exists():
        import shutil

        shutil.rmtree(root)

    fps = 30
    n_frames = 12
    try:
        dataset = LeRobotDataset.create(
            repo_id="local/arm27_smoke",
            fps=fps,
            features=features,
            root=root,
            robot_type="arm_27",
            use_videos=True,
        )
    except Exception as exc:
        check("LeRobotDataset.create accepts the schema", False, f"{type(exc).__name__}: {exc}")
        return

    check("LeRobotDataset.create accepts the schema", True)

    cam = FakeCamera()
    rng = np.random.default_rng(0)
    try:
        for i in range(n_frames):
            state = np.array([0.1 * i, -0.5 + 0.01 * i, 0.9 + 0.005 * i], dtype=np.float32)
            action = state + 0.005
            dataset.add_frame({
                "observation.state": state,
                "observation.images.cam": cam.async_read(),
                "action": action,
                "task": "smoke test: push the block",
            })
        dataset.save_episode()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
    except Exception as exc:
        check(f"add_frame/save_episode accepted {n_frames} frames", False,
              f"{type(exc).__name__}: {exc}")
        return

    check(f"add_frame/save_episode accepted {n_frames} frames", True)

    try:
        reopened = LeRobotDataset("local/arm27_smoke", root=root)
        check("dataset re-opens from disk", len(reopened) == n_frames,
              f"len={len(reopened)}")
        item = reopened[0]
        check("read-back frame exposes state / action / image",
              "observation.state" in item and "action" in item
              and "observation.images.cam" in item,
              f"keys={sorted(k for k in item if not k.startswith('_'))}")
        state0 = np.asarray(item["observation.state"]).ravel()
        check("read-back state matches what was written",
              state0.shape == (3,) and abs(float(state0[1]) - (-0.5)) < 1e-5,
              f"state[0]={state0}")
    except Exception as exc:
        check("dataset re-opens from disk", False, f"{type(exc).__name__}: {exc}")

    size_mb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6
    print(f"    dataset on disk: {size_mb:.2f} MB for {n_frames} frames "
          f"({size_mb * 1024 / n_frames:.1f} KB/frame at 640x480)")


def main() -> int:
    print("LeRobot adapter smoke test (no hardware required)")
    print(f"  HF_HOME         = {os.environ.get('HF_HOME')}")
    print(f"  HF_LEROBOT_HOME = {os.environ.get('HF_LEROBOT_HOME')}")
    print()
    test_abstract_interfaces()
    print()
    test_feature_consistency()
    print()
    test_dataset_roundtrip()
    print()
    if FAILURES:
        print(f"SMOKE TEST FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
