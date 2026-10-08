# LeRobot × 27_engineer 三轴达妙机械臂 × Orbbec Gemini 2

在 PC 上用 LeRobot 完成 **遥操作采集 → 训练 ACT → 自主推理** 的最小闭环。

**架构一句话**：STM32H723 保留实时控制（1 kHz MIT 伺服、重力前馈、软限位、肩肘联锁），
PC 只做 LeRobot 该做的事（相机、数据集、策略）。两者之间走 UART7@921600。

> 📄 必读：[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)（需求与架构决策，含被否决的方案与原因）
> 📄 环境实况：[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)（每一条都是实测，含 6 个已踩的坑）

---

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| **1a** | 环境、相机取流、遥测解析、时间对齐 | ✅ **已完成并验证** |
| **1b** | 只读采集录成 LeRobotDataset | ⏳ 代码就绪，**差接上 H7 实机** |
| **2** | 反向通道（PC→H7 目标角 + 超时失能） | ⏳ 需先出固件方案 |
| **3** | 租云卡训练 ACT、部署 | ⏳ |

### 已验证的事实（全部有测试支撑）

| 项 | 结果 | 验证方式 |
|---|---|---|
| Orbbec Gemini 2 取流 | **29.95 Hz**，抖动 p95−p50 = **1.5 ms**，深度 ~77% 有效 | `tools/probe_camera.py` |
| UART7 遥测解析 | 27 字段 schema 与固件格式串逐字段对齐 | `tools/probe_telemetry.py --self-test` |
| 20↔30 Hz 时间对齐 | 最差关节误差 **1.6e-3 rad ≈ 末端 0.5 mm** | `tools/test_sync.py`（6 套） |
| LeRobot 适配层 | 抽象接口、特征一致性、**数据集真实往返**全部通过 | `tools/test_pipeline_smoke.py` |
| CLI 类型发现 | `--robot.type=arm_27` / `--teleop.type=arm_telemetry` 可用 | 真实 `lerobot-record` |

### ⚠️ 硬件时间戳不可用

Gemini 2 的 `get_timestamp_us` 在本机返回常数 `2^33`（哨兵值），彩色与深度相同。
原因是 Windows 需要管理员运行 `obsensor_metadata_win10.ps1 -op install_all` 注册 UVC 元数据设备，
且**每换一个设备都要重跑**。

**这不影响方案**：相机设备时钟与 STM32 的 `t_ms` 本来就是两个独立时钟，
对齐注定要在 PC 宿主时钟上做。系统已按此设计，并用 `Frame.timestamps_look_real()` 显式识别该哨兵值。

---

## 快速开始

### 在新设备上拉起（克隆之后第一步）

```powershell
git clone https://github.com/Yang9008-max/lerobot-arm27.git
cd lerobot-arm27

# 一条命令建好整个环境（幂等，可重复跑）：venv + LeRobot + 相机 SDK + 本项目 editable 安装
pwsh -ExecutionPolicy Bypass -File tools\setup_env.ps1

# 每个新 shell 先做这个（把缓存全部重定向出 C 盘，并自动处理 Clash 代理）
. .\tools\env.ps1
```

要点：

- **仓库根目录是自动推导的**（从脚本自身位置），克隆到任何盘/任何目录都能跑，
  不需要改脚本。要覆盖时设 `$env:ARM_LEROBOT_ROOT`。
- 基础解释器默认找 `D:\Python\Python313\python.exe`；换机器时用 `$env:ARM_PYTHON`
  指定一个 **Python 3.12+**（3.13 最好，理由见 `docs/ENVIRONMENT.md` §1）。
- `.venv` 和 `.cache` **不在仓库里**（1.6 GB + 0.5 GB），由 `setup_env.ps1` 重建。
- 仓库里带了 `data/arm27_push`（5 集真实采集）；其它 `data/*` 是早期的探针/测速集，没有上传。
- 全部实测环境事实（版本、代理陷阱、相机档位、踩过的坑）见 `docs/ENVIRONMENT.md`。

```powershell
# 每个新 shell 先做这个（把缓存全部重定向出只剩 3 GB 的 C 盘，并自动处理 Clash 代理）
. .\tools\env.ps1
```

### 离线验证（不需要任何硬件）

```powershell
.venv\Scripts\python.exe tools\probe_telemetry.py --self-test   # 解析器 + 运动学
.venv\Scripts\python.exe tools\test_sync.py                     # 时间对齐（6 套）
.venv\Scripts\python.exe tools\test_pipeline_smoke.py           # LeRobot 适配 + 数据集往返
```

### 相机

```powershell
.venv\Scripts\python.exe tools\probe_camera.py --list
.venv\Scripts\python.exe tools\probe_camera.py --profiles
.venv\Scripts\python.exe tools\probe_camera.py --seconds 5 --save
```

### 遥测（需接上 H7 并上电）

```powershell
.venv\Scripts\python.exe tools\probe_telemetry.py --list
.venv\Scripts\python.exe tools\probe_telemetry.py --autodetect
.venv\Scripts\python.exe tools\probe_telemetry.py COM8 --seconds 15
```

`--autodetect` 会扫描所有 COM 口找出发送 `watch,` / `kin,` 行的那个——
本机有 5 个 CH340，靠名字猜是没用的。

---

## 采集数据（阶段 1b）

> ❗ `--robot.discover_packages_path=arm_lerobot` **不能省**。
> LeRobot 的第三方插件自动发现只认 dist 名以 `lerobot_robot_` / `lerobot_camera_` / `lerobot_teleoperator_`
> 开头的包，而本项目的 dist 叫 `arm-lerobot`，不匹配。

```powershell
. D:\MyTrain\LeRobot\tools\env.ps1

.venv\Scripts\python.exe -m lerobot.scripts.lerobot_record `
  --robot.type=arm_27 `
  --robot.discover_packages_path=arm_lerobot `
  --robot.port=COM8 `
  --robot.cameras="{ front: {type: gemini, width: 640, height: 480, fps: 30} }" `
  --teleop.type=arm_telemetry `
  --teleop.discover_packages_path=arm_lerobot `
  --teleop.port=COM8 `
  --dataset.repo_id=local/arm27_push_block `
  --dataset.root=D:\MyTrain\LeRobot\data\arm27_push_block `
  --dataset.fps=30 `
  --dataset.num_episodes=30 `
  --dataset.episode_time_s=20 `
  --dataset.reset_time_s=10 `
  --dataset.single_task="push the block into the target area" `
  --play_sounds=false
```

> ❗ **`--play_sounds=false` 必须加。** LeRobot 用 PowerShell 的 `System.Speech` 做语音提示，
> 本机加载该程序集失败（`CalledProcessError`），而且它**会让整个 record 进程崩溃**——
> 实测在一切其他步骤都正常的情况下，收尾播报 "Stop recording" 时直接把进程带崩。

> ❗ `--robot.cameras=` 的值是 draccus 的内联 YAML，引号必须保留。
> 不加 `--robot.cameras=` 也能跑（数据集里就没有图像），可用来单独验证串口链路。

**采集前必须做的三件事：**
1. **底盘锁死**。臂在工程车上，笔记本用 USB-TTL 拴着，相机线还跟着臂走。
2. **摆好相机**。它现在朝向人和桌面，不是工作面。
3. **确认右三档打到 UP**（臂使能档），否则动作源是死的。

**采集中如果看到 `TimeoutError: no usable arm telemetry`，说明遥测超过 0.5 s 没更新。**
这是**故意的硬失败**：把陈旧关节角和新鲜图像配在一起，会得到一个看起来正常、
实际把策略训坏的数据集。宁可丢一集。

---

## 目录

```
arm_lerobot/                  (src\arm_lerobot)
├── arm_model.py        运动学模型与关节限位，常数逐条对齐固件
├── telemetry.py        UART7 报文解析 + 后台读取 + 共享 reader 注册表
├── sync.py             ClockMapper（去抖动的 t_ms→宿主时钟映射）+ 插值重采样
├── camera.py           Orbbec Gemini 2 设备封装（RGB + 对齐深度）
├── lerobot_camera.py   GeminiCameraConfig / GeminiCamera   → --camera.type=gemini
├── lerobot_robot.py    ArmRobotConfig / ArmRobot           → --robot.type=arm_27
└── lerobot_teleop.py   ArmTelemetryConfig / ArmTelemetry   → --teleop.type=arm_telemetry

tools\
├── env.ps1               每个 shell 先 source：缓存重定向 + Clash 代理自动探测
├── setup_env.ps1         幂等的环境搭建脚本
├── probe_camera.py       相机探针（设备/档位/帧率/时间戳/深度质量）
├── probe_telemetry.py    遥测探针 + 离线自测
├── test_sync.py          时间对齐离线测试（合成真值）
├── test_pipeline_smoke.py LeRobot 适配层 + 数据集往返冒烟测试
├── record_episode.py     无头单集录制（等使能 + 等真的动起来 两道触发）
└── inspect_dataset.py    数据集体检：时序、state/action、限位，并把视频解回 PNG
```

---

## 设计要点（为什么这么做）

### 1. 阶段 1 零固件改动

`watch` 遥测行里**同时**有：

- `qb/qa/qe` —— 反馈关节角 → **observation**
- `qbt/qat/qet` —— 目标关节角 → **action**

而 H7 本来就在用 HT-10A 驱动机械臂。所以 PC **只读**就能完成采集，
`rxNop` 那个空函数完全不用动。反向通道只在阶段 2 推理时才需要。

副产品：**记录下来的 action 与阶段 2 要下发给 H7 的量是同一个物理量**，
所以训练时的动作空间和推理时的指令天然一致。

### 2. 遥操作的 leader 就是 HT-10A 本身

`ArmTelemetry` 不做任何映射，它只把 H7 已经算好的目标角暴露成 action。
所以**手感不用改、不用换手柄、不用做 USB HID**。

### 3. 深度的定位

深度**不声明为 observation feature**。因为 `hw_to_dataset_features` 会把任何 `(H, W, 1)`
变成 `observation.images.*` 且 `is_depth_map=True`，而 `dataset_to_policy_features`
把所有 image/video 一律归为 `FeatureType.VISUAL` —— **声明了就会被 ACT 当第二路相机吃掉**。

深度通过 `GeminiCamera.async_read_depth()` 提供（float32 **米**，因为 LeRobot 按 dtype 推断单位），
用于实时后处理（成功判定、物体定位），不进数据集。这样数据集干净、ACT 只吃 RGB+state。

### 4. 时间对齐：不信任到达时间

`t_ms` 是干净的，USB 到达时间带抖动。所以先用滑动窗口最小二乘把 `t_ms` 映射回宿主时钟
（实测把 ~1.5 ms 抖动压到 0.05 ms 的偏移误差），再在 `t_ms` 空间插值。

相机 30 Hz 对遥测 20 Hz，导致**约 95% 的帧落在两条遥测线之间或之后**。
早期版本把这种情况"按住上一帧的值"，实测误差 `2.35e-2 rad`（= 关节速度 × 50 ms，235 个编码器刻度）。
改为沿最后区间**线性外推**后降到 `1.6e-3 rad`，**改善 14.7 倍**。

### 5. Robot 与 Teleop 共享一个串口

LeRobot 把两者构造成两个独立对象，但本项目中两者都需要同一份遥测
（Robot 出观测、Teleop 出动作）。两个 reader 抢一个 COM 口会各拿到一半的行，
所以用**带引用计数的进程内共享注册表**。

而且 record 循环是 `get_observation()` 之后才 `get_action()`，两者时刻不同。
LeRobot 没有传递时间戳的钩子，所以最近一次对齐快照缓存在共享 reader 上，让 action 复用 observation 的时刻。

---

## 已知取舍（不要"修"）

| 现象 | 说明 |
|---|---|
| `pip check` 报 `pyorbbecsdk2 requires opencv-python` | 有意为之。用 `--no-deps` 装相机 SDK，避免与 LeRobot 的 `opencv-python-headless` 互相覆盖 `cv2` |
| `pip check` 报 `av==13.0.0` 冲突 | 真实冲突但无害。`av` 只被 Orbbec 示例用于录像，SDK 核心不依赖。**不要降到 13**，那会弄坏 LeRobot |
| `torchcodec` 加载失败告警 | FFmpeg DLL 缺失，LeRobot **自动回落 pyav**，不影响使用 |
| 相机硬件时间戳是常数 `2^33` | 见上文，不影响方案 |
| `CameraConfig` 与 `OrbbecCamera` 的配置类重名 | 适配层用 `CaptureConfig` 别名导入 |

## 无头采集（推荐用于验证与批量采集）

`lerobot-record` 交互式、要靠人按回车，而且**不知道臂是否使能**——失能时固件把目标强制成
`(0,0,0)`（实测确认），录进去的 action 全是垃圾。`record_episode.py` 是它的可脚本化替代，
用的是**同一套 LeRobot API**（`hw_to_dataset_features` / `build_dataset_frame` / `LeRobotDataset`），
所以数据集格式完全一致。

```powershell
# 无相机：只验证机械臂这一侧（相机还没装到末端时用这个）
.venv\Scripts\python.exe tools\record_episode.py `
  --no-camera --seconds 15 `
  --repo-id local/arm27_motion --root D:\MyTrain\LeRobot\data\arm27_motion

# 带相机（分辨率就用 640x480，理由见下表）
.venv\Scripts\python.exe tools\record_episode.py `
  --camera front --width 640 --height 480 --seconds 20 `
  --repo-id local/arm27_push --root D:\MyTrain\LeRobot\data\arm27_push
```

### ⚠️ 分辨率与帧率必须按实测声明（实测定案）

LeRobot 的时间戳按 `timestamp = frame_index / fps` 生成，所以**声明的 fps 必须等于循环真实达到的 fps**，
否则数据集的时间轴整体被缩放，策略学到的速度和现实不符。

| 分辨率 | 循环周期 | 实测帧率（多次） | 结论 |
|---|---|---|---|
| **640×480** | 34–36 ms | **27.8 – 29.2 fps** | ✅ 用这个，但**声明 `--fps 25`** |
| 1280×720 | 50 ms | **19.96 fps** | ❌ 声称 30 会让时钟快 1.50 倍，**数据不可用于训练** |

这台 Ryzen 5 5600U 在单进程里搬不动 1280×720（相机泵解码一次 + 写数据集再搬一次，GIL 下互相抢占），
即使在 640×480 下也守不住 30 fps。**`--fps 25` 给 10% 余量**；25 fps 对这条臂足够
（峰值约 0.5 rad/s，每帧只动 0.02 rad）。

`record_episode.py` 会在实际帧率低于声明值 5% 以上时**打印醒目告警并以退出码 1 结束**。

它比其他录制方式多三道保护，每一道都来自实测观察：

| 保护 | 为什么 |
|---|---|
| **等臂使能**才开始 | 失能时 `q*` 被固件强制成 `(0,0,0)`，action 无意义 |
| **使能后等 2 秒** | 使能会让臂 slew 到目标（实测底轴 0.239 rad @ ~0.95 rad/s） |
| **等关节真的动起来**才开始计时 | 否则每集开头都录进"人还没碰到摇杆"的死时间 |

第三道最关键：脚本一直在等，**检测到任一关节变化超过 `--motion-threshold`（默认 0.05 rad）才开录**，
所以你不用赶时间。调参用 `--wait-motion-s` / `--motion-threshold` / `--no-wait-motion`。

### 录完必须体检

```powershell
.venv\Scripts\python.exe tools\inspect_dataset.py `
  --root D:\MyTrain\LeRobot\data\arm27_motion --repo-id local/arm27_motion
```

检查项：时间戳是否均匀、`state` 与 `action` 是否真的不同（若相同说明动作源接错了）、
关节是否越限，**并把录进去的视频解码回 PNG 写进 `outputs/`**，让人眼确认没有损坏。

> ⚠️ **没有图像的数据集不能训练 ACT。** ACT 要求至少一路 image 或 `env_state`
> （`policies/act/configuration_act.py`）。所以无相机采集**只用于验证遥测链路与时间对齐**，
> 正式训练数据必须等相机装到末端之后重采。

## 下一步

1. **接上 H7 实机验证遥测**（`probe_telemetry.py --autodetect`），拿到真实 COM 口
2. 采 5 条试跑，用 `lerobot-dataset-viz` 回放确认图像与关节角对齐
3. 采 30–50 条，传云卡训 ACT
4. 阶段 2：出固件方案（UART7 加 CRC 目标角帧、`rxNop` 换解析器、超时失能、
   遥控器降级为使能/急停闸门），按 `AI_GUARDRAILS.md` 先确认再动手
