# 环境实况（已验证，非推测）

> 本文档记录的每一条都是**在本机实测**得出的，不是照抄文档。
> 验证时间：2026-10-07
> 配套文档：[REQUIREMENTS.md](REQUIREMENTS.md)（需求与架构决策）

---

## 1. 解释器与虚拟环境

| 项 | 值 |
|---|---|
| 基础解释器 | `D:\Python\Python313\python.exe` —— Python **3.13.5** |
| 虚拟环境 | `D:\MyTrain\LeRobot\.venv`（**1.59 GB**） |
| 项目 Python | `D:\MyTrain\LeRobot\.venv\Scripts\python.exe` |
| msys64 备用 | `D:\msys64\ucrt64\bin\python.exe` —— Python 3.12.11（**本项目未使用**） |

**为什么选 3.13 而不是 3.12**：`pyorbbecsdk2` 在 PyPI 上有 `cp313-cp313-win_amd64` wheel，而 msys64 那个 3.12 **没有相机 SDK**。LeRobot 0.6.1 要求 `python>=3.12`，3.13 合格。选 3.12 反而会丢掉相机。

## 2. 已安装版本

| 包 | 版本 | 备注 |
|---|---|---|
| **lerobot** | **0.6.1** | PyPI 最新；`requires_python >=3.12` |
| torch | **2.11.0+cpu** | `cuda=False`（本机 AMD 核显，符合预期） |
| torchvision | 0.26.0 | |
| numpy | 2.2.6 | Lerobot 约束 `<2.3` |
| opencv-python-headless | 4.13.0.92 | `cv2.__version__ == 4.13.0` |
| **pyorbbecsdk2** | **2.1.2** | venv 内 `import` 通过 |
| av | 15.1.0 | |
| pyserial | 3.5 | 读 UART7 用 |
| pygame | 2.6.1 | |
| datasets / accelerate / huggingface-hub | 4.8.5 / 1.15.0 / 1.33.0 | |

安装 extras：`lerobot[dataset,training,core-scripts,viz]`

## 3. ⚠️ 两处依赖冲突（**均为有意为之，不要"修"**）

### 3.1 `pip check` 会报 `pyorbbecsdk2 requires opencv-python, which is not installed`

**这是故意的。** `pyorbbecsdk2` 依赖 `opencv-python`，LeRobot 依赖 `opencv-python-headless`，**两者都提供同一个 `cv2` 包，一起装会互相覆盖文件**。

做法：用 `--no-deps` 安装 `pyorbbecsdk2`，只补它真正需要的依赖。
已验证：`cv2 4.13.0` 导入正常，相机取流正常。

### 3.2 `pyorbbecsdk2 requires av==13.0.0, but you have av 15.1.0`

**这是真实的版本冲突，但无害。**
- `pyorbbecsdk2` 对 py3.13 钉死 `av==13.0.0`
- LeRobot 要求 `av>=15.0.0`

`av` **只被 Orbbec 的示例脚本用于录像**，SDK 核心不依赖它；我们也不用。
已验证：`import pyorbbecsdk` 成功，取流成功。

> ❗ **不要把 `av` 降到 13** —— 那会弄坏 LeRobot 的视频编解码。

## 4. C 盘保护（必须遵守）

本机 **C 盘只剩 3.3 GB**，`pip` 和 HuggingFace 的缓存默认都写 `C:\Users\Administrator\.cache`，**不在第一步重定向就一定会装到一半爆盘**。

`tools/env.ps1` 已把这些全部重定向到 D 盘：

| 变量 | 值 |
|---|---|
| `PIP_CACHE_DIR` | `D:\MyTrain\LeRobot\.cache\pip` |
| `TMP` / `TEMP` | `D:\MyTrain\LeRobot\.cache\tmp` |
| `HF_HOME` | `D:\MyTrain\LeRobot\.cache\huggingface` |
| `TORCH_HOME` | `D:\MyTrain\LeRobot\.cache\torch` |

**实测结果：整个安装过程结束后 C 盘从 3.5 GB 只降到 3.3 GB（系统自身开销），D 盘用掉约 3.1 GB。重定向有效。**

## 5. 网络与代理（**这是个会咬人的坑**）

实测连通性：

| 端点 | 结果 |
|---|---|
| `pypi.tuna.tsinghua.edu.cn` | ✅ 49–281 ms，**最快，用这个** |
| `pypi.org` | ✅ 787 ms |
| `download.pytorch.org/whl/cpu` | ✅ 2424 ms |
| `huggingface.co` | ⚠️ **直连 DNS 失败**；开梯子后解析为 `198.18.0.6`（Clash fake-IP）并返回 200 |
| `hf-mirror.com` | ⚠️ DNS 通但请求超时 |
| `raw.githubusercontent.com` | ❌ DNS 失败 |

### 🚨 关键：Clash 是 **fake-IP 模式**

本机跑着 `verge-mihomo`（Clash Verge），监听 `127.0.0.1:7897`，系统代理已开。

- PowerShell / WinINET **会**走系统代理，所以能通
- **Python 的 `requests` / `pip` / `huggingface_hub` 不读 Windows 系统代理**，它们会拿着 `198.18.x.x` 这个**假 IP 去直连，然后必然超时**

因此 `tools/env.ps1` 会自动探测代理端口并导出 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`；**梯子没开时自动跳过**，不会把环境弄死。

### HuggingFace 的默认策略

`HF_HUB_OFFLINE=1`（默认离线）。原因：HF 只有在梯子开着时才可达，而**采集数据跑到一半因为一次网络调用挂死，比不能用 Hub 糟糕得多**。
需要 Hub 时：`$env:LEROBOT_HF_OFFLINE='0'` 后再 dot-source `env.ps1`。

本项目的设计与 Hub 解耦：数据集用**本地 root 路径**，训练在云卡机器上做，模型用文件传输。

## 6. 相机实况（Orbbec Gemini 2）

| 项 | 实测值 |
|---|---|
| 型号 / 序列号 | Orbbec Gemini 2 / `AY6V163008W` |
| VID:PID | `0x2BC5:0x0670` |
| 连接 | **USB3.0** |
| 彩色可用档位 | `640x480@30 RGB`（另有 1280x720@30、1920x1080@30 RGB） |
| 深度可用档位 | **1280x800 / 640x400 / 320x200**，格式 `Y16` / `Y14` / `RLE` |
| **当前配置** | 彩色 640x480@30 + 深度 **640x400@30**，深度对齐到彩色 → 输出 640×480 |
| **实测帧率** | **29.95 Hz**，超时 0，丢帧 0 |
| **抖动** | 帧间均值 32.8 ms，p95−p50 = **1.5 ms**，max 110 ms（偶发一次） |
| 深度质量 | 有效率 ~74–80%，典型 175 mm ~ 12 m，中位 ~520 mm |
| `depth_scale` | **1.0**（raw 单位即 mm） |

### 6.1 为什么深度选 640×400 而不是设备默认的 1280×800

对齐后输出固定是彩色的 640×480，多出来的 2.56 倍像素纯属浪费。
**实测：降到 640×400 后帧间抖动从 2.4 ms 改善到 1.5 ms，数据质量无损失。**

### 6.2 🚨 硬件时间戳不可用（已实测确认）

```
timestamp API: get_timestamp_us
color ts     : 8589934592 us      ← 第一次测量
depth ts     : 8589934592 us
color ts     : 12884901888 us     ← 第二次测量（换了分辨率）
depth ts     : 12884901888 us
```

两次的值都满足：**彩色与深度完全相同**，且**都是 2³² 的整数倍**（`2×2³²` 与 `3×2³²`，低 32 位恒为 0）。

> ⚠️ **不要用固定值去比较。** 它并非常数——早期版本把检查写成硬编码比较 `2³³`，
> 结果第二次测量（`3×2³²`）会被**误判成可用**。正确做法是同时检查
> 「彩色 == 深度」和「低 32 位为 0」，权威判据则是 `CameraStats.timestamp_fps()`
> （它有帧历史，能看出这个值根本不在推进）。已修复。

原因（官方文档确认）：Windows 下通过 UVC 拿设备元数据，必须**以管理员权限**运行
`pyorbbecsdk\shared\obsensor_metadata_win10.ps1 -op install_all` 完成注册，
而且**每接入一个新设备都要重新跑一次**。

**决策：暂不修。** 因为——
1. 需要管理员权限，且换设备就要重来；
2. **更根本的是，相机的设备时钟与 STM32 的 `t_ms` 本来就是两个独立时钟，没有共同基准，对齐注定要在 PC 宿主时钟上做**，硬件时间戳只是锦上添花。

对齐方案因此是：**统一用宿主 `time.monotonic()` 打时间戳**，遥测另记 STM32 `t_ms` 用于抖动校正。
`Frame.timestamps_look_real()` 会显式识别这个哨兵值，防止下游误信一个永不前进的时钟。

### 6.3 一个已知的库陷阱

官方 `pyorbbecsdk/examples/utils.py:139` 用 `np.resize(data, (h, w, 3))` 解码图像。
**`np.resize` 在尺寸不匹配时会重复数据而不是报错**，会静默产出看起来正常的垃圾图。
我们的 `camera.py` 改用 `reshape` + **显式长度校验**，宁可抛异常也不悄悄出错。

## 7. 遥测链路实况（UART7）

| 项 | 值 | 证据 |
|---|---|---|
| 物理层 | UART7 @ **921600** 8N1，ASCII 行 | `arm.h:398` |
| 速率 | **20 Hz**（`ARM_KIN_UART_HZ`） | `arm.h:402` |
| 每周期两行 | `kin`（17 字段）/ `watch`（**27 字段**） | `arm.cpp:56-58` |
| 缩放 | 角度 ×1e4、长度 ×10、角速度/力矩 ×1e3（因 `nano.specs` 无 `%f`） | `arm.cpp:30-39` |
| **action 来源** | `qbt/qat/qet` = 目标关节角 | `arm.cpp:79-81` |
| **observation 来源** | `qb/qa/qe` = 反馈关节角 | `arm.cpp:76-78` |
| PC 侧现成解析 | `tools/arm_watch.py:24-29`（本机已跑通） | |

> **关键设计事实：observation 与 action 在同一行里白送。** 这是阶段 1「零固件改动只读采集」得以成立的根本原因。

解析器已用**离线自测**验证：28 token 边界、8 类畸形输入拒绝、IK/FK 往返自洽、肩肘联锁判定。
实机验证仍需接上 H7（见 `tools/probe_telemetry.py`）。

## 8. 复现命令

```powershell
# 每个新 shell 先做这个
. D:\MyTrain\LeRobot\tools\env.ps1

# 全新安装（幂等，可重复跑）
pwsh -ExecutionPolicy Bypass -File D:\MyTrain\LeRobot\tools\setup_env.ps1

# 离线验证解析器与运动学（不需要硬件）
.venv\Scripts\python.exe tools\probe_telemetry.py --self-test

# 相机
.venv\Scripts\python.exe tools\probe_camera.py --list
.venv\Scripts\python.exe tools\probe_camera.py --profiles
.venv\Scripts\python.exe tools\probe_camera.py --seconds 5 --save

# 遥测（需接上 H7 并通电）
.venv\Scripts\python.exe tools\probe_telemetry.py --list
.venv\Scripts\python.exe tools\probe_telemetry.py --autodetect
```

## 9. 踩过的坑（避免重复）

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `pwsh` 报 `0xC0000142` 无输出 | **`workspace-write` 沙箱下 shell 无法启动**；已用诊断脚本排除 ACL（判定 `NOT_THIS_CLASS`） | 每条命令走一次性提权，或把会话切到完全权限 |
| 2 | 脚本报"未对文件进行数字签名" | PowerShell 执行策略 | `Set-ExecutionPolicy -Scope Process Bypass` |
| 3 | `pip install` 卡死，CPU 增量 0 | **进度条 `\r` 写满管道，读取端未排空 → 死锁** | `--progress-bar off`，输出重定向到文件 |
| 4 | `OBError: NULL pointer passed for argument "deviceMgr"` | `Context()` 临时对象被 GC，SDK 的 device manager 随之销毁 | 持有 `Context` 引用直到用完 device list |
| 5 | `CameraConfig.depth_*` 配置不生效 | `_pick_profile` 在 `fmt=None` 时直接走设备默认 | 扫描 profile 列表自行匹配 |
| 6 | 担心 `av` / `opencv` 依赖告警 | 见 §3，均为设计取舍 | 不要按 `pip check` 去"修" |
