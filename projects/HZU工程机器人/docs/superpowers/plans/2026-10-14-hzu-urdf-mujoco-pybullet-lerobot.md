# HZU URDF、MuJoCo / PyBullet 与 LeRobot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从现有机械设计资料恢复可追溯的机器人模型；根据用户对缺失外形的纠正，优先交付包含电机、底盘、四轮与机械臂的整机几何/运动学 URDF，再完善碰撞、动力学、MuJoCo / PyBullet 控制及本地 LeRobot 数据采集。

**Architecture:** 原始 CAD → 文件索引与几何库 → 连杆位姿/关节证据配置 → URDF 和 MJCF → 两个仿真器共用的关节控制接口 → LeRobot 数据集。各阶段保存机器可读结果，缺失几何和未确认参数具有明确状态；事实测量与估值分开。

**Tech Stack:** Windows PowerShell；已安装 Python 3.13.5、CadQuery 2.8.0、OpenCascade 7.9.3.1.1、NumPy 2.4.6、Pillow 12.3.0、py7zr 1.1.3；MuJoCo 已安装，版本由环境报告记录。PyBullet、LeRobot 在仿真接入阶段使用独立环境安装并锁定实际版本。

## Global Constraints

**用户纠正后的当前交付范围（2026-10-15）:** 不能把缺失电机、底盘和轮组的局部显示包当作所需机器人模型交付。优先从原始 SLDPRT 显示缓存恢复电机、底盘和轮组外形，结合已有原生全装配位姿生成包含底盘、四轮、机械臂及电机的 URDF。先完成可查看、可核对坐标与活动轴的整机运动学描述；真实质量/限位/麦轮接触、被动机构和执行器标定单独记录，不能把外形恢复成功称为动力学验收完成。

本轮步骤：① 固定 MIT 读取器 `blussyya/sldprt-format-research` 的版本并批量只读导出原生米制网格、保留解析诊断；② 按实例比对原生包围盒、核对电机/轮轴，更新几何覆盖；③ 将底盘、四个轮输出、机械臂输出和电机外壳分组，保留额外传动/被动机构简化状态；④ 实际导出 URDF/相对网格路径，检查图结构、文件与米制尺度，加载渲染并验证关节运动；⑤ 交付真实文件和整机图，列出仍未验证的物理参数。

**本轮完成（2026-10-15）:** 已导出 `models/hzu_robot/hzu_visual.urdf`。578 个显示实例/188 项复用网格/11 个 link，含全部十台主要电机外形；六个机械臂输出关节和四个轮轴实际由 MuJoCo 3.14.0 加载。27 组姿态、全部导出顶点读回、192 个来源文件哈希与 12 项轮轴同轴检查通过。保存总装中可见机器人实例均已记账；仅排除相机视场锥、UWB 遮挡体及场外矿石示意。八个减震器、两个 8009 和电磁阀用已有 STEP 小装配替代缺失的子网格。完整说明、覆盖清单、假设、验证与渲染在 `models/hzu_robot/`。这是几何/输出侧运动学交付，未完成碰撞、真实惯量/限位、被动机构约束、驱动映射、PyBullet 或 LeRobot 验收。

- 用户没有 SolidWorks，也无法联系原作者。
- 目标包含 MuJoCo、PyBullet、LeRobot；当前优先保证整机几何覆盖，再完善物理与控制。
- 原始 CAD 文件作为只读输入，输出统一放在新建目录。
- CAD 几何分析使用 mm；模型、动作和仿真使用 m、rad、s、kg。转换显式写入元数据。
- 关节数量、轴线、零位和实际限位由证据决定，不能按文件名预设自由度。
- 从几何外形计算的体积/单位密度惯量不是已验证的真实质量/惯量。
- 商业读取器的安装、版本兼容、授权和导出权限作为条件分支核实；不依赖商业工具才能开始工作。
- 第一阶段关节动作采用输出侧绝对位置目标，状态采用实际关节位置；关节名称、顺序和单位在两个仿真器及 LeRobot 中一致。
- LeRobot 数据保存在本地，数据集创建与上传是独立操作。
- 分析和几何索引可以完成，但准确模型交付依赖关节证据闭合。未确认项不伪装成已确认。

## 当前输入与基线

- `HZU工程完整装配.SLDASM`：整机装配入口。
- `机械臂1/机械臂装配.SLDASM`、底座/大臂/小臂分装：机械臂入口。
- `末端零件/末端装配.SLDASM`：末端入口。
- `底架/底架装配.SLDASM`、`轮组/轮组装配L.SLDASM`、`轮组/轮组装配R.SLDASM`：整机扩展输入。
- 已展开 80 个 STEP/STP，其中 76 个没有装配实例；其他为电机、阀件和避震器小装配。
- 已成功读取 8 个 STEP/STP 的实体几何。ZIP/7z 已检查，没有发现现成 URDF。
- 完整可行性分析见 `docs/urdf-route-analysis-2026-10-14.md`。

## 文件结构与职责

```text
scripts/urdf_pipeline/
  inventory.py             文件索引、重复文件、STEP 结构摘要、候选文件映射
  geometry.py              读取 STEP、单位转换、STL、孔轴候选及几何属性
  assemble.py              按组件位姿变换并合并刚体几何
  build_descriptions.py    证据配置 → URDF、MJCF、关节顺序和模型清单
  validate_model.py        图结构、路径、单位、惯性和运动学验证
  sim.py                   MuJoCo/PyBullet 统一接口及离线回放
  record_lerobot.py        仿真轨迹 → 本地 LeRobotDataset
tests/urdf_pipeline/
  test_geometry.py         合成几何的比例、体积、惯量和轴线验证
  test_model.py            链结构、局部位姿及跨仿真器运动学验证
  test_dataset.py          帧数、时间、状态/动作顺序和回放验证
urdf_work/
  evidence/inventory.json  源文件与哈希、结构摘要、歧义映射
  evidence/environment.json 实际 Python 与依赖版本
  evidence/geometry.json   每项几何的结果、错误与源文件引用
  evidence/geometry.csv    便于人工检查的几何摘要
  evidence/reconstruction.md 运动链、参数证据和缺失项
  assets/part_meshes/      按内容哈希命名的米制源零件网格
  config/assembly.json    装配变换和刚体分组
  config/joints.json      关节及其证据
  config/physics.json     质量、惯量和动力学参数及可信状态
  config/cameras.json     相机安装变换与渲染配置
models/hzu_arm/
  robot.urdf              机械臂通用 URDF
  robot.xml               MuJoCo MJCF
  joint_order.json        唯一关节顺序与动作语义
  meshes/                 连杆局部坐标下的网格
  validation.json         模型加载、FK 与仿真检查结果
models/hzu_robot/         验证后的整机模型
datasets/hzu_arm_sim/     本地 LeRobot 数据集
```

源零件网格用于检查和重建，不能直接等同于已完成的机器人连杆网格。

## Task 1：建立可追溯文件与几何索引（立即执行）

**Files:** Create `scripts/urdf_pipeline/inventory.py`、`geometry.py`、`tests/urdf_pipeline/test_geometry.py`；生成 `urdf_work/evidence/` 和 `urdf_work/assets/part_meshes/`。

**Interfaces:**
- `build_inventory(root: Path) -> dict`：输出 `schema_version`、`files`、`step_files`、`duplicate_groups`、`cad_matches`。
- 每个文件记录相对路径、字节数、扩展名和完整 SHA-256；几何任务按完整 SHA-256 去重。
- `analyze_shape(shape: cq.Shape) -> dict`：输入 OpenCascade 默认 mm 制 shape；输出包围盒、重心、体积、单位密度惯量、圆柱/圆弧候选。
- `process_step(path: Path, mesh_path: Path) -> dict`：输出分析结果并生成米制 binary STL；异常成为独立失败记录。
- `read_stl_bounds(path: Path) -> dict`：独立读取 binary STL 的顶点范围，用于输出单位检查。

- [x] 索引原始目录，排除 `docs`、`scripts`、`tests`、`urdf_work`、`models`、`datasets`；保存环境版本。
- [x] 保留同名 CAD 文件的全部候选，只有唯一名称匹配才能标成唯一；重复内容使用哈希判断。
- [x] 提取 STEP 中 PRODUCT、装配实例、实体计数和导出器信息。
- [x] 用合成 10×20×30 mm 盒体验证 STL 为 0.01×0.02×0.03 m，体积为 6e-6 m³。
- [x] 用相同盒体验证单位密度惯量为对角 `[6.5e-10, 5e-10, 2.5e-10]` m⁵；圆柱测试验证轴方向与半径。
- [x] 批量读取去重 STEP，记录实体/面数、尺寸、圆柱/圆弧轴候选、体积和错误；每项完成后保存结果，以便中断后恢复。
- [x] 对几何记录与 STL 独立比较尺寸；失败项保留源文件和错误说明。

Run:
```powershell
& 'D:\Python\Python313\python.exe' scripts/urdf_pipeline/inventory.py
& 'D:\Python\Python313\python.exe' -m unittest discover -s tests/urdf_pipeline -p test_geometry.py -v
& 'D:\Python\Python313\python.exe' scripts/urdf_pipeline/geometry.py
```

**验收:** 所有原始 STEP 都有索引记录；相同内容共享一项几何结果；每项结果是成功或明确失败。厘米/毫米网格不会未经转换进入仿真。

**2026-10-14 执行结果:** 545 个源资料文件及 11 个锁文件完成索引；80 个 STEP 路径去重为 78 项，78 项全部成功导出米制 STL，最大 CAD/STL 包围盒坐标差约 0.0095 mm。4 项合成几何测试通过。源文件重新核对哈希后没有变化。产物说明见 `urdf_work/README.md`；候选装配轴线见 `urdf_work/evidence/reconstruction.md`。MuJoCo 实际版本为 3.14.0；PyBullet/LeRobot 尚未安装。

## Task 2：恢复装配位置，确定重建分支

**Files:** Create `urdf_work/evidence/reconstruction.md`、`urdf_work/config/assembly.json`。

**Interfaces:** `assembly.json` 每个组件具有 `instance_id`、`source_sha256`、`link`、`matrix_mm`（4×4）、`evidence`、`confidence`。矩阵作用为 `p_assembly_mm = matrix_mm @ p_part_mm`；旋转必须正交且 det≈1。

- [x] 尝试从机械臂原始装配读取组件与位姿；实测 swformat 可以读取保存的 XML 组件树和位姿，商业读取器保留为备用分支。
- [x] 保存全部 30 个原生装配的结构化组件表；逐一记录找不到的引用、嵌入式载荷及同名歧义。
- [x] 核对关键左右侧板、两套电机及支撑孔轴的位置；用 6 组测得的孔轴约束及实际 MuJoCo 渲染确认组件没有堆在原点。
- 无可用读取器或位姿恢复失败时使用孔/轴几何配准；本次已读取保存位姿，该兜底分支未触发。
- [x] 检查场景只显示名称/导入路径与局部包围盒均支持的已有几何；未获得网格的组件记录为缺失，不补造。
- [x] 记录所采用的原始 XML 引用、矩阵约定、方向和平移单位、局部包围盒误差与 6 组同轴检查残差；后续若使用几何配准仍需两个独立定位特征。

**验收:** 装配变换合法、来源可追溯，关键组件位置能够一致解释安装结构。若关键间隔无法确定，保存缺失项，继续可独立完成的几何分析，不宣称精确装配已恢复。

**2026-10-15 执行结果:** 原生 410 个文件全部成功拆流；409 个有组件/特征 XML，406 个有预览。机械臂恢复 124 个实例、整机恢复 646 个实例，所有保存变换均通过刚体检查。关键板件六组孔轴对齐通过，大臂候选两轴间距 300 mm；底座电机/输出轴间距 100 mm。7 个 3MF 全部导出米制 STL。18 个已有网格实例完成 MuJoCo 保存位姿渲染；这是无活动关节的检查场景。556 个源文件及锁文件哈希未变。结果见 `docs/urdf-phase2-native-recovery-2026-10-15.md`。刚体分组、关节参数和未覆盖的原生网格仍属于后续任务。

## Task 3：确认刚体分组和关节树

**Files:** Create `urdf_work/config/joints.json`；完善 `reconstruction.md`。

**Interfaces:** 每个关节记录 `name`、`parent`、`child`、`type`、`origin_xyz_m`、`origin_rpy_rad`、`axis`、`zero_definition`、`limits`、`transmission` 和逐参数证据。`origin` 表示父 link 到子 link 的零位变换；`axis` 用 joint 局部坐标表示。

- [ ] 依据相对运动把侧板、紧固件和壳体合并为刚体，固定零件无需各自创建活动关节。（已记录 7 个主要结构候选组；电机转定子、轴承内外圈、差动输入轮及紧固件尚未全部分配。）
- [x] 结合候选孔轴、电机输出轴、轴承和传动结构确认活动分界。（六轴输出主链已自主复核，22 项轴线检查通过；完整活动 CAD 配合仍未求解。）
- [ ] 明确输出侧轴与电机轴的区别，记录同步带/锥齿轮比及正负方向依据。
- [ ] 确定零位与机械运动范围；没有证据的限位写成“未确认”，模型生成器拒绝将其用于正式发布配置。
- [ ] 检查模型是否存在闭环/耦合；若有，选定树形主链并记录剩余约束在 MJCF/控制层的实现方案。
- [x] 冻结 `joint_order.json` 的名字顺序；关节数量由本阶段决定。（六个输出侧关节，硬件通道映射单独标定。）

**验收:** 一个根、无环、每个非根 link 有一个父 joint；所有活动轴均有证据；推断值仍保留来源与状态。

**2026-10-15 进展:** 6 轴候选与源证据已保存至 `urdf_work/config/joints.json`，输出侧候选为底座回转、大臂俯仰、小臂俯仰、前臂轴向旋转、腕部俯仰、吸盘自转。17 项轴线检查及 3 项合成测试通过。跨文档机械臂保存位姿 123/124 一致（UWB 碳管轴向姿态不同），末端 43/43 一致。两台 2006 的差动映射、限位与控制零位未冻结，当前不满足正式模型发布。审核表见 `docs/urdf-joint-review-2026-10-15.md`。

**自主复核补充:** 六轴输出主链与模型局部坐标已选定。新增腕部 5 项轴线检查（总 22 项），30 个网格实例组成的 6 关节 MuJoCo 检查模型通过 45 组独立正运动学对照和 6 项随动检查。打印网格实测 20/20 带轮、18/51 锥齿轮，理想腕部差动秩为 2；硬件编码器符号、内部减速和机械限位仍未标定。保存姿态仅作为模型参考零位。详见 `docs/urdf-joint-self-verification-2026-10-15.md`。可继续生成标注用途的运动学草稿，未宣称动力学与正式发布验收完成。

## Task 4：构建机械臂 URDF 与独立网格

**本轮执行范围（用户选择“按 link 整理网格”）:** 先完成独立 link 网格包，不将本轮交付称为已生成 URDF。新增 `assemble.py`（归属、坐标转换、STL/OBJ 导出）、`validate_link_meshes.py`（从文件读回、还原原装配与逐轴姿态、检查报告和渲染），以及 `test_link_meshes.py`（已知旋转/单位/法线/二进制往返）。复用已核查的 30 个显示实例，另加入有独立孔轴和 STEP 包围盒依据的底盘机械臂安装板；不自动纳入差动输入齿轮、合成轴承圈或未解码的原生电机外形。

产物为 `models/hzu_arm/meshes/visual/<link>.stl` 和对应 OBJ、`link_mesh_manifest.json`、覆盖/缺失清单，以及读回验证和预览。七个 link 的网格均使用各 link 局部米制坐标，后续 URDF visual origin 可用零值、scale 为 `1 1 1`。保留可表达的非退化输入面，仅对源零面积面或表示精度造成的塌缩面进行可追溯清理；不用布尔融合把相对运动件焊到一起，不计算未经依据的物理质量。验证源网格与 CAD 哈希、相对变换、三角形数量、法线、有限数值和单次单位转换；对零姿态及逐轴/组合姿态重新装配与原始实例变换对照。

**网格子项完成:** 7 个 link、31 个显示实例，导出 7 STL + 7 OBJ。990,192 个输入面移除 78 个有索引/原因记录的退化面后保留 990,114 面；5 项合成测试和 21 组全顶点重装配/运动检查通过，STL 最大误差 `1.514e-8 m`，OBJ `6.481e-11 m`。输出零面积面为 0。说明见 `docs/urdf-link-meshes-2026-10-15.md`，产物见 `models/hzu_arm/README.md`。本子项尚未生成 URDF。

**Files:** Create `assemble.py`、`build_descriptions.py`、`validate_model.py`、`tests/urdf_pipeline/test_model.py`；生成 `models/hzu_arm/robot.urdf` 和 `meshes/`。

**Interfaces:**
- `selected_components(e, joints, frames) -> list[dict]` 与 `assemble.py --config ...`：将已核查显示组件变换到 link 局部坐标并拼接输出，生成 `link_mesh_manifest.json`；几何输出不等同于封闭实体或完整质量组。
- `build_urdf(links: dict, joints: dict, output: Path) -> Path`：生成相对网格路径 URDF。
- `validate_description(path: Path) -> dict`：图结构、路径、有限数值、轴归一化、尺寸和参数状态检查。

- [ ] 加入合成两关节链测试，使用 90° 旋转检查末端坐标，避免矩阵顺序错误。
- [x] 连杆 visual 网格只在导出阶段缩放一次；URDF 网格 scale 使用 `1 1 1`。（源网格已为 m，本次仅换坐标；七个 link 的可用 origin/scale 写入清单。）
- [ ] 对 collision 使用简单几何或凸件；保留 visual 网格，不把螺纹/齿面全部带入接触计算。
- [ ] 生成 URDF、关节顺序和参数可信状态文件；输出根 link 固定的机械臂第一版。
- [ ] 验证网格路径和零位重装配，逐个关节正向转动并保存验证结果。

Run:
```powershell
& 'D:\Python\Python313\python.exe' scripts/urdf_pipeline/assemble.py --config urdf_work/config/assembly.json
& 'D:\Python\Python313\python.exe' scripts/urdf_pipeline/build_descriptions.py --model arm
& 'D:\Python\Python313\python.exe' scripts/urdf_pipeline/validate_model.py models/hzu_arm/robot.urdf
```

**验收:** 零位装配与几何重建一致；关节逐一运动正确；网格采用 m；参数状态清楚。此版可以是验证用模型，尚不等同于可信动力学模型。

## Task 5：补充物理参数并接入 MuJoCo / PyBullet

**Files:** Create `urdf_work/config/physics.json`、`sim.py`；生成 `models/hzu_arm/robot.xml` 和 `validation.json`。

**Interfaces:**
- `SimAdapter.reset(q: ndarray | None) -> dict`、`step(q_target_rad: ndarray) -> dict`、`render(camera: str) -> ndarray`、`close() -> None`。
- 固定物理步长 `1/240 s`；控制/采集频率 `30 Hz`，每控制周期 8 个物理步。
- 状态为实际输出侧关节角（rad），动作是同一顺序的绝对位置目标（rad）。
- `physics.json` 每个 link 保存质量 kg、局部重心 m、重心处惯量 kg·m² 及证据；每个 joint 保存 effort、velocity、damping 等及来源。

- [ ] 从材料密度和实体计算初估质量，并独立处理电机等内部构造未知的零件。
- [ ] 验证惯量对称、正定及主惯量三角约束；在变换和组合时正确应用旋转与平行轴定理。
- [ ] 明确位置控制执行器增益/力矩限幅是仿真控制设置，不能冒充电机实测参数。
- [ ] 在独立运行环境安装并锁定实际 MuJoCo/PyBullet 版本。
- [ ] PyBullet 用固定基座加载 URDF；MuJoCo 将 URDF 编译转换并补充所需执行器和相机，输出独立 MJCF。
- [ ] 执行零位、单关节小幅运动和预设多关节轨迹；对两引擎的零位/指定配置 FK 比较，位置误差目标 ≤1 mm、姿态误差 ≤0.1°。
- [ ] 执行 10 s 静止保持、复位重复性和接触检查，记录误差和未验证参数。

Run:
```powershell
python scripts/urdf_pipeline/sim.py --backend mujoco --model models/hzu_arm --validate
python scripts/urdf_pipeline/sim.py --backend pybullet --model models/hzu_arm --validate
```

**验收:** 两引擎均能加载、复位、控制和回放，FK 一致；动态差异保存到报告，不以相同轨迹文件掩盖实际关节响应差异。

## Task 6：建立 LeRobot 本地数据采集与回放

**Files:** Create `urdf_work/config/cameras.json`、`record_lerobot.py`、`tests/urdf_pipeline/test_dataset.py`；生成 `datasets/hzu_arm_sim/`。

**Interfaces:**
- 默认外部相机 `front`，RGB 640×480、30 fps；腕部相机只有在安装变换明确后加入。
- `observation.state`：float32[N]，按 `joint_order.json` 排列的实际关节位置 rad。
- `action`：float32[N]，相同顺序的绝对关节位置目标 rad。
- `observation.images.front`：uint8[480,640,3] RGB，保存真实仿真渲染。
- 记录时刻 t 的状态/图像和随后施加的 action_t，再推进 8 个物理步；数据时间为仿真时间，独立于墙钟渲染速度。
- 吸盘是布尔执行器，若实现则单独保存动作/状态字段，不能伪装成旋转关节角。
- 使用独立 LeRobot 环境及实际安装版本的官方 `LeRobotDataset` API，目标为 v3 格式；写入完整环境锁定文件。

- [ ] 根据实际 LeRobot 版本核对 `create/add_frame/save_episode/finalize` 的签名，使用官方 API 生成 Parquet/视频/元数据。
- [ ] 从同一仿真适配器记录 3 段不同轨迹，每段 5 s、150 帧；轨迹作为工程烟测数据，任务标签使用 `joint motion validation`。
- [ ] 检查 450 帧、3 段 episode，帧时间间隔 `1/30 s`、名字顺序与单位一致。
- [ ] 由官方加载器重新读入数据；检查相机颜色/尺寸、视频帧同步、状态与目标动作的差异。
- [ ] 回放同一动作序列，比较保存状态与回放实际状态，记录误差。
- [ ] 如需训练流程烟测，运行最小数据加载/批处理验证；完整模仿学习需要独立设计任务、遥操作或专家演示，关节扫动不作为有效抓取演示。

**验收:** 本地 LeRobotDataset 能重读和回放，时间/动作语义一致；数据集无外部上传依赖。

## Task 7：整机扩展与任务演示

**Files:** Generate `models/hzu_robot/`；扩展 assembly、joints、physics 和 cameras 配置。

- [ ] 恢复底架和机械臂安装变换，验证轮距、轴距及轮径。
- [ ] 加入四个驱动轮；展示/运动学可先简化，麦轮接触仿真需独立建模和验证。
- [ ] 按需要加入存矿机构和避震；机械闭环或接触约束在所选后端分别处理。
- [ ] LeRobot 若加入底盘，动作字段与机械臂字段分开定义，明确轮速或车体速度语义，版本化 feature schema。
- [ ] 在参数可信度允许后设计矿石搬运仿真任务，记录成功/失败标准及专家演示来源。
- [ ] 输出整机模型、仿真演示、数据说明和已知偏差报告。

**验收:** 整机几何与安装关系成立；任务成功条件明确；数据包含真实任务演示，物理简化有记录。

## 执行检查点

1. 文件/几何库可用：现在开始，最先交付。
2. 装配与关节证据闭合：决定能否生成准确机械臂；缺失项需要进一步几何检查或实物数据。
3. 机械臂在 MuJoCo/PyBullet 正确运动：再接 LeRobot。
4. LeRobot 数据闭环：再推进任务演示和整机。

每个阶段更新复选框和产物路径。当前用户已授权开始工作，按本会话继续执行；遇到关键证据缺失时明确报告具体缺口，并继续其他可独立完成的工作。

## 验证与版本记录

几何验证使用标准库 unittest 和合成形状，避免依赖原始零件才能运行。模型验证使用两引擎加载和独立 FK 对照；数据验证使用 LeRobot 官方读回。CAD 分析环境、仿真环境和 LeRobot 环境分别记录版本，避免升级破坏现有 CAD 工具链。

## 参考

- 已核实 CAD 原生读取支持：https://cadexchanger.com/solidworks/
- STEP 交互导出：https://github.com/Democratizing-Dexterous/step2urdf
- LeRobot 数据集官方文档：https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- LeRobot 自定义接口文档：https://huggingface.co/docs/lerobot/en/integrate_hardware

LeRobot 文档在 2026-10-14 查阅。接口以实施时锁定的实际版本为准；仿真模型和数据接口是两项独立交付，URDF 本身不包含 LeRobot 采集逻辑。
