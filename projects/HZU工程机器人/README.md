# HZU 工程机器人完整工程快照

本目录保存 `D:\MyTrain\projects\engineer\HZU工程机器人` 的工程资料、原始 CAD、模型、分析脚本、证据及检查结果。

- 快照共 **3057 个文件**，未压缩 2,118,026,931 字节。
- 原始资料 **545 个文件**全部包含，SHA-256 与原始索引一致；原始 CAD 未修改。
- ZIP64 压缩后 1,353,277,234 字节，分为 **117 卷**，每卷最多 32 MiB（尚未上传部分细分为最多 8 MiB）。
- 排除 Git 内部目录、Python 缓存、虚拟环境、SolidWorks 锁文件、测试临时目录和 `urdf_work/publish` 上传暂存副本。

两个原始 SLDPRT 各约 212 MB，超过 GitHub 普通文件大小限制，因此工程以无损压缩分卷保存。各卷是同一个 ZIP 文件按字节拆分的片段，需要一起使用。代码、文档和关节配置也直接列在本目录中，方便网页浏览。

## 还原整个工程

克隆或下载包含这些文件的仓库分支后，在仓库根目录执行（只需 Python 标准库）：

```powershell
# 校验分卷和压缩包内的全部文件，不写出工程。
python "projects/HZU工程机器人/restore_project.py" --check

# 默认还原到本目录；相同文件跳过，内容不同的现有文件会停止并报告。
python "projects/HZU工程机器人/restore_project.py"

# 也可指定新的输出目录。
python "projects/HZU工程机器人/restore_project.py" --output "D:/HZU工程机器人"
```

`archive_manifest.json` 记录每卷以及每个工程文件的 SHA-256、大小和原始相对路径。还原后，`HZU工程完整装配.SLDASM`、各零件目录、`models`、`scripts`、`tests`、`docs` 和 `urdf_work` 按原目录结构恢复。

## 直接查看 URDF

无需还原快照，可使用仓库已有的 [整机 URDF 模型包](../../models/hzu_robot/README.md) 和 `tools/view_hzu.py`。快照同时保留本地的完整 OBJ/STL 模型与历史检查文件。

URDF 当前是整机几何/输出侧运动学版：包含底盘、四轮、六轴机械臂和主要电机外形。碰撞、真实惯量、实物限位、被动机构与驱动标定仍待完善；它与仓库原有三轴 `arm_27` 实机接口不同。
