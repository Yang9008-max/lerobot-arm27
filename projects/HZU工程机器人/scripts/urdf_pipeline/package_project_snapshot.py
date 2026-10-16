"""Losslessly package the engineering workspace, excluding caches and upload copies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from inventory import ROOT

OUTPUT=ROOT/'urdf_work/publish/project_snapshot'
PROJECT=OUTPUT/'projects/HZU工程机器人'
PART_BYTES=32*1024*1024
SKIP_NAMES={'.git','__pycache__','.pytest_cache','.mypy_cache','.venv','node_modules'}
SKIP_PREFIXES={'urdf_work/publish','urdf_work/test_tmp'}


class Volumes:
    def __init__(self,directory):
        directory.mkdir(parents=True,exist_ok=True)
        self.directory=directory
        self.total=0
        self.current=None
        self.records=[]

    def tell(self): return self.total
    def seek(self,*args): raise OSError('Forward-only split ZIP')
    def flush(self):
        if self.current: self.current.flush()

    def finish_part(self):
        if self.current:
            self.current.close()
            self.records[-1]['sha256']=self.digest.hexdigest()
            self.current=None

    def write(self,data):
        original=len(data)
        while data:
            if self.current is None:
                name='hzu-project.zip.'+str(len(self.records)+1).zfill(3)
                self.current=(self.directory/name).open('wb')
                self.digest=hashlib.sha256()
                self.records.append({'path':'archive/'+name,'bytes':0})
            count=min(len(data),PART_BYTES-self.records[-1]['bytes'])
            piece=data[:count]
            self.current.write(piece);self.digest.update(piece)
            self.total+=count;self.records[-1]['bytes']+=count
            data=data[count:]
            if self.records[-1]['bytes']==PART_BYTES: self.finish_part()
        return original


def select_files():
    files=[]
    for directory,folders,names in os.walk(ROOT):
        relative=Path(directory).relative_to(ROOT)
        folders[:]=sorted(name for name in folders if name not in SKIP_NAMES and
                          (relative/name).as_posix() not in SKIP_PREFIXES)
        for name in sorted(names):
            if name.startswith('~$') or name.endswith(('.pyc','.pyo')): continue
            path=Path(directory)/name
            if path.is_symlink(): raise ValueError('Review symbolic link before archiving: '+str(path))
            files.append(path)
    return sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())


def main():
    files=select_files()
    original=json.loads((ROOT/'urdf_work/evidence/inventory.json').read_text(encoding='utf-8'))
    expected={r['path']:r['sha256'] for r in original['files'] if r['role']=='source_input'}
    PROJECT.mkdir(parents=True,exist_ok=True)
    writer=Volumes(PROJECT/'archive')
    records=[]
    total=sum(p.stat().st_size for p in files)
    print(json.dumps({'stage':'packing','files':len(files),'uncompressed_bytes':total}),flush=True)
    with zipfile.ZipFile(writer,mode='w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for index,path in enumerate(files,1):
            relative=path.relative_to(ROOT).as_posix()
            before=path.stat()
            info=zipfile.ZipInfo.from_file(path,arcname=relative,strict_timestamps=False)
            info.compress_type=zipfile.ZIP_DEFLATED
            info._compresslevel=6
            digest=hashlib.sha256()
            with path.open('rb') as source,archive.open(info,'w',force_zip64=True) as target:
                while block:=source.read(1024*1024):
                    digest.update(block);target.write(block)
            after=path.stat()
            if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
                raise ValueError('File changed during packing: '+relative)
            value=digest.hexdigest()
            if relative in expected and value!=expected[relative]:
                raise ValueError('Original CAD/source differs from indexed content: '+relative)
            records.append({'path':relative,'bytes':before.st_size,'sha256':value,
                            'original_source':relative in expected})
            if index%50==0 or index==len(files):
                print(json.dumps({'packed':index,'total':len(files),'archive_bytes':writer.total,
                                  'last_file':relative},ensure_ascii=False),flush=True)
    writer.finish_part()
    present={r['path'] for r in records}
    if missing:=set(expected)-present: raise ValueError('Original inputs missing: '+str(missing))
    manifest={'schema_version':1,'source_directory':str(ROOT),'archive_format':'ZIP64 deflate; byte-concatenated 32 MiB volumes',
              'files':records,'parts':writer.records,'total_files':len(records),'uncompressed_bytes':total,
              'archive_bytes':writer.total,'original_source_files':len(expected),
              'exclusions':sorted(SKIP_NAMES|SKIP_PREFIXES|{'~$* (SolidWorks locks)','*.pyc','*.pyo'}),
              'original_sources_match_inventory':True,'original_cad_modified':False}
    (PROJECT/'archive_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Expose small text sources for GitHub review without restoring the snapshot.
    for path in files:
        relative=path.relative_to(ROOT)
        if relative.parts[0] in {'docs','scripts','tests'} or relative.as_posix().startswith('urdf_work/config/'):
            target=PROJECT/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    shutil.copyfile(ROOT/'scripts/urdf_pipeline/restore_project_snapshot.py',PROJECT/'restore_project.py')
    readme=f'''# HZU 工程机器人完整工程快照

本目录保存 `D:\\MyTrain\\projects\\engineer\\HZU工程机器人` 的工程资料、原始 CAD、模型、分析脚本、证据及检查结果。

- 快照共 **{len(records)} 个文件**，未压缩 {total:,} 字节。
- 原始资料 **{len(expected)} 个文件**全部包含，SHA-256 与原始索引一致；原始 CAD 未修改。
- ZIP64 压缩后 {writer.total:,} 字节，分为 **{len(writer.records)} 卷**，每卷最多 32 MiB。
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
'''
    (PROJECT/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['total_files','original_source_files','uncompressed_bytes','archive_bytes']},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
