# LeRobot environment bootstrap for 27_engineer arm project
# Target: Windows, Python 3.13.  Caches go next to the repo, never on C:.
# ASCII-only on purpose: avoids encoding surprises when invoked as a script file.
#
# Usage:  pwsh -ExecutionPolicy Bypass -File tools\setup_env.ps1
# Idempotent: safe to re-run.
#
# Portable: the repo root comes from this file's location, so cloning to another
# drive/folder works.  Override the base interpreter with $env:ARM_PYTHON.

$ErrorActionPreference = 'Continue'

$Root    = Split-Path -Parent $PSScriptRoot
$Venv    = Join-Path $Root '.venv'
$Cache   = Join-Path $Root '.cache'
$Index   = 'https://pypi.tuna.tsinghua.edu.cn/simple'
$Vpy     = Join-Path $Venv 'Scripts\python.exe'

# Base interpreter: explicit override, else the known local install, else PATH.
$PyBase  = $env:ARM_PYTHON
if (-not $PyBase) { $PyBase = 'D:\Python\Python313\python.exe' }
if (-not (Test-Path $PyBase)) {
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) {
        $PyBase = $onPath.Source
        Write-Host "ARM_PYTHON not set and D:\Python\Python313 not found; using $PyBase" -ForegroundColor Yellow
    } else {
        Write-Host "!!! No Python interpreter found. Set `$env:ARM_PYTHON to a Python 3.12+ executable." -ForegroundColor Red
        exit 1
    }
}

function Step($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Check($label) {
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! $label FAILED (exit $LASTEXITCODE)" -ForegroundColor Red }
    else { Write-Host "OK  $label" -ForegroundColor Green }
}

# ---------------------------------------------------------------- 1. dirs
Step '1. Create directory layout'
$dirs = @(
    $Root,
    (Join-Path $Root 'docs'),
    (Join-Path $Root 'tools'),
    (Join-Path $Root 'configs'),
    (Join-Path $Root 'src'),
    (Join-Path $Root 'data'),
    (Join-Path $Root 'outputs'),
    (Join-Path $Cache 'pip'),
    (Join-Path $Cache 'tmp'),
    (Join-Path $Cache 'huggingface'),
    (Join-Path $Cache 'torch')
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
Get-ChildItem $Root -Force | Select-Object Mode, Name | Format-Table -AutoSize

# ------------------------------------------------- 2. redirect caches off C:
Step '2. Redirect caches off C: (CRITICAL - C: has ~3.5 GB free)'
$env:PIP_CACHE_DIR = Join-Path $Cache 'pip'
$env:TMP           = Join-Path $Cache 'tmp'
$env:TEMP          = $env:TMP
$env:HF_HOME       = Join-Path $Cache 'huggingface'
$env:TORCH_HOME    = Join-Path $Cache 'torch'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
# HuggingFace is unreachable from this machine (DNS fails for huggingface.co,
# hf-mirror.com times out). Force everything to local-only mode.
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Get-ChildItem env: | Where-Object { $_.Name -in 'PIP_CACHE_DIR','TMP','TEMP','HF_HOME','TORCH_HOME','HF_HUB_OFFLINE' } |
    Select-Object Name, Value | Format-Table -AutoSize

# ---------------------------------------------------------------- 3. venv
Step '3. Create virtualenv (Python 3.13)'
if (-not (Test-Path $Vpy)) {
    & $PyBase -m venv $Venv
    Check 'venv creation'
} else {
    Write-Host "venv already exists: $Venv"
}
& $Vpy -V
& $Vpy -c "import sys; print('executable:', sys.executable); print('version:', sys.version)"

# ---------------------------------------------------------------- 4. pip
Step '4. Upgrade pip / setuptools / wheel (TUNA mirror)'
& $Vpy -m pip install --upgrade pip setuptools wheel -i $Index
Check 'pip upgrade'
& $Vpy -m pip -V

# ------------------------------------------------------------- 5. lerobot
Step '5. Install lerobot[dataset,training,core-scripts,viz] (this takes a while)'
Write-Host "Note: LeRobot 0.6.1 requires python>=3.12, torch>=2.7,<2.12 -> expects torch 2.11 on cp313"
& $Vpy -m pip install "lerobot[dataset,training,core-scripts,viz]" -i $Index --timeout 120 --retries 5
Check 'lerobot install'

# ------------------------------------------- 5b. this project, editable
# Required for LeRobot to discover the plugin: lerobot-record resolves
# --robot.type=arm_27 through the installed distribution's entry points, so the
# package has to be installed, not merely present on disk.
Step '5b. Install this project in editable mode (arm-lerobot)'
& $Vpy -m pip install -e $Root -i $Index --timeout 120 --retries 5
Check 'arm-lerobot editable install'
$env:PYTHONPATH = Join-Path $Root 'src'

# ------------------------------------------------- 6. camera sdk + helpers
Step '6. Install Orbbec camera SDK (--no-deps to protect opencv-python-headless)'
Write-Host "Why --no-deps: pyorbbecsdk2 depends on opencv-python, LeRobot on opencv-python-headless;"
Write-Host "both ship the same cv2 package, so letting pip resolve both would clobber one of them."
& $Vpy -m pip install pyorbbecsdk2 --no-deps -i $Index --timeout 120 --retries 5
Check 'pyorbbecsdk2 install'

Step '6b. Install remaining real deps of pyorbbecsdk2 + serial for UART7'
& $Vpy -m pip install av pygame pynput pyserial -i $Index --timeout 120 --retries 5
Check 'av/pygame/pynput/pyserial install'

# ---------------------------------------------------------------- 7. verify
Step '7. Verify'
& $Vpy -c @"
from importlib.metadata import version, PackageNotFoundError
for pkg in ['lerobot','torch','torchvision','numpy','opencv-python-headless','opencv-python','av','huggingface-hub','pyorbbecsdk2','pyserial','datasets','accelerate','arm-lerobot']:
    try: print(f'{pkg:26s} {version(pkg)}')
    except PackageNotFoundError: print(f'{pkg:26s} <NOT INSTALLED>')
"@
Write-Host ""
Write-Host "--- import smoke test ---"
& $Vpy -c "import numpy, cv2, torch, torchvision; print('numpy', numpy.__version__); print('cv2', cv2.__version__); print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
& $Vpy -c "import pyorbbecsdk; print('pyorbbecsdk import OK')"
& $Vpy -c "import arm_lerobot; print('arm_lerobot import OK:', arm_lerobot.__file__)"
& $Vpy -c "from arm_lerobot.camera import OrbbecCamera; print('OrbbecCamera import OK')"
& $Vpy -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobotDataset import OK')"
& $Vpy -c "from lerobot.robots.robot import Robot; print('Robot base class import OK')"
Check 'import smoke test'

# ---------------------------------------------------------------- 8. report
Step '8. Disk report'
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -ne $null } |
    Select-Object Name, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}} | Format-Table -AutoSize
Write-Host ""
Write-Host "venv size:"
$sz = (Get-ChildItem $Venv -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
"{0:N2} GB" -f ($sz/1GB)
Write-Host ""
Write-Host "DONE. Activate with:  $Venv\Scripts\Activate.ps1"
