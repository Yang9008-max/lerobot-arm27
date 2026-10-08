# Source this in every shell before touching this project:  . .\tools\env.ps1
#
# What it does:
#   1. Puts ALL caches on D: - C: has only ~3.5 GB free.
#   2. Auto-detects a local HTTP proxy (Clash Verge / mihomo here) and exports
#      HTTP_PROXY/HTTPS_PROXY for Python.  This is REQUIRED: the proxy runs in
#      fake-IP mode, so PowerShell (WinINET) works but Python's requests / pip /
#      huggingface_hub do NOT read the Windows proxy setting and would try to
#      dial the fake 198.18.x.x address directly and time out.
#   3. Uses a fast, verified mirror for pip by default.
#   4. Keeps HuggingFace in offline mode by default so a data-collection run can
#      never hang on a network call.  Set $env:LEROBOT_HF_OFFLINE='0' to allow Hub access.
#
# The project root is derived from this file's own location, so the repo can be
# cloned to any drive or folder.  Override with $env:ARM_LEROBOT_ROOT if needed.

$Root  = $env:ARM_LEROBOT_ROOT
if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
$Cache = Join-Path $Root '.cache'
$Venv  = Join-Path $Root '.venv'
$Vpy   = Join-Path $Venv 'Scripts\python.exe'

# ---------------------------------------------------------------- caches on D:
$env:PIP_CACHE_DIR        = Join-Path $Cache 'pip'
$env:TMP                  = Join-Path $Cache 'tmp'
$env:TEMP                 = $env:TMP
$env:HF_HOME              = Join-Path $Cache 'huggingface'
$env:TORCH_HOME           = Join-Path $Cache 'torch'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

# ------------------------------------------------------- proxy auto-detection
# Clash Verge / mihomo was observed listening on 127.0.0.1:7897 on this machine.
$proxyPort = $null
foreach ($p in 7897, 7890, 7891, 10809, 10808, 1080, 8889, 2080) {
    if (Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue) {
        $proxyPort = $p
        break
    }
}
if ($proxyPort) {
    $env:HTTP_PROXY  = "http://127.0.0.1:$proxyPort"
    $env:HTTPS_PROXY = $env:HTTP_PROXY
    $env:ALL_PROXY   = $env:HTTP_PROXY
    $env:NO_PROXY    = 'localhost,127.0.0.1,::1'
    $proxyState = "on (127.0.0.1:$proxyPort)"
} else {
    Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY -ErrorAction SilentlyContinue
    $proxyState = 'none detected'
}

# ------------------------------------------------------------------ pip index
# TUNA measured 260 ms for a HEAD and has every wheel we need.
# Override with:  $env:PIP_INDEX_URL = 'https://pypi.org/simple'
if (-not $env:PIP_INDEX_URL) {
    $env:PIP_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'
}
$env:PIP_TRUSTED_HOST = ([uri]$env:PIP_INDEX_URL).Host

# ------------------------------------------------------------- HuggingFace mode
# Default offline: huggingface.co is only reachable while the proxy is up, and a
# hang in the middle of a collection run is much worse than not having the Hub.
if ($env:LEROBOT_HF_OFFLINE -eq '0') {
    $env:HF_HUB_OFFLINE       = '0'
    $env:TRANSFORMERS_OFFLINE = '0'
} else {
    $env:HF_HUB_OFFLINE       = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
}

Write-Host "LeRobot env ready." -ForegroundColor Green
Write-Host "  root   : $Root"
Write-Host "  python : $Vpy"
Write-Host "  caches : $Cache"
Write-Host "  index  : $env:PIP_INDEX_URL"
Write-Host "  proxy  : $proxyState"
Write-Host "  HF     : $(if ($env:HF_HUB_OFFLINE -eq '1') { 'offline (set LEROBOT_HF_OFFLINE=0 to enable)' } else { 'ONLINE' })"
