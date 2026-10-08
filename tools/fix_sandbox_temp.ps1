#Requires -Version 5.1
<#
  Diagnose and repair the DSH Windows sandbox temp-directory failure.

  Symptom this fixes
  ------------------
      sandbox mode "workspace-write" is requested but no sandbox backend is usable
      Runner failure: windows-acl-run: --temp is not an existing directory:
                      C:\Users\<user>\AppData\Local\Temp\dsh-XXXXXX

  and every confined command dies with exit code 0xC0000142
  (STATUS_DLL_INIT_FAILED), because the sandboxed process cannot initialise at all.

  Why writing a fixed directory by hand does not help
  --------------------------------------------------
  The runner picks a NEW random suffix on every invocation, so the directory it
  wants can never be pre-created.  The real question is why the runner cannot
  create it.  The file tools (normal user token) can write there; the sandbox
  runner uses a RESTRICTED token, so this is an ACL question, not a missing
  directory.

  What this script does
  ---------------------
    1. Prints your identity and the ACLs of the temp directory and its parents.
    2. Saves the current SDDL of the temp directory so the change can be undone.
    3. Adds an inheritable FullControl ACE for the signed-in user on the temp
       directory ONLY (no recursion into existing children, no changes to parents,
       no removal of any existing entry).
    4. Creates and removes a test subdirectory to confirm the right is now present.
    5. Prints the exact rollback command.

  Run it in a normal (non-sandboxed) PowerShell:

      pwsh -ExecutionPolicy Bypass -File D:\MyTrain\LeRobot\tools\fix_sandbox_temp.ps1

  Add -DiagnoseOnly to change nothing and just print the facts.
#>

[CmdletBinding()]
param(
    [switch]$DiagnoseOnly
)

$ErrorActionPreference = 'Continue'

function Head($t) { Write-Host ""; Write-Host "=== $t ===" -ForegroundColor Cyan }

# --------------------------------------------------------------- 1. identity
Head "identity"
Write-Host ("whoami        : " + (whoami))
Write-Host ("user SID      : " + (whoami /user | Select-String 'S-1-5-21' | ForEach-Object { $_.Line.Trim() }))
Write-Host ("PowerShell    : " + $PSVersionTable.PSVersion)
Write-Host ("Is elevated   : " + ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator))

# ------------------------------------------------------------- 2. temp dirs
$temp = [System.IO.Path]::GetTempPath().TrimEnd('\')
Write-Host ("TEMP          : $temp")

$chain = @()
$p = $temp
while ($p -and $p -ne (Split-Path $p -Parent)) {
    $chain += $p
    $p = Split-Path $p -Parent
}
$chain += $p

Head "ACL chain (nearest first)"
foreach ($dir in $chain) {
    if (-not (Test-Path $dir)) { Write-Host "  $dir  <MISSING>"; continue }
    $acl = Get-Acl $dir -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "  $dir" -ForegroundColor Yellow
    Write-Host ("    owner : " + $acl.Owner)
    foreach ($ace in $acl.Access) {
        $inh = if ($ace.IsInherited) { 'inherited' } else { 'EXPLICIT ' }
        Write-Host ("    {0} {1,-45} {2,-28} {3}" -f $inh, $ace.IdentityReference, $ace.AccessControlType, $ace.FileSystemRights)
    }
}

# ------------------------------------------------- 3. can we actually create?
Head "can the CURRENT (unrestricted) token create a subdirectory?"
$probe = Join-Path $temp ("dsh-probe-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
try {
    New-Item -ItemType Directory -Path $probe -ErrorAction Stop | Out-Null
    Write-Host "  YES  created $probe" -ForegroundColor Green
    Remove-Item $probe -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "       (this only proves the normal token works; the sandbox uses a"
    Write-Host "        restricted token, which is the one that fails)"
} catch {
    Write-Host "  NO   $($_.Exception.Message)" -ForegroundColor Red
}

if ($DiagnoseOnly) {
    Write-Host ""
    Write-Host "DiagnoseOnly: nothing was changed." -ForegroundColor Green
    exit 0
}

# ------------------------------------------------------------ 4. back up SDDL
Head "backup"
$backupDir = 'D:\MyTrain\LeRobot\outputs'
if (-not (Test-Path $backupDir)) { New-Item -ItemType Directory -Path $backupDir -Force | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $backupDir "temp_acl_$stamp.txt"
try {
    $sddl = (Get-Acl $temp -ErrorAction Stop).Sddl
    "dir  : $temp"      | Out-File -FilePath $backup -Encoding utf8
    "sddl : $sddl"      | Out-File -FilePath $backup -Encoding utf8 -Append
    Write-Host "  saved previous ACEs to $backup" -ForegroundColor Green
    Write-Host "  SDDL was: $sddl"
} catch {
    Write-Host "  FAILED to read the ACL: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  stopping rather than changing something we cannot restore." -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------ 5. grant
Head "grant the signed-in user inheritable FullControl on the temp directory"
$me = "$env:USERDOMAIN\$env:USERNAME"
Write-Host "  granting: $me : (OI)(CI)F   on   $temp"
Write-Host "  scope   : this directory only - existing children, parents and all"
Write-Host "            current entries are left untouched"
$out = & icacls $temp /grant "${me}:(OI)(CI)F" 2>&1
$out | ForEach-Object { Write-Host "    $_" }
Write-Host "  icacls exit code: $LASTEXITCODE"

Head "ACL after the change"
& icacls $temp 2>&1 | ForEach-Object { Write-Host "    $_" }

# --------------------------------------------------------------- 6. verify
Head "verify"
$probe2 = Join-Path $temp ("dsh-probe-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
try {
    New-Item -ItemType Directory -Path $probe2 -ErrorAction Stop | Out-Null
    Write-Host "  created $probe2" -ForegroundColor Green
    Remove-Item $probe2 -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  removed it again" -ForegroundColor Green
} catch {
    Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Head "rollback (only if you need to undo the grant)"
Write-Host "    icacls `"$temp`" /remove:g `"$me`""
Write-Host "  or restore the full previous ACL from:"
Write-Host "    $backup"

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host "Now go back to the DSH session and try a confined command again."
Write-Host "If it STILL fails with 0xC0000142, paste the 'ACL chain' section above -"
Write-Host "the missing right is then on a parent directory or belongs to a different"
Write-Host "SID, and guessing further would only make a mess."
