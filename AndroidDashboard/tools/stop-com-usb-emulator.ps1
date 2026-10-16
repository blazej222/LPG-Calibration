$ErrorActionPreference = 'Stop'
$dashboardRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $dashboardRoot 'build/usb-api36-avd/session.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Host 'No managed USB test session.'; return }
$state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
$adb = Join-Path $state.SdkPath 'platform-tools/adb.exe'
$devices = & $adb devices | Out-String
if ($devices -match "(?m)^$($state.Serial)\s+device") {
    $actualName = (& $adb -s $state.Serial shell getprop ro.boot.qemu.avd_name | Out-String).Trim()
    if ($actualName -ne $state.AvdName) { throw "Refusing to stop a different emulator: $actualName" }
    & $adb -s $state.Serial emu kill
} elseif ($devices -match "(?m)^$($state.Serial)\s+offline") {
    $owned = Get-CimInstance Win32_Process -Filter "ProcessId = $($state.EmulatorPid)" -ErrorAction SilentlyContinue
    if ($owned -and $owned.CommandLine -match [regex]::Escape($state.AvdName)) {
        & $adb -s $state.Serial emu kill
    } else { throw 'Cannot verify ownership of the offline emulator.' }
}
$bridge = Get-CimInstance Win32_Process -Filter "ProcessId = $($state.BridgePid)" -ErrorAction SilentlyContinue
$bridgePath = Join-Path $PSScriptRoot 'com_usb_bridge.py'
if ($bridge) {
    if ($bridge.CommandLine -notmatch [regex]::Escape($bridgePath)) { throw 'Refusing to stop an unrelated Python process.' }
    Stop-Process -Id $bridge.ProcessId
}
Wait-Process -Id $state.EmulatorPid -Timeout 20 -ErrorAction SilentlyContinue
Write-Host "USB test session stopped; $($state.ComPort) released. Existing ECU emulator and other AVDs were not stopped."
