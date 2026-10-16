param(
    [string]$Serial = 'emulator-5556',
    [string]$SdkPath = "$env:LOCALAPPDATA/Android/Sdk"
)
$ErrorActionPreference = 'Stop'
$adb = Join-Path $SdkPath 'platform-tools/adb.exe'

function Read-Adb([string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    $lines = & $adb -s $Serial @Arguments 2>&1 | ForEach-Object { $_.ToString() }
    ($lines -join "`n").Trim()
}
function Wait-Framework([int]$Seconds = 300) {
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        $state = Read-Adb -Arguments @('shell', 'getprop', 'sys.boot_completed')
        $features = Read-Adb -Arguments @('shell', 'pm', 'list', 'features')
        if ($state -eq '1' -and $features -match 'feature:android.hardware') { return $features }
        Start-Sleep -Seconds 2
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Android framework did not start on $Serial within $Seconds seconds"
}

$features = Wait-Framework
if ($features -notmatch 'feature:android.hardware.usb.host') {
    Write-Host 'Enabling USB host in the test emulator overlay...'
    $rootResult = Read-Adb -Arguments @('root')
    if ($rootResult -match 'cannot run as root') { throw 'This system image does not support adb root. Use Google APIs without Play Store.' }
    Start-Sleep -Seconds 2
    # Some images return a nonzero remount status for /system while /vendor is writable.
    $mountResult = Read-Adb -Arguments @('remount')
    $writable = Read-Adb -Arguments @('shell', 'test -w /vendor/etc/permissions && echo WRITABLE')
    if ($writable -notmatch 'WRITABLE' -and $mountResult -match 'reboot') {
        Write-Host 'Restarting the test AVD once to activate its writable overlay...'
        Read-Adb -Arguments @('reboot') | Out-Null
        Start-Sleep -Seconds 5
        Wait-Framework | Out-Null
        Read-Adb -Arguments @('root') | Out-Null
        Start-Sleep -Seconds 2
        $mountResult = Read-Adb -Arguments @('remount')
        $writable = Read-Adb -Arguments @('shell', 'test -w /vendor/etc/permissions && echo WRITABLE')
    }
    if ($writable -notmatch 'WRITABLE') {
        throw "The emulator vendor overlay is not writable. Start with -writable-system. Remount output: $mountResult"
    }
    $featureFile = Join-Path $PSScriptRoot 'android.hardware.usb.host.xml'
    $copyResult = Read-Adb -Arguments @('push', $featureFile, '/vendor/etc/permissions/android.hardware.usb.host.xml')
    if ($copyResult -notmatch 'pushed') { throw "Cannot copy USB host configuration: $copyResult" }
    Read-Adb -Arguments @('shell', 'chmod 644 /vendor/etc/permissions/android.hardware.usb.host.xml; restorecon /vendor/etc/permissions/android.hardware.usb.host.xml') | Out-Null
    # Restart Android services to reread feature XML without rebooting the emulator's ADB transport.
    Read-Adb -Arguments @('shell', 'setprop sys.boot_completed 0; stop; start') | Out-Null
    Start-Sleep -Seconds 3
    $features = Wait-Framework
    if ($features -notmatch 'feature:android.hardware.usb.host') { throw 'USB host feature was not loaded after restarting Android services.' }
}
$state = Read-Adb -Arguments @('shell', 'dumpsys', 'usb')
$dashboardRoot = Split-Path -Parent $PSScriptRoot
$state | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $dashboardRoot "build/$Serial-usb-state.txt")
if ($state -notmatch 'host_manager') { throw 'UsbHostManager was not started by this image.' }
if ($state -notmatch 'vendor_id=1027' -or $state -notmatch 'product_id=24577') {
    throw 'USB host is enabled, but QEMU FTDI 0403:6001 was not detected. Check QEMU controller and usb-serial arguments.'
}
Write-Host 'USB host ready: QEMU USB SERIAL / FTDI 0403:6001'
