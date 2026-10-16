param(
    [ValidatePattern('^COM\d+$')][string]$ComPort = 'COM19',
    [ValidateRange(1, 65535)][int]$BridgePort = 34567,
    [ValidateRange(5554, 5682)][int]$EmulatorPort = 5556,
    [string]$SdkPath,
    [switch]$Headless,
    [switch]$TraceBridge
)
$ErrorActionPreference = 'Stop'
if ($EmulatorPort % 2 -ne 0) { throw 'EmulatorPort must be even.' }
$dashboardRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $dashboardRoot
if (-not $SdkPath) {
    $properties = Join-Path $dashboardRoot 'local.properties'
    if (Test-Path -LiteralPath $properties) {
        $sdkLine = Get-Content -LiteralPath $properties | Where-Object { $_ -match '^sdk\.dir=' } | Select-Object -First 1
        if ($sdkLine) { $SdkPath = $sdkLine.Substring(8).Replace('\\', '\').Replace('\:', ':') }
    }
    if (-not $SdkPath) { $SdkPath = Join-Path $env:LOCALAPPDATA 'Android/Sdk' }
}
$emulatorExe = Join-Path $SdkPath 'emulator/emulator.exe'
$adb = Join-Path $SdkPath 'platform-tools/adb.exe'
$imagePath = Join-Path $SdkPath 'system-images/android-36/google_apis/x86_64'
$pythonExe = Join-Path $repoRoot 'GasECU Reverse Engineering/.venv/Scripts/python.exe'
foreach ($file in @($emulatorExe, $adb, $pythonExe, (Join-Path $imagePath 'system.img'))) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing: $file. Run install-usb-test-image.py or install the required SDK/Python tools first." }
}
$serial = "emulator-$EmulatorPort"
$devices = & $adb devices | Out-String
if ($devices -match "(?m)^$serial\s") { throw "$serial is already running. Choose another -EmulatorPort or stop the previous test session." }
$portProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $BridgePort)
try { $portProbe.Start() }
catch { throw "Local bridge port $BridgePort is occupied. Stop the previous test bridge or choose -BridgePort." }
finally { $portProbe.Stop() }

$avdName = 'GAS_ECU_USB_API36'
$avdRoot = Join-Path $dashboardRoot 'build/usb-api36-avd'
$avdPath = Join-Path $avdRoot "$avdName.avd"
New-Item -ItemType Directory -Force -Path $avdPath | Out-Null
$configPath = Join-Path $avdPath 'config.ini'
if (-not (Test-Path -LiteralPath $configPath)) {
    @"
AvdId=$avdName
avd.ini.displayname=GAS ECU USB Android 16
abi.type=x86_64
tag.id=google_apis
tag.display=Google APIs
image.sysdir.1=system-images\android-36\google_apis\x86_64\
PlayStore.enabled=false
hw.cpu.arch=x86_64
hw.cpu.ncore=4
hw.ramSize=2048
hw.lcd.width=1024
hw.lcd.height=600
hw.lcd.density=160
hw.gpu.enabled=yes
hw.gpu.mode=auto
hw.keyboard=yes
hw.mainKeys=no
hw.audioInput=no
hw.audioOutput=no
hw.camera.back=none
hw.camera.front=none
hw.sdCard=no
showDeviceFrame=no
disk.dataPartition.size=4G
fastboot.forceColdBoot=yes
runtime.network.latency=none
runtime.network.speed=full
"@ | Set-Content -Encoding ASCII -LiteralPath $configPath
}
$registration = "avd.ini.encoding=UTF-8`npath=$avdPath`ntarget=android-36"
Set-Content -Encoding ASCII -LiteralPath (Join-Path $avdRoot "$avdName.ini") -Value $registration
$userAvdRoot = Join-Path $env:USERPROFILE '.android/avd'
New-Item -ItemType Directory -Force -Path $userAvdRoot | Out-Null
$userRegistration = Join-Path $userAvdRoot "$avdName.ini"
if ((Test-Path -LiteralPath $userRegistration) -and (Get-Content -Raw -LiteralPath $userRegistration) -notmatch [regex]::Escape("path=$avdPath")) {
    throw "A different AVD already uses this name: $userRegistration"
}
Set-Content -Encoding ASCII -LiteralPath $userRegistration -Value $registration
$statePath = Join-Path $avdRoot 'session.json'
$bridgeLog = Join-Path $avdRoot 'bridge.log'
$bridgeError = Join-Path $avdRoot 'bridge-error.log'
$emulatorLog = Join-Path $avdRoot 'emulator.log'
$emulatorError = Join-Path $avdRoot 'emulator-error.log'
$bridgeScript = Join-Path $PSScriptRoot 'com_usb_bridge.py'
$bridgeProcess = $null
$emulatorProcess = $null
$oldAvdRoot = $env:ANDROID_AVD_HOME
try {
    $bridgeArgs = @('-u', "`"$bridgeScript`"", '--com', $ComPort, '--port', "$BridgePort")
    if ($TraceBridge) { $bridgeArgs += '--trace' }
    $bridgeProcess = Start-Process -FilePath $pythonExe -ArgumentList $bridgeArgs -WindowStyle Hidden `
        -RedirectStandardOutput $bridgeLog -RedirectStandardError $bridgeError -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    do {
        if ($bridgeProcess.HasExited) { throw "Cannot open $ComPort. See $bridgeError" }
        if ((Test-Path -LiteralPath $bridgeLog) -and (Get-Content -Raw -LiteralPath $bridgeLog) -match 'READY ') { break }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    if ((Get-Content -Raw -LiteralPath $bridgeLog) -notmatch 'READY ') { throw 'Bridge did not become ready.' }
    $env:ANDROID_AVD_HOME = $avdRoot
    $emulatorArgs = @('-avd', $avdName, '-port', "$EmulatorPort", '-no-snapshot', '-no-audio', '-writable-system', '-gpu', 'swiftshader')
    if ($Headless) { $emulatorArgs += '-no-window' }
    $emulatorArgs += @('-qemu', '-device', 'qemu-xhci,id=lpgxhci',
        '-chardev', "socket,id=lpgcom,host=127.0.0.1,port=$BridgePort",
        '-device', 'usb-serial,chardev=lpgcom,bus=lpgxhci.0')
    $windowStyle = if ($Headless) { 'Hidden' } else { 'Normal' }
    $emulatorProcess = Start-Process -FilePath $emulatorExe -ArgumentList $emulatorArgs -WindowStyle $windowStyle `
        -RedirectStandardOutput $emulatorLog -RedirectStandardError $emulatorError -PassThru
    @{ Serial = $serial; AvdName = $avdName; BridgePid = $bridgeProcess.Id; EmulatorPid = $emulatorProcess.Id; ComPort = $ComPort; SdkPath = $SdkPath } |
        ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $statePath
    Write-Host "Waiting for $serial (first boot can take a few minutes)..."
    & (Join-Path $PSScriptRoot 'enable-emulator-usb-host.ps1') -Serial $serial -SdkPath $SdkPath
    $apk = Join-Path $dashboardRoot 'app/build/outputs/apk/debug/app-debug.apk'
    if (Test-Path -LiteralPath $apk) {
        & $adb -s $serial install -r $apk
        if ($LASTEXITCODE -ne 0) { throw 'APK installation failed.' }
        & $adb -s $serial shell am start -n com.blazej222.lpgdashboard/.MainActivity
    }
    Write-Host "READY: $serial / $avdName / USB FTDI 0403:6001 / $ComPort"
    Write-Host 'In GAS ECU: Ustawienia -> QEMU USB SERIAL -> Polacz -> allow USB access.'
    Write-Host "Keep the existing ECU emulator on COM18 running. Logs: $avdRoot"
    Write-Host 'Stop this test session with tools/stop-com-usb-emulator.ps1 to release the COM port.'
} catch {
    if ($emulatorProcess -and -not $emulatorProcess.HasExited) { & $adb -s $serial emu kill | Out-Null }
    if ($bridgeProcess -and -not $bridgeProcess.HasExited) { Stop-Process -Id $bridgeProcess.Id }
    throw
} finally { $env:ANDROID_AVD_HOME = $oldAvdRoot }
