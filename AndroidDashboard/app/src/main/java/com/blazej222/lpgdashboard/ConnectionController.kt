package com.blazej222.lpgdashboard

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.net.Uri
import android.os.SystemClock
import android.os.Build
import androidx.core.content.ContextCompat
import com.blazej222.lpgdashboard.protocol.*
import com.hoho.android.usbserial.driver.UsbSerialDriver
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class UsbChoice(val device: UsbDevice, val portIndex: Int?, val driverName: String?) {
    val key: String get() = "${device.deviceName}:$portIndex"
    val savedKey: String get() = "${device.vendorId}:${device.productId}:$portIndex"
    val title: String get() = "${device.productName ?: "Urządzenie USB"} · ${driverName ?: "brak sterownika UART"}"
    val details: String get() = "VID %04X · PID %04X · %s%s".format(
        device.vendorId, device.productId, device.deviceName, portIndex?.let { " · port ${it + 1}" } ?: "")
}

data class DashboardSettings(
    val decoder: DecoderConfig = DecoderConfig(),
    val pollIntervalMs: Long = 200,
    val assertControlLines: Boolean = true,
    val originalPolling: Boolean = false,
    val averageInjectionTimes: Boolean = true,
    val injectorCount: Int = 4,
)

enum class ConnectionPhase { DISCONNECTED, PERMISSION, CONNECTING, INITIALIZING, STREAMING, STOPPING, ERROR }
data class DashboardState(
    val devices: List<UsbChoice> = emptyList(),
    val selectedKey: String? = null,
    val settings: DashboardSettings = DashboardSettings(),
    val phase: ConnectionPhase = ConnectionPhase.DISCONNECTED,
    val message: String = "Wybierz kabel USB w ustawieniach",
    val telemetry: Telemetry? = null,
    val fresh: Boolean = false,
    val lastFrameMs: Long? = null,
    val frames: Long = 0,
    val timeouts: Int = 0,
    val rejectedFrames: Int = 0,
    val firmware: String = "",
    val trace: List<String> = emptyList(),
    val logFile: File? = null,
    val recording: RecordingSettings = RecordingSettings(),
    val recordingStatus: RecordingStatus = RecordingStatus(),
    val choosingRecordingDirectory: Boolean = false,
) {
    val busy: Boolean get() = phase in listOf(ConnectionPhase.PERMISSION, ConnectionPhase.CONNECTING,
        ConnectionPhase.INITIALIZING, ConnectionPhase.STREAMING, ConnectionPhase.STOPPING)
}

/** Process-shared state; the foreground service, rather than the activity, owns the USB session. */
class ConnectionController internal constructor(context: Context) {
    private val context = context.applicationContext
    private val controllerScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val usb = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private val preferences = context.getSharedPreferences("dashboard", Context.MODE_PRIVATE)
    private val mutableState = MutableStateFlow(DashboardState(settings = loadSettings(), recording = RecordingSettings(
        enabled = preferences.getBoolean("recordingEnabled", false),
        directoryUri = preferences.getString("recordingDirectory", null),
        directoryLabel = preferences.getString("recordingDirectoryLabel", "") ?: "",
    )))
    val state = mutableState.asStateFlow()
    private val csvStore = CsvDocumentStore(context.contentResolver)
    private val recordingMutex = Mutex()
    private var csvDocument: CsvDocument? = null
    private var csvSequence: Long = -1
    private var sessionJob: Job? = null
    private var pendingChoice: UsbChoice? = null
    private val permissionAction = "${context.packageName}.USB_PERMISSION"
    private var stopMessage = "Rozłączono"
    private var serviceFailure: String? = null

    private val permissionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != permissionAction) return
            val choice = pendingChoice ?: return
            @Suppress("DEPRECATION")
            val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
            if (device?.deviceId != choice.device.deviceId) return
            pendingChoice = null
            if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false) && usb.hasPermission(choice.device)) {
                open(choice)
            } else fail("Nie udzielono dostępu do USB. Wybierz Połącz, aby spróbować ponownie.")
        }
    }
    private val deviceReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            @Suppress("DEPRECATION")
            val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
            if (intent.action == UsbManager.ACTION_USB_DEVICE_DETACHED && device != null) {
                val selected = state.value.devices.find { it.key == state.value.selectedKey }
                if (selected?.device?.deviceId == device.deviceId) disconnect("Odłączono kabel USB")
            }
            refreshDevices()
        }
    }

    init {
        ContextCompat.registerReceiver(context, permissionReceiver, IntentFilter(permissionAction), ContextCompat.RECEIVER_NOT_EXPORTED)
        // These USB attachment actions are protected system broadcasts.
        ContextCompat.registerReceiver(context, deviceReceiver, IntentFilter().apply {
            addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }, ContextCompat.RECEIVER_EXPORTED)
        refreshDevices()
    }

    private fun loadSettings(): DashboardSettings {
        fun <T : Enum<T>> enumOrDefault(key: String, values: Array<T>, default: T): T =
            values.find { it.name == preferences.getString(key, null) } ?: default
        return DashboardSettings(
            DecoderConfig(
                enumOrDefault("map", MapMode.entries.toTypedArray(), MapMode.OBSERVED),
                enumOrDefault("pressure", GasPressureMode.entries.toTypedArray(), GasPressureMode.REGISTER_FALLBACK),
                preferences.getInt("analog", 0).takeIf { it in 1..6 },
                preferences.getBoolean("thirdTemperature", false)),
            preferences.getLong("interval", 200).coerceIn(100, 2000),
            preferences.getBoolean("controlLines", true), preferences.getBoolean("originalPolling", false),
            averageInjectionTimes = preferences.getBoolean("averageInjectionTimes", true),
            injectorCount = preferences.getInt("injectorCount", 4).coerceIn(1, 8))
    }

    fun updateSettings(settings: DashboardSettings) {
        if (state.value.busy) return
        preferences.edit().putString("map", settings.decoder.mapMode.name)
            .putString("pressure", settings.decoder.gasPressureMode.name)
            .putInt("analog", settings.decoder.analogMode ?: 0)
            .putBoolean("thirdTemperature", settings.decoder.thirdTemperatureFromPayload)
            .putLong("interval", settings.pollIntervalMs).putBoolean("controlLines", settings.assertControlLines)
            .putBoolean("originalPolling", settings.originalPolling)
            .putBoolean("averageInjectionTimes", settings.averageInjectionTimes)
            .putInt("injectorCount", settings.injectorCount.coerceIn(1, 8)).apply()
        mutableState.update { it.copy(settings = settings, fresh = false) }
    }

    fun setAverageInjectionTimes(enabled: Boolean) {
        preferences.edit().putBoolean("averageInjectionTimes", enabled).apply()
        mutableState.update { it.copy(settings = it.settings.copy(averageInjectionTimes = enabled)) }
    }

    fun setInjectorCount(count: Int) {
        require(count in 1..8)
        preferences.edit().putInt("injectorCount", count).apply()
        mutableState.update { it.copy(settings = it.settings.copy(injectorCount = count)) }
    }

    fun setRecordingEnabled(enabled: Boolean) {
        if (enabled && state.value.recording.directoryUri == null) {
            recordingPickerFailed("Najpierw wybierz folder zapisu CSV")
            return
        }
        preferences.edit().putBoolean("recordingEnabled", enabled).apply()
        mutableState.update { it.copy(recording = it.recording.copy(enabled = enabled, sequence = it.recording.sequence + 1),
            recordingStatus = it.recordingStatus.copy(error = null)) }
        if (!enabled) controllerScope.launch(Dispatchers.IO) {
            recordingMutex.withLock {
                if (!state.value.recording.enabled || csvSequence != state.value.recording.sequence) closeCsvLocked()
            }
        }
    }

    fun selectRecordingDirectory(uri: Uri) {
        if (state.value.busy || state.value.choosingRecordingDirectory) return
        mutableState.update { it.copy(choosingRecordingDirectory = true) }
        controllerScope.launch {
            try {
                val label = withContext(Dispatchers.IO) { csvStore.selectDirectory(uri) }
                val previous = state.value.recording.directoryUri
                preferences.edit().putString("recordingDirectory", uri.toString())
                    .putString("recordingDirectoryLabel", label).apply()
                mutableState.update { it.copy(recording = it.recording.copy(directoryUri = uri.toString(), directoryLabel = label),
                    recordingStatus = RecordingStatus()) }
                if (previous != null && previous != uri.toString()) {
                    withContext(Dispatchers.IO) { runCatching { csvStore.releaseDirectory(previous) } }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { recordingPickerFailed("Nie można wybrać folderu: ${e.message ?: e.javaClass.simpleName}") }
            finally { mutableState.update { it.copy(choosingRecordingDirectory = false) } }
        }
    }

    fun recordingPickerFailed(message: String) {
        mutableState.update { it.copy(recordingStatus = it.recordingStatus.copy(error = message)) }
    }

    internal suspend fun recordTelemetry(sample: CsvSample) = withContext(Dispatchers.IO) {
        recordingMutex.withLock {
            val config = state.value.recording
            if (!config.enabled) {
                closeCsvLocked()
                return@withLock
            }
            try {
                if (csvDocument != null && csvSequence != config.sequence && !closeCsvLocked()) return@withLock
                val directory = config.directoryUri ?: throw IOException("Wybierz folder zapisu CSV")
                val document = csvDocument ?: csvStore.create(directory, sample.timestampMs, sample.elapsedRealtimeMs).also {
                    csvDocument = it
                    csvSequence = config.sequence
                    mutableState.update { current -> current.copy(recordingStatus = RecordingStatus(active = true, fileName = it.name)) }
                }
                document.csv.append(sample)
                mutableState.update { it.copy(recordingStatus = it.recordingStatus.copy(rows = document.csv.rows)) }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                closeCsvLocked()
                recordingFailed("Zapis CSV zatrzymany: ${e.message ?: e.javaClass.simpleName}. Sprawdź folder i wolne miejsce.")
            }
        }
    }

    private fun recordingFailed(message: String) {
        preferences.edit().putBoolean("recordingEnabled", false).apply()
        mutableState.update { it.copy(recording = it.recording.copy(enabled = false),
            recordingStatus = it.recordingStatus.copy(active = false, error = message)) }
    }

    // Only called with recordingMutex held; closing never runs on the UI thread.
    private fun closeCsvLocked(): Boolean {
        val document = csvDocument ?: return true
        csvDocument = null
        var closed = true
        try { document.csv.close() }
        catch (e: Exception) {
            closed = false
            recordingFailed("Nie udało się zamknąć pliku CSV: ${e.message ?: e.javaClass.simpleName}")
        }
        mutableState.update { it.copy(recordingStatus = it.recordingStatus.copy(active = false)) }
        return closed
    }

    fun refreshDevices() {
        val choices = usb.deviceList.values.sortedBy { it.deviceName }.flatMap { device ->
            val driver = UsbSerialProber.getDefaultProber().probeDevice(device)
            driver?.ports?.indices?.map { UsbChoice(device, it, driver.javaClass.simpleName.removeSuffix("SerialDriver")) }
                ?: listOf(UsbChoice(device, null, null))
        }
        val previous = state.value.selectedKey
        val saved = preferences.getString("device", null)
        val candidates = choices.filter { it.portIndex != null && it.savedKey == saved }
        val selected = choices.find { it.key == previous }?.key ?: candidates.singleOrNull()?.key
        mutableState.update { it.copy(devices = choices, selectedKey = selected) }
    }

    fun selectDevice(key: String) {
        if (state.value.busy) return
        val choice = state.value.devices.find { it.key == key && it.portIndex != null } ?: return
        preferences.edit().putString("device", choice.savedKey).apply()
        mutableState.update { it.copy(selectedKey = key, message = "Wybrano ${choice.title}") }
    }

    fun connect() {
        if (state.value.busy || sessionJob != null || state.value.choosingRecordingDirectory) return
        refreshDevices()
        val choice = state.value.devices.find { it.key == state.value.selectedKey && it.portIndex != null }
            ?: return fail("Wybierz obsługiwany port USB w ustawieniach")
        if (usb.hasPermission(choice.device)) open(choice) else {
            pendingChoice = choice
            mutableState.update { it.copy(phase = ConnectionPhase.PERMISSION, message = "Oczekiwanie na zgodę Androida na dostęp do USB", fresh = false) }
            // USB service fills in EXTRA_DEVICE / EXTRA_PERMISSION_GRANTED when sending the result.
            val flags = PendingIntent.FLAG_UPDATE_CURRENT or
                if (Build.VERSION.SDK_INT >= 31) PendingIntent.FLAG_MUTABLE else 0
            val intent = PendingIntent.getBroadcast(context, choice.device.deviceId,
                Intent(permissionAction).setPackage(context.packageName), flags)
            try { usb.requestPermission(choice.device, intent) }
            catch (e: Exception) { pendingChoice = null; fail("Nie można uzyskać dostępu USB: ${e.message}") }
        }
    }

    fun disconnect(message: String = "Rozłączono") {
        stopMessage = message
        pendingChoice = null
        sessionJob?.cancel()
        mutableState.update { it.copy(phase = if (sessionJob == null) ConnectionPhase.DISCONNECTED else ConnectionPhase.STOPPING,
            message = message, fresh = false) }
    }

    private fun fail(message: String) {
        mutableState.update { it.copy(phase = ConnectionPhase.ERROR, message = message, fresh = false) }
    }

    private fun open(choice: UsbChoice) {
        if (sessionJob != null) return
        mutableState.update { it.copy(phase = ConnectionPhase.CONNECTING, message = "Otwieranie USB · 9600 8N1",
            telemetry = null, fresh = false, lastFrameMs = null, frames = 0, timeouts = 0, rejectedFrames = 0, firmware = "", trace = emptyList(), logFile = null) }
        try {
            ContextCompat.startForegroundService(context, Intent(context, ConnectionService::class.java)
                .setAction(ConnectionService.ACTION_CONNECT).putExtra(ConnectionService.EXTRA_DEVICE_KEY, choice.key))
        } catch (e: Exception) {
            serviceFailed("Nie można uruchomić połączenia w tle: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    internal fun hasUsbPermission(key: String?): Boolean = state.value.devices
        .find { it.key == key && it.portIndex != null }?.let { usb.hasPermission(it.device) } == true

    internal fun startSession(key: String?) {
        if (sessionJob != null || state.value.phase != ConnectionPhase.CONNECTING) return
        val choice = state.value.devices.find { it.key == key && it.portIndex != null }
            ?: return serviceFailed("Wybrany port USB nie jest dostępny")
        if (!usb.hasPermission(choice.device)) return serviceFailed("Brak dostępu do USB. Połącz ponownie i zaakceptuj zgodę.")
        val settings = state.value.settings
        serviceFailure = null
        sessionJob = controllerScope.launch {
            try {
                withContext(Dispatchers.IO) { runSession(choice, settings) }
            } catch (e: CancellationException) {
                mutableState.update { it.copy(phase = if (serviceFailure == null) ConnectionPhase.DISCONNECTED else ConnectionPhase.ERROR,
                    message = serviceFailure ?: stopMessage, fresh = false) }
                throw e
            } catch (e: Exception) { fail("Błąd połączenia: ${e.message ?: e.javaClass.simpleName}") }
            finally { sessionJob = null }
        }
    }

    internal fun serviceFailed(message: String) {
        pendingChoice = null
        if (sessionJob == null) fail(message) else {
            serviceFailure = message
            sessionJob?.cancel()
            mutableState.update { it.copy(phase = ConnectionPhase.STOPPING, message = message, fresh = false) }
        }
    }

    private suspend fun runSession(choice: UsbChoice, settings: DashboardSettings) {
        val driver: UsbSerialDriver = UsbSerialProber.getDefaultProber().probeDevice(choice.device)
            ?: throw IOException("Brak obsługi tego adaptera USB")
        val port = driver.ports.getOrNull(choice.portIndex ?: -1) ?: throw IOException("Nie znaleziono wybranego portu")
        val connection = usb.openDevice(choice.device) ?: throw IOException("Nie można otworzyć USB; sprawdź zgodę i kabel")
        val coroutine = currentCoroutineContext()
        var writer: java.io.BufferedWriter? = null
        try {
            port.open(connection)
            port.setParameters(9600, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
            // Drivers without control-line support can still exchange serial data.
            for (line in listOf("DTR", "RTS")) {
                try {
                    if (line == "DTR") port.setDTR(settings.assertControlLines) else port.setRTS(settings.assertControlLines)
                } catch (_: UnsupportedOperationException) { /* unsupported by this driver */ }
            }
            val directory = File(context.filesDir, "traces").apply { mkdirs() }
            directory.listFiles()?.sortedByDescending { it.lastModified() }?.drop(4)?.forEach { it.delete() }
            val log = File(directory, "diego-${SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.ROOT).format(Date())}.log")
            writer = log.bufferedWriter()
            writer.write("# GAS ECU · 9600 8N1 · ${choice.details} · $settings\n")
            writer.flush()
            mutableState.update { it.copy(logFile = log) }
            var loggedCharacters = 0
            val session = DiegoSession(object : SerialTransport {
                override fun write(bytes: ByteArray, timeoutMs: Int) { port.write(bytes, timeoutMs) }
                override fun read(buffer: ByteArray, timeoutMs: Int): Int = port.read(buffer, timeoutMs)
            }, trace = { direction, bytes ->
                val line = "${SystemClock.elapsedRealtime()} $direction ${bytes.hex()}"
                if (loggedCharacters < 4 * 1024 * 1024) {
                    writer.write(line + "\n"); writer.flush(); loggedCharacters += line.length + 1
                    if (loggedCharacters >= 4 * 1024 * 1024) { writer.write("# Limit 4 MiB; dalszy ruch nie jest zapisywany\n"); writer.flush() }
                }
                mutableState.update { it.copy(trace = (it.trace + line).takeLast(16)) }
            }, checkActive = { coroutine.ensureActive() })
            delay(100)
            session.drain()
            for ((index, request) in StartupSequence.commands.withIndex()) {
                mutableState.update { it.copy(phase = ConnectionPhase.INITIALIZING,
                    message = "Inicjalizacja ${index + 1}/${StartupSequence.commands.size} · ${request.wire.hex()}") }
                val reply = session.exchange(request)
                if (request.command == 0x02 && reply != null) {
                    val firmware = reply.copyOfRange(1, reply.size).toString(Charsets.US_ASCII)
                    mutableState.update { it.copy(firmware = firmware) }
                }
                if (request.pauseMs > 0) delay(request.pauseMs)
            }
            var failures = 0
            val poll = StartupCommand(DiegoProtocol.request(0x0B, 0x0B), 50)
            while (true) {
                coroutine.ensureActive()
                val start = SystemClock.elapsedRealtime()
                try {
                    val payload = session.exchange(poll)!!
                    // Never substitute emulator register constants for a live pressure measurement.
                    var r0d: Int? = null
                    var r0f: Int? = null
                    if ((payload[1].toInt() and 255) >= 0xFA && settings.decoder.gasPressureMode == GasPressureMode.REGISTER_FALLBACK) {
                        try {
                            r0d = DiegoProtocol.word(session.exchange(StartupCommand(DiegoProtocol.request(0x40, 0x0D), 3))!!, 1)
                            r0f = DiegoProtocol.word(session.exchange(StartupCommand(DiegoProtocol.request(0x40, 0x0F), 3))!!, 1)
                        } catch (_: ReplyTimeout) {
                            session.drain()
                            mutableState.update { it.copy(timeouts = it.timeouts + 1) }
                            r0d = null; r0f = null
                        }
                    }
                    coroutine.ensureActive()
                    val decoded = DiegoProtocol.decode(payload, TemperatureTable.values, settings.decoder, r0d, r0f)
                    val timestampMs = System.currentTimeMillis()
                    val receivedMs = SystemClock.elapsedRealtime()
                    mutableState.update { it.copy(phase = ConnectionPhase.STREAMING, message = "Odczyt na żywo · USB · 9600 8N1",
                        telemetry = decoded, fresh = true, lastFrameMs = receivedMs, frames = it.frames + 1,
                        rejectedFrames = session.rejectedFrames) }
                    val snapshot = state.value
                    recordTelemetry(CsvSample(decoded, timestampMs, receivedMs, snapshot.frames,
                        snapshot.settings.injectorCount, settings.decoder, snapshot.timeouts, snapshot.rejectedFrames, snapshot.firmware))
                    if (settings.originalPolling) {
                        // Remaining captured cyclic queries are only consumed; dashboard values come from 0B.
                        for (command in listOf(
                            StartupCommand(DiegoProtocol.request(0x40, 0x29), 3),
                            StartupCommand(DiegoProtocol.request(0x64, 0xA5), 12),
                            StartupCommand(DiegoProtocol.request(0x40, 0x28), 3),
                            StartupCommand(DiegoProtocol.request(0x53, 0x00), 2),
                            StartupCommand(DiegoProtocol.request(0x40, 0x0D), 3),
                            StartupCommand(DiegoProtocol.request(0x40, 0x0F), 3),
                            StartupCommand(DiegoProtocol.request(0x3F, 0x00), 2),
                        )) session.exchange(command)
                    }
                    failures = 0
                } catch (e: ReplyTimeout) {
                    failures++
                    mutableState.update { it.copy(fresh = false, message = "Brak nowych danych ($failures/3) · ${e.message}",
                        timeouts = it.timeouts + 1, rejectedFrames = session.rejectedFrames) }
                    if (failures >= 3) throw IOException("Trzy kolejne timeouty. Sprawdź kabel i zasilanie ECU, następnie połącz ponownie.")
                    session.drain()
                }
                delay((settings.pollIntervalMs - (SystemClock.elapsedRealtime() - start)).coerceAtLeast(10))
            }
        } finally {
            withContext(NonCancellable + Dispatchers.IO) { recordingMutex.withLock { closeCsvLocked() } }
            runCatching { writer?.close() }
            runCatching { port.close() }
            connection.close()
        }
    }

    internal suspend fun close() {
        disconnect()
        sessionJob?.join()
        withContext(NonCancellable + Dispatchers.IO) { recordingMutex.withLock { closeCsvLocked() } }
        context.unregisterReceiver(permissionReceiver)
        context.unregisterReceiver(deviceReceiver)
        controllerScope.cancel()
    }

    companion object {
        @Volatile private var instance: ConnectionController? = null
        fun get(context: Context): ConnectionController = instance ?: synchronized(this) {
            instance ?: ConnectionController(context).also { instance = it }
        }
    }
}
