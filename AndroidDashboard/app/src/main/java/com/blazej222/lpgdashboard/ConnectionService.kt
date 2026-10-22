package com.blazej222.lpgdashboard

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.os.IBinder
import android.os.PowerManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.map

/** Started only after USB permission, while handling the user's Connect action. */
class ConnectionService : Service() {
    private lateinit var controller: ConnectionController
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var observer: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var foreground = false
    private var latestStartId = 0

    override fun onCreate() {
        super.onCreate()
        controller = ConnectionController.get(this)
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Połączenie ze sterownikiem LPG", NotificationManager.IMPORTANCE_LOW)
        )
    }

    @SuppressLint("WakelockTimeout") // Held for the explicit USB session; released in onDestroy after disconnect.
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        latestStartId = startId
        if (intent?.action == ACTION_DISCONNECT) {
            controller.disconnect("Rozłączono z powiadomienia")
            if (!controller.state.value.busy) finishService()
            return START_NOT_STICKY
        }
        val key = intent?.getStringExtra(EXTRA_DEVICE_KEY)
        if (intent?.action != ACTION_CONNECT || !controller.hasUsbPermission(key)) {
            controller.serviceFailed("Kabel USB jest niedostępny lub cofnięto zgodę na dostęp")
            finishService()
            return START_NOT_STICKY
        }
        try {
            if (!foreground) {
                startForeground(NOTIFICATION_ID, notification(controller.state.value), ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
                foreground = true
                wakeLock = getSystemService(PowerManager::class.java).newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK, "$packageName:DiegoUsbSession"
                ).apply { setReferenceCounted(false); acquire() }
            }
            controller.startSession(key)
            if (observer == null) observer = scope.launch {
                // USB traces and frame counters change often; notify only when user-visible status changes.
                controller.state.map { NotificationState(it.phase, it.recordingStatus.active,
                    it.recordingStatus.fileName, it.recordingStatus.error) }.distinctUntilChanged().collect {
                    val state = controller.state.value
                    if (!state.busy) finishService()
                    else getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(state))
                }
            }
        } catch (e: Exception) {
            controller.serviceFailed("Nie można uruchomić połączenia w tle: ${e.message ?: e.javaClass.simpleName}")
            finishService()
        }
        return START_NOT_STICKY
    }

    private fun notification(state: DashboardState): Notification {
        val open = PendingIntent.getActivity(this, 0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 1,
            Intent(this, ConnectionService::class.java).setAction(ACTION_DISCONNECT),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val text = when (state.phase) {
            ConnectionPhase.CONNECTING -> "Otwieranie połączenia USB"
            ConnectionPhase.INITIALIZING -> "Inicjalizacja sterownika LPG"
            ConnectionPhase.STOPPING -> "Zamykanie połączenia i pliku CSV"
            else -> when {
                state.recordingStatus.error != null -> "Odczyt USB w tle · błąd zapisu CSV"
                state.recordingStatus.active -> "Odczyt USB i rejestracja CSV w tle"
                else -> "Odczyt parametrów LPG w tle"
            }
        }
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_connection_notification)
            .setContentTitle("GAS ECU · Diego G3")
            .setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text +
                if (state.recordingStatus.active) "\n${state.recordingStatus.fileName}" else ""))
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(Icon.createWithResource(this, android.R.drawable.ic_menu_close_clear_cancel),
                "Rozłącz", stop).build())
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .build()
    }

    private fun finishService() {
        if (foreground) stopForeground(STOP_FOREGROUND_REMOVE)
        foreground = false
        stopSelfResult(latestStartId)
    }

    override fun onDestroy() {
        if (controller.state.value.busy) controller.disconnect("Usługa połączenia została zatrzymana")
        scope.cancel()
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private data class NotificationState(val phase: ConnectionPhase, val recording: Boolean, val fileName: String?, val error: String?)

    companion object {
        const val ACTION_CONNECT = "com.blazej222.lpgdashboard.CONNECT"
        const val ACTION_DISCONNECT = "com.blazej222.lpgdashboard.DISCONNECT"
        const val EXTRA_DEVICE_KEY = "deviceKey"
        private const val CHANNEL_ID = "diego_usb_connection"
        private const val NOTIFICATION_ID = 1
    }
}
