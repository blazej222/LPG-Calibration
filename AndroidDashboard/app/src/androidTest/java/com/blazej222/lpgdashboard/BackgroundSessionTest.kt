package com.blazej222.lpgdashboard

import android.app.Notification
import android.app.NotificationManager
import android.content.Context
import android.os.PowerManager
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test

/** Opt-in integration test against a connected laboratory ECU emulator and a selected CSV folder. */
class BackgroundSessionTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context get() = instrumentation.targetContext
    private fun awaitCondition(message: String, timeoutMs: Long = 10_000, condition: () -> Boolean) {
        val deadline = System.nanoTime() + timeoutMs * 1_000_000L
        while (!condition() && System.nanoTime() < deadline) Thread.sleep(50)
        assertTrue(message, condition())
    }
    private fun shell(command: String) {
        instrumentation.uiAutomation.executeShellCommand(command).use { descriptor ->
            android.os.ParcelFileDescriptor.AutoCloseInputStream(descriptor).use { it.readBytes() }
        }
    }

    @Test fun sameUsbSessionAndCsvSurviveBackgroundScreenOffAndActivityDestruction() {
        assumeTrue("Requires a connected ECU emulator and CSV recording; pass backgroundSessionTest=true",
            InstrumentationRegistry.getArguments().getString("backgroundSessionTest") == "true")
        val controller = ConnectionController.get(context)
        ActivityScenario.launch(MainActivity::class.java).use {
            // Instrumentation starts a fresh process. Allow setup via the real UI/USB permission dialog.
            awaitCondition("Connect the ECU emulator and enable CSV recording", 180_000) {
                controller.state.value.phase == ConnectionPhase.STREAMING && controller.state.value.recordingStatus.active
            }
        }
        val originalFile = controller.state.value.recordingStatus.fileName
        val originalLog = controller.state.value.logFile
        fun checkProgress() {
            val frame = controller.state.value.frames
            val rows = controller.state.value.recordingStatus.rows
            awaitCondition("Telemetry and CSV stopped in background") {
                controller.state.value.frames >= frame + 3 && controller.state.value.recordingStatus.rows >= rows + 3
            }
            assertEquals(ConnectionPhase.STREAMING, controller.state.value.phase)
            assertEquals(originalFile, controller.state.value.recordingStatus.fileName)
            assertEquals(originalLog, controller.state.value.logFile)
        }

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.moveToState(Lifecycle.State.CREATED)
            checkProgress()
            try {
                shell("input keyevent KEYCODE_SLEEP")
                awaitCondition("Screen did not switch off") { !context.getSystemService(PowerManager::class.java).isInteractive }
                checkProgress()
            } finally {
                shell("input keyevent KEYCODE_WAKEUP")
                shell("wm dismiss-keyguard")
            }
            scenario.moveToState(Lifecycle.State.RESUMED)
            scenario.recreate()
            checkProgress()
        }
        // Both the original activity and its ViewModel are now destroyed.
        checkProgress()
        ActivityScenario.launch(MainActivity::class.java).use {
            checkProgress()
            val notifications = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val notification = notifications.activeNotifications.first { it.notification.channelId == "diego_usb_connection" }.notification
            assertTrue(notification.flags and Notification.FLAG_FOREGROUND_SERVICE != 0)
            val stop = notification.actions.first { it.title.toString() == "Rozłącz" }
            stop.actionIntent.send()
            awaitCondition("Notification action did not close USB and CSV") {
                controller.state.value.phase == ConnectionPhase.DISCONNECTED && !controller.state.value.recordingStatus.active
            }
            awaitCondition("Foreground notification was not removed") {
                notifications.activeNotifications.none { it.notification.channelId == "diego_usb_connection" }
            }
            assertEquals(originalFile, controller.state.value.recordingStatus.fileName)
        }
    }
}
