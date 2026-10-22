package com.blazej222.lpgdashboard

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.core.content.FileProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.ViewModelProvider
import com.blazej222.lpgdashboard.ui.theme.LPGDashboardTheme
import java.io.File

class MainActivity : ComponentActivity() {
    private var connection: ConnectionViewModel? = null
    private val recordingDirectoryPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) connection?.selectRecordingDirectory(uri)
    }
    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        // Denial hides the drawer notification on Android 13+, but must not prevent the USB session.
        connection?.connect()
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val model = ViewModelProvider(this)[ConnectionViewModel::class.java]
        connection = model
        handleUsbAttachment(intent)
        setContent {
            val state by model.state.collectAsStateWithLifecycle()
            DisposableEffect(state.busy) {
                if (state.busy) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                onDispose { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
            LPGDashboardTheme(darkTheme = true, dynamicColor = false) {
                DashboardApp(state, model, ::shareLog, ::chooseRecordingDirectory, ::connect)
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleUsbAttachment(intent)
    }

    private fun handleUsbAttachment(intent: Intent?) {
        if (intent?.action != UsbManager.ACTION_USB_DEVICE_ATTACHED) return
        @Suppress("DEPRECATION")
        val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE) ?: return
        connection?.selectAttachedDevice(device)
    }

    private fun chooseRecordingDirectory() {
        val model = connection ?: return
        if (model.state.value.busy || model.state.value.choosingRecordingDirectory) return
        try {
            recordingDirectoryPicker.launch(model.state.value.recording.directoryUri?.let(android.net.Uri::parse))
        } catch (e: Exception) {
            model.recordingPickerFailed("Nie można otworzyć wyboru folderu: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    private fun connect() {
        val model = connection ?: return
        val preferences = getSharedPreferences("dashboard", MODE_PRIVATE)
        if (model.state.value.selectedKey != null && Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED &&
            !preferences.getBoolean("notificationPermissionRequested", false)) {
            preferences.edit().putBoolean("notificationPermissionRequested", true).apply()
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            model.connect()
        }
    }
    private fun shareLog(file: File) {
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Zapisz lub udostępnij log komunikacji"))
    }
}
