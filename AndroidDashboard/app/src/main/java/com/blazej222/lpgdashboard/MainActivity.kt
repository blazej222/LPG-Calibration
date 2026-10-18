package com.blazej222.lpgdashboard

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.blazej222.lpgdashboard.ui.theme.LPGDashboardTheme
import java.io.File

class MainActivity : ComponentActivity() {
    private var connection: ConnectionViewModel? = null
    private val recordingDirectoryPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) connection?.selectRecordingDirectory(uri)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val model: ConnectionViewModel = viewModel()
            connection = model
            val state by model.state.collectAsStateWithLifecycle()
            DisposableEffect(state.busy) {
                if (state.busy) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                onDispose { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
            LPGDashboardTheme(darkTheme = true, dynamicColor = false) {
                DashboardApp(state, model, ::shareLog, ::chooseRecordingDirectory)
            }
        }
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

    override fun onStop() {
        if (!isChangingConfigurations && connection?.state?.value?.busy == true) {
            connection?.disconnect("Odczyt zatrzymany · aplikacja w tle")
        }
        super.onStop()
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
