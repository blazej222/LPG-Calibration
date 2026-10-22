package com.blazej222.lpgdashboard

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel

/** UI facade. Clearing an activity's ViewModel must not stop the service-owned session. */
class ConnectionViewModel(application: Application) : AndroidViewModel(application) {
    private val controller = ConnectionController.get(application)
    val state = controller.state

    fun connect() = controller.connect()
    fun disconnect(message: String = "Rozłączono") = controller.disconnect(message)
    fun refreshDevices() = controller.refreshDevices()
    fun selectDevice(key: String) = controller.selectDevice(key)
    fun updateSettings(settings: DashboardSettings) = controller.updateSettings(settings)
    fun setAverageInjectionTimes(enabled: Boolean) = controller.setAverageInjectionTimes(enabled)
    fun setInjectorCount(count: Int) = controller.setInjectorCount(count)
    fun setRecordingEnabled(enabled: Boolean) = controller.setRecordingEnabled(enabled)
    fun selectRecordingDirectory(uri: Uri) = controller.selectRecordingDirectory(uri)
    fun recordingPickerFailed(message: String) = controller.recordingPickerFailed(message)
}
