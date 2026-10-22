package com.blazej222.lpgdashboard

import android.app.Application
import android.net.Uri
import android.provider.DocumentsContract
import androidx.test.platform.app.InstrumentationRegistry
import com.blazej222.lpgdashboard.protocol.*
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test

/** Storage integration only: feeds decoded fixture samples and never opens an ECU/USB port. */
class CsvRecordingStorageTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val application get() = instrumentation.targetContext.applicationContext as Application
    private fun main(action: () -> Unit) = instrumentation.runOnMainSync(action)
    private fun createController(): ConnectionController {
        lateinit var result: ConnectionController
        main { result = ConnectionController(application) }
        return result
    }
    private fun close(controller: ConnectionController) = runBlocking {
        withContext(Dispatchers.Main) { controller.close() }
    }
    private fun sample(frame: Long = 1): CsvSample {
        val hex = "0B 79 69 93 9B 6B 17 65 BA A1 76 81 76 01 76 01 76 01 76 01 76 01 76 01 76 " +
            "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 17 72 00 00 FF 01 88 00 02"
        val payload = hex.split(' ').map { it.toInt(16).toByte() }.toByteArray()
        val config = DecoderConfig()
        return CsvSample(DiegoProtocol.decode(payload, TemperatureTable.values, config),
            System.currentTimeMillis(), android.os.SystemClock.elapsedRealtime(), frame, 4, config, 0, 0, "CSV storage test")
    }
    private fun awaitCondition(condition: () -> Boolean) {
        val deadline = System.nanoTime() + 5_000_000_000L
        while (!condition() && System.nanoTime() < deadline) Thread.sleep(10)
        assertTrue("Timed out waiting for recording state", condition())
    }
    private fun readCsv(tree: Uri, name: String): String {
        val resolver = application.contentResolver
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        resolver.query(children, arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)!!.use { cursor ->
            while (cursor.moveToNext()) {
                if (cursor.getString(1) == name) {
                    val document = DocumentsContract.buildDocumentUriUsingTree(tree, cursor.getString(0))
                    return resolver.openInputStream(document)!!.bufferedReader(Charsets.UTF_8).use { it.readText() }
                }
            }
        }
        error("CSV file not found: $name")
    }

    @Test fun selectedSafFolderSupportsRecordingRestartAndSavedSettings() {
        val uriText = InstrumentationRegistry.getArguments().getString("csvTestDirectoryUri")
        assumeTrue("Choose a test folder in the app and pass csvTestDirectoryUri to run this storage test", uriText != null)
        val tree = Uri.parse(uriText!!)
        val model = createController()
        try {
            main { model.selectRecordingDirectory(tree) }
            awaitCondition { !model.state.value.choosingRecordingDirectory }
            assertNull(model.state.value.recordingStatus.error)
            assertEquals(uriText, model.state.value.recording.directoryUri)
            main { model.setRecordingEnabled(true) }
            runBlocking { model.recordTelemetry(sample()); model.recordTelemetry(sample(2)) }
            val firstName = model.state.value.recordingStatus.fileName!!
            assertEquals(2L, model.state.value.recordingStatus.rows)
            main { model.setRecordingEnabled(false) }
            awaitCondition { !model.state.value.recordingStatus.active }
            val content = readCsv(tree, firstName)
            assertTrue(content.startsWith("\uFEFFschema_version;timestamp_utc;"))
            assertEquals(3, content.trimEnd().lines().size)
            assertTrue(content.contains(";860.0;"))

            main { model.setRecordingEnabled(true) }
            runBlocking { model.recordTelemetry(sample(3)) }
            val secondName = model.state.value.recordingStatus.fileName!!
            assertNotEquals(firstName, secondName)
            // Exercise rapid off/on as well as ordinary stop: an earlier file must not be reused.
            main { model.setRecordingEnabled(false); model.setRecordingEnabled(true) }
            runBlocking { model.recordTelemetry(sample(4)) }
            assertNotEquals(secondName, model.state.value.recordingStatus.fileName)
            main { model.setRecordingEnabled(false) }
            awaitCondition { !model.state.value.recordingStatus.active }
            assertEquals(2, readCsv(tree, secondName).trimEnd().lines().size)
        } finally {
            main { model.setRecordingEnabled(false) }
            awaitCondition { !model.state.value.recordingStatus.active }
            close(model)
        }
        val restored = createController()
        try {
            assertEquals(uriText, restored.state.value.recording.directoryUri)
            assertFalse(restored.state.value.recording.enabled)
        } finally { close(restored) }
    }

    @Test fun restoredFolderWithoutGrantStopsCsvWithoutChangingConnectionPhase() {
        val preferences = application.getSharedPreferences("dashboard", 0)
        val previousDirectory = preferences.getString("recordingDirectory", null)
        val previousEnabled = preferences.getBoolean("recordingEnabled", false)
        preferences.edit().putString("recordingDirectory",
            "content://com.android.externalstorage.documents/tree/primary%3A__missing_csv_test__")
            .putBoolean("recordingEnabled", true).commit()
        val model = createController()
        try {
            val phase = model.state.value.phase
            runBlocking { model.recordTelemetry(sample()) }
            assertFalse(model.state.value.recording.enabled)
            assertFalse(model.state.value.recordingStatus.active)
            assertNotNull(model.state.value.recordingStatus.error)
            assertEquals(phase, model.state.value.phase)
        } finally {
            close(model)
            preferences.edit().putString("recordingDirectory", previousDirectory)
                .putBoolean("recordingEnabled", previousEnabled).commit()
        }
    }
}
