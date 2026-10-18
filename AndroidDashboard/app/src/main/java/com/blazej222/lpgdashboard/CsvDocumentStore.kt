package com.blazej222.lpgdashboard

import android.content.ContentResolver
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import java.io.IOException
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

data class RecordingSettings(
    val enabled: Boolean = false,
    val directoryUri: String? = null,
    val directoryLabel: String = "",
    val sequence: Long = 0,
)

data class RecordingStatus(
    val active: Boolean = false,
    val fileName: String? = null,
    val rows: Long = 0,
    val error: String? = null,
)

internal data class CsvDocument(val name: String, val csv: TelemetryCsvWriter)

/** Uses a persisted system-picker grant, without a broad storage permission. */
internal class CsvDocumentStore(private val resolver: ContentResolver) {
    fun selectDirectory(uri: Uri): String {
        val directory = DocumentsContract.buildDocumentUriUsingTree(uri, DocumentsContract.getTreeDocumentId(uri))
        val label = resolver.query(directory, arrayOf(
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
            DocumentsContract.Document.COLUMN_FLAGS,
        ), null, null, null)?.use { cursor ->
            if (!cursor.moveToFirst() || cursor.getString(1) != DocumentsContract.Document.MIME_TYPE_DIR ||
                cursor.getInt(2) and DocumentsContract.Document.FLAG_DIR_SUPPORTS_CREATE == 0) {
                throw IOException("Ten folder nie pozwala tworzyć plików")
            }
            cursor.getString(0)
        } ?: throw IOException("Nie można odczytać wybranego folderu")
        resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        return label
    }

    fun releaseDirectory(uri: String) {
        resolver.releasePersistableUriPermission(Uri.parse(uri),
            Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
    }

    fun create(directoryUri: String, timestampMs: Long, elapsedMs: Long): CsvDocument {
        val tree = Uri.parse(directoryUri)
        if (resolver.persistedUriPermissions.none { it.uri == tree && it.isWritePermission }) {
            throw IOException("Brak dostępu do folderu. Wybierz folder ponownie")
        }
        val directory = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val time = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS").withZone(ZoneOffset.UTC)
            .format(Instant.ofEpochMilli(timestampMs))
        val name = "gas-ecu-$time-${UUID.randomUUID().toString().take(8)}.csv"
        val uri = DocumentsContract.createDocument(resolver, directory, "text/csv", name)
            ?: throw IOException("Nie można utworzyć pliku CSV")
        val output = resolver.openOutputStream(uri, "w")?.bufferedWriter(Charsets.UTF_8)
            ?: throw IOException("Nie można otworzyć pliku CSV do zapisu")
        try {
            return CsvDocument(name, TelemetryCsvWriter(output, elapsedMs))
        } catch (e: Exception) {
            runCatching { output.close() }
            throw e
        }
    }
}
