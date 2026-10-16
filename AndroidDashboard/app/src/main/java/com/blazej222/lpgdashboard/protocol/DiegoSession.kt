package com.blazej222.lpgdashboard.protocol

import java.io.IOException

data class StartupCommand(val wire: ByteArray, val payloadSize: Int, val pauseMs: Long = 0) {
    val command: Int get() = wire[1].toInt() and 255
    val echoedArgument: Int? get() = if (command in listOf(0x07, 0x12, 0x51)) wire[2].toInt() and 255 else null
}

interface SerialTransport {
    fun write(bytes: ByteArray, timeoutMs: Int)
    fun read(buffer: ByteArray, timeoutMs: Int): Int
}

class ReplyTimeout(message: String) : IOException(message)

/** One outstanding request at a time; no dependence on USB packet boundaries. */
class DiegoSession(
    private val transport: SerialTransport,
    private val trace: (String, ByteArray) -> Unit = { _, _ -> },
    private val checkActive: () -> Unit = {},
    private val nanoTime: () -> Long = System::nanoTime,
) {
    var rejectedFrames = 0
        private set
    fun exchange(request: StartupCommand, timeoutMs: Int = 1000): ByteArray? {
        checkActive()
        transport.write(request.wire, 1000)
        trace("TX", request.wire)
        if (request.payloadSize == 0) return null
        val parser = ReplyAccumulator(request.payloadSize, request.command, request.echoedArgument)
        val buffer = ByteArray(512)
        val deadline = nanoTime() + timeoutMs * 1_000_000L
        try {
            while (nanoTime() < deadline) {
                checkActive()
                val size = transport.read(buffer, 100)
                if (size > 0) {
                    val chunk = buffer.copyOf(size)
                    trace("RX", chunk)
                    parser.feed(chunk).firstOrNull()?.let { return it }
                }
            }
            throw ReplyTimeout("Brak odpowiedzi na ${request.wire.hex()} (${timeoutMs} ms)")
        } finally { rejectedFrames += parser.rejectedCandidates }
    }

    // Clear late fragments before retrying; the wire protocol has no transaction identifier.
    fun drain() {
        val buffer = ByteArray(512)
        val deadline = nanoTime() + 500_000_000L
        while (nanoTime() < deadline) {
            checkActive()
            val count = transport.read(buffer, 100)
            if (count == 0) return
            trace("RX (spóźnione)", buffer.copyOf(count))
        }
        throw IOException("Port nie osiągnął przerwy w odbiorze; połącz ponownie")
    }
}

fun ByteArray.hex(): String = joinToString(" ") { "%02X".format(it.toInt() and 255) }
