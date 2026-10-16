package com.blazej222.lpgdashboard.protocol

import kotlin.math.max
import kotlin.math.round

enum class MapMode { UNKNOWN, OBSERVED, ALTERNATE }
enum class GasPressureMode { UNKNOWN, REGISTER_FALLBACK, STANDARD }

data class DecoderConfig(
    val mapMode: MapMode = MapMode.OBSERVED,
    val gasPressureMode: GasPressureMode = GasPressureMode.REGISTER_FALLBACK,
    val analogMode: Int? = null,
    val thirdTemperatureFromPayload: Boolean = false,
) // Defaults explicitly select the handoff's measured installation profile, not ECU autodetection.

data class Reading(
    val raw: Int,
    val decoded: Double?,
    val unit: String,
    val confidence: String,
    val configMode: String? = null,
    val source: String,
)

data class Telemetry(
    val rawPayload: List<Int>,
    val rpm: Reading,
    val reducerTemperature: Reading,
    val gasTemperature: Reading,
    val gasPressure: Reading,
    val mapPressure: Reading,
    val voltageCandidate: Reading,
    val levelRaw: Reading,
    val analog: Reading,
    val thirdTemperature: Reading,
    val petrol: List<Reading>,
    val gas: List<Reading>,
    val stateCode: Int,
    val gasIconRed: Boolean,
) {
    val stateLabel: String get() = listOf(
        "Wyłączony", "Zapłon (1)", "Zapłon (2)", "Benzyna · ciśnienie czerwone",
        "Oczekiwanie", "Gaz", "Benzyna (6)", "Benzyna (7)",
    )[stateCode]
}

object DiegoProtocol {
    val petrolOffsets = listOf(9, 11, 13, 15, 25, 27, 29, 31)
    val gasOffsets = listOf(17, 19, 21, 23, 33, 35, 37, 39)
    fun request(command: Int, argument: Int): ByteArray {
        require(command in 0..255 && argument in 0..255)
        return byteArrayOf(0xC0.toByte(), command.toByte(), argument.toByte(),
            ((0xC0 + command + argument) and 255).toByte())
    }
    fun checksum(payload: ByteArray): Int = (payload.sumOf { it.toInt() and 255 } - 0x30) and 255
    fun frame(payload: ByteArray): ByteArray = byteArrayOf(0xD0.toByte()) + payload + checksum(payload).toByte()
    fun word(bytes: ByteArray, offset: Int): Int =
        ((bytes[offset].toInt() and 255) shl 8) or (bytes[offset + 1].toInt() and 255)

    fun decode(payload: ByteArray, temperatures: DoubleArray, config: DecoderConfig,
               r0d: Int? = null, r0f: Int? = null): Telemetry {
        require(payload.size == 50 && payload[0] == 0x0B.toByte())
        require(temperatures.size == 256)
        require(r0d == null || r0d in 0..65535)
        require(r0f == null || r0f in 0..65535)
        val raw = payload.map { it.toInt() and 255 }
        fun field(i: Int, value: Double?, unit: String, confidence: String,
                  mode: String? = null) = Reading(raw[i], value, unit, confidence, mode, "0B[$i]")
        val n = word(payload, 7)
        val pressure = when {
            config.gasPressureMode == GasPressureMode.UNKNOWN -> null
            raw[1] < 0xFA || config.gasPressureMode == GasPressureMode.STANDARD ->
                max(0.0, (raw[1] / 256.0 - 0.04) / 0.369)
            r0d == null || r0f == null -> null
            r0f > r0d -> 0.0
            else -> max(0.0, ((r0d - r0f) / 256.0 + 0.00842) / 0.242)
        }
        val map = when (config.mapMode) {
            MapMode.UNKNOWN -> null
            MapMode.OBSERVED -> max(0.0, (raw[41] / 256.0 + 0.00842) / 0.242)
            MapMode.ALTERNATE -> max(0.0, (raw[41] / 256.0 - 0.04) / 0.369)
        }
        fun injections(offsets: List<Int>, petrol: Boolean) = offsets.mapIndexed { index, offset ->
            val w = word(payload, offset)
            val mask = if (!petrol) 0xFFFF else when(index) { 0 -> 0x1FFF; 1 -> 0x7FFF; else -> 0xFFFF }
            Reading(w, (w and mask) / 187.0, "ms", if (petrol && index < 2) "X/M" else "X",
                source = "0B[$offset..${offset + 1}]")
        }
        return Telemetry(raw,
            Reading(n, if (n == 0) 0.0 else 10.0 * round(2_244_000.0 / n), "obr/min", "X/M", source = "0B[7..8]"),
            field(2, temperatures[raw[2]], "°C", "X/M"),
            field(3, temperatures[raw[3]], "°C", "X"),
            field(1, pressure, "bar", "X/M", config.gasPressureMode.name).copy(
                source = if (raw[1] >= 0xFA && config.gasPressureMode == GasPressureMode.REGISTER_FALLBACK)
                    "0B[1], 40/0D=${r0d ?: "?"}, 40/0F=${r0f ?: "?"}" else "0B[1]"),
            field(41, map, "bar", if (config.mapMode == MapMode.OBSERVED) "X/M" else "X", config.mapMode.name),
            field(4, raw[4] * 5.0 / 256.0 / 0.215, "V", "X; nazwa H"),
            field(5, null, "RAW", "N"),
            field(6, if (config.analogMode in 1..6) raw[6] * 0.02 else null, "V", "X warunkowe; nazwa H", config.analogMode?.toString()),
            field(47, if (config.thirdTemperatureFromPayload) temperatures[raw[47]] else null, "°C", "X warunkowe; nazwa H",
                if (config.thirdTemperatureFromPayload) "payload[47]" else "nieustalony"),
            injections(petrolOffsets, true), injections(gasOffsets, false),
            (word(payload, 9) shr 13) and 7, word(payload, 11) and 0x8000 != 0)
    }
}

/** A fixed-length, checksum-checked stream reader. Markers inside a payload are data. */
class ReplyAccumulator(private val payloadSize: Int, private val command: Int, private val argument: Int? = null) {
    private var buffer = byteArrayOf()
    var rejectedCandidates: Int = 0
        private set
    init { require(payloadSize in 1..1024) }
    fun feed(chunk: ByteArray): List<ByteArray> {
        buffer += chunk
        val replies = mutableListOf<ByteArray>()
        val size = payloadSize + 2
        while (buffer.isNotEmpty()) {
            val start = buffer.indexOf(0xD0.toByte())
            if (start < 0) { buffer = byteArrayOf(); break }
            if (start > 0) buffer = buffer.copyOfRange(start, buffer.size)
            if (buffer.size < 2) break
            if ((buffer[1].toInt() and 255) != command) {
                buffer = buffer.copyOfRange(1, buffer.size); continue
            }
            if (buffer.size < size) break
            val payload = buffer.copyOfRange(1, size - 1)
            if (DiegoProtocol.checksum(payload) == (buffer[size - 1].toInt() and 255) &&
                (argument == null || (payload[1].toInt() and 255) == argument)) {
                replies += payload
                buffer = buffer.copyOfRange(size, buffer.size)
            } else {
                rejectedCandidates++
                buffer = buffer.copyOfRange(1, buffer.size)
            }
        }
        return replies
    }
}
