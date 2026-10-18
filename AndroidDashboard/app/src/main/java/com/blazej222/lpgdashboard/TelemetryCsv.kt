package com.blazej222.lpgdashboard

import com.blazej222.lpgdashboard.protocol.DecoderConfig
import com.blazej222.lpgdashboard.protocol.Telemetry
import com.blazej222.lpgdashboard.protocol.hex
import java.io.Closeable
import java.io.Writer
import java.time.Instant

internal data class CsvSample(
    val telemetry: Telemetry,
    val timestampMs: Long,
    val elapsedRealtimeMs: Long,
    val frameNumber: Long,
    val injectorCount: Int,
    val decoder: DecoderConfig,
    val timeouts: Int,
    val rejectedFrames: Int,
    val firmware: String,
)

/** UTF-8 is chosen by the caller. A fixed schema survives changes to the displayed injector count. */
internal class TelemetryCsvWriter(private val output: Writer, private val startElapsedMs: Long) : Closeable {
    var rows: Long = 0
        private set

    init {
        output.write("\uFEFF")
        writeRow(headers)
        output.flush()
    }

    fun append(sample: CsvSample) {
        require(sample.injectorCount in 1..8)
        val t = sample.telemetry
        val values = mutableListOf<Any?>(
            1, Instant.ofEpochMilli(sample.timestampMs).toString(),
            (sample.elapsedRealtimeMs - startElapsedMs).coerceAtLeast(0), sample.frameNumber,
            t.rpm.decoded, t.gasPressure.decoded, t.mapPressure.decoded,
            t.reducerTemperature.decoded, t.gasTemperature.decoded, t.batteryVoltage.decoded,
            t.stateCode, t.stateLabel, sample.injectorCount,
            t.petrol.averageInjectionTimeMs(sample.injectorCount), t.gas.averageInjectionTimeMs(sample.injectorCount),
        )
        for (readings in listOf(t.petrol, t.gas)) {
            for (index in 0 until 8) values += if (index < sample.injectorCount) readings.getOrNull(index)?.decoded else null
        }
        values.addAll(listOf(
            t.levelRaw.raw, t.analog.decoded, t.analog.raw, t.thirdTemperature.decoded, t.thirdTemperature.raw,
            t.gasIconRed, sample.decoder.mapMode.name, sample.decoder.gasPressureMode.name,
            sample.decoder.analogMode, sample.decoder.thirdTemperatureFromPayload,
            t.analog.confidence, t.thirdTemperature.confidence, t.gasPressure.source,
            sample.timeouts, sample.rejectedFrames, sample.firmware,
            t.rawPayload.map { it.toByte() }.toByteArray().hex(),
        ))
        writeRow(values)
        output.flush() // Commit every sample instead of buffering a whole journey.
        rows++
    }

    private fun writeRow(values: List<Any?>) {
        output.write(values.joinToString(";") { value ->
            val text = value?.toString() ?: ""
            if (text.any { it == ';' || it == '"' || it == '\r' || it == '\n' })
                "\"${text.replace("\"", "\"\"")}\"" else text
        })
        output.write("\r\n")
    }

    override fun close() = output.close()

    companion object {
        val headers = listOf(
            "schema_version", "timestamp_utc", "elapsed_ms", "frame_number",
            "rpm", "gas_pressure_bar", "map_pressure_bar", "reducer_temperature_c", "gas_temperature_c", "battery_voltage_v",
            "state_code", "state_label", "injector_count", "petrol_average_ms", "gas_average_ms",
        ) + (1..8).map { "petrol_${it}_ms" } + (1..8).map { "gas_${it}_ms" } + listOf(
            "lpg_level_raw", "analog_v", "analog_raw", "third_temperature_c", "third_temperature_raw",
            "gas_icon_red", "map_mode", "gas_pressure_mode", "analog_mode", "third_temperature_from_payload",
            "analog_confidence", "third_temperature_confidence", "gas_pressure_source",
            "timeouts", "rejected_frames", "firmware", "payload_0b_hex",
        )
    }
}
