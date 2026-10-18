package com.blazej222.lpgdashboard

import com.blazej222.lpgdashboard.protocol.*
import org.junit.Assert.*
import org.junit.Test
import java.io.IOException
import java.io.StringWriter
import java.util.Locale

class TelemetryCsvTest {
    private fun sample(count: Int = 4, elapsed: Long = 2000, config: DecoderConfig = DecoderConfig()): CsvSample {
        val payload = javaClass.classLoader!!.getResourceAsStream("baseline.hex")!!.bufferedReader().use { it.readText() }
            .trim().split(Regex("\\s+")).map { it.toInt(16).toByte() }.toByteArray()
        return CsvSample(DiegoProtocol.decode(payload, TemperatureTable.values, config), 1700000000123,
            elapsed, 7, count, config, 2, 3, "Diego G3")
    }

    // Independent delimiter/quote reader, including escaped quotes and embedded newlines.
    private fun parseCsv(text: String): List<List<String>> {
        val rows = mutableListOf<List<String>>()
        var row = mutableListOf<String>()
        val field = StringBuilder()
        var quoted = false
        val source = text.removePrefix("\uFEFF")
        var index = 0
        while (index < source.length) {
            val character = source[index++]
            when {
                character == '"' && quoted && index < source.length && source[index] == '"' -> { field.append('"'); index++ }
                character == '"' -> quoted = !quoted
                character == ';' && !quoted -> { row += field.toString(); field.setLength(0) }
                character == '\r' && !quoted -> Unit
                character == '\n' && !quoted -> {
                    row += field.toString(); field.setLength(0); rows += row; row = mutableListOf()
                }
                else -> field.append(character)
            }
        }
        assertFalse("Unterminated CSV quote", quoted)
        return rows
    }

    @Test fun completeSamplePreservesReadingsPrecisionUnitsRawAndMonotonicTime() {
        val output = StringWriter()
        val sample = sample()
        val csv = TelemetryCsvWriter(output, 2000)
        csv.append(sample)
        csv.append(sample.copy(elapsedRealtimeMs = 2250, frameNumber = 8))
        csv.close()
        val rows = parseCsv(output.toString())
        assertEquals(3, rows.size)
        assertTrue(output.toString().startsWith("\uFEFF"))
        assertEquals(rows[0].size, rows[1].size)
        assertEquals(rows[0].size, rows[2].size)
        val data = rows[0].zip(rows[1]).toMap()
        assertEquals("2023-11-14T22:13:20.123Z", data["timestamp_utc"])
        assertEquals("0", data["elapsed_ms"])
        assertEquals("250", rows[0].zip(rows[2]).toMap()["elapsed_ms"])
        assertEquals(33.53631112, data.getValue("reducer_temperature_c").toDouble(), 1e-8)
        assertEquals(sample.telemetry.batteryVoltage.decoded!!, data.getValue("battery_voltage_v").toDouble(), 0.0)
        assertEquals("860.0", data["rpm"])
        assertEquals("5", data["state_code"])
        assertEquals("2", data["timeouts"])
        assertEquals("3", data["rejected_frames"])
        assertEquals(sample.telemetry.rawPayload.map { it.toByte() }.toByteArray().hex(), data["payload_0b_hex"])
        assertEquals(2L, csv.rows)
    }

    @Test fun configuredChannelsAndAveragesRemainConsistentAcrossCountChanges() {
        val base = sample()
        val t = base.telemetry.copy(
            petrol = base.telemetry.petrol.mapIndexed { index, reading -> reading.copy(decoded = if (index == 0) 0.0 else 2.0) },
            gas = base.telemetry.gas.map { it.copy(decoded = 4.0) },
        )
        val output = StringWriter()
        TelemetryCsvWriter(output, 2000).use {
            it.append(base.copy(telemetry = t))
            it.append(base.copy(telemetry = t, injectorCount = 6))
        }
        val rows = parseCsv(output.toString())
        val first = rows[0].zip(rows[1]).toMap()
        val second = rows[0].zip(rows[2]).toMap()
        assertEquals("1.5", first["petrol_average_ms"])
        assertEquals("4.0", first["gas_average_ms"])
        assertEquals("0.0", first["petrol_1_ms"])
        assertEquals("", first["petrol_5_ms"])
        assertEquals("2.0", second["petrol_5_ms"])
        assertEquals("4.0", second["gas_6_ms"])
        assertEquals("", second["gas_7_ms"])
        assertEquals(rows[0].size, rows[2].size)
    }

    @Test fun unknownReadingsAreBlankAndNumbersAreIndependentOfPhoneLocale() {
        val previous = Locale.getDefault()
        try {
            Locale.setDefault(Locale.forLanguageTag("pl-PL"))
            val output = StringWriter()
            TelemetryCsvWriter(output, 2000).use {
                it.append(sample(config = DecoderConfig(mapMode = MapMode.UNKNOWN, gasPressureMode = GasPressureMode.UNKNOWN)))
            }
            val rows = parseCsv(output.toString())
            val data = rows[0].zip(rows[1]).toMap()
            assertEquals("", data["map_pressure_bar"])
            assertEquals("", data["gas_pressure_bar"])
            assertEquals("", data["analog_v"])
            assertEquals("23", data["analog_raw"])
            assertTrue(data.getValue("battery_voltage_v").contains('.'))
            assertEquals("X warunkowe; nazwa H", data["analog_confidence"])
        } finally { Locale.setDefault(previous) }
    }

    @Test fun textWithDelimiterQuotesAndNewlinesSurvivesCsvRoundTrip() {
        val firmware = "Diego; \"G3\"\r\nNowa linia"
        val output = StringWriter()
        TelemetryCsvWriter(output, 2000).use { it.append(sample().copy(firmware = firmware)) }
        val rows = parseCsv(output.toString())
        assertEquals(firmware, rows[0].zip(rows[1]).toMap()["firmware"])
    }

    @Test fun writeFailureIsReportedAndDoesNotCountAnUncommittedRow() {
        var flushes = 0
        var closed = false
        val output = object : StringWriter() {
            override fun flush() { if (++flushes > 1) throw IOException("disk full") }
            override fun close() { closed = true }
        }
        val csv = TelemetryCsvWriter(output, 2000)
        assertThrows(IOException::class.java) { csv.append(sample()) }
        assertEquals(0L, csv.rows)
        csv.close()
        assertTrue(closed)
    }
}
