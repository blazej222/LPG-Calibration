package com.blazej222.lpgdashboard

import com.blazej222.lpgdashboard.protocol.*
import org.junit.Assert.*
import org.junit.Test
import java.util.Locale

class DiegoProtocolTest {
    private fun resource(name: String) = javaClass.classLoader!!.getResourceAsStream(name)!!.bufferedReader().use { it.readText() }
    private fun hex(text: String) = text.trim().split(Regex("\\s+")).map { it.toInt(16).toByte() }.toByteArray()
    private fun baseline() = hex(resource("baseline.hex"))
    private fun decode(payload: ByteArray = baseline(), config: DecoderConfig = DecoderConfig(), r0d: Int? = null, r0f: Int? = null) =
        DiegoProtocol.decode(payload, TemperatureTable.values, config, r0d, r0f)
    // Source fixtures contain no quoted comma-bearing cells; empty columns must be preserved.
    private fun csv(name: String): List<Map<String, String>> {
        val rows = resource(name).trim().lines()
        val header = rows.first().removePrefix("\uFEFF").split(',')
        return rows.drop(1).map { header.zip(it.split(',')).toMap() }
    }

    @Test fun baselineMatchesVerifiedDecoder() {
        val t = decode()
        assertEquals(860.0, t.rpm.decoded!!, 0.0)
        assertEquals(33.53631112, t.reducerTemperature.decoded!!, 0.00001)
        assertEquals(18.23821472, t.gasTemperature.decoded!!, 0.00001)
        assertEquals(1.1725101626, t.gasPressure.decoded!!, 0.000001)
        assertEquals(0.40605, t.mapPressure.decoded!!, 0.00001)
        assertEquals(5, t.stateCode)
        assertTrue(t.gasIconRed)
        assertEquals(listOf(2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0), t.petrol.map { it.decoded })
        assertEquals(t.petrol.map { it.decoded }, t.gas.map { it.decoded })
        assertNull(t.levelRaw.decoded)
        assertNull(t.analog.decoded)
        assertNull(t.thirdTemperature.decoded)
        assertEquals(50, t.rawPayload.size)
    }

    @Test fun allFiveRpmMeasurementsMatchGui() {
        val rows = csv("DiegoG3_measurements.csv")
        assertEquals(5, rows.size)
        rows.forEach { assertEquals(it.getValue("displayed_in_DiegoG3").toDouble(), decode(hex(it.getValue("payload_hex"))).rpm.decoded!!, 0.0) }
    }

    @Test fun allFifteenMapMeasurementsMatchGui() {
        val rows = csv("DiegoG3_measurements(2).csv")
        assertEquals(15, rows.size)
        rows.forEach {
            val result = decode(hex(it.getValue("payload_hex"))).mapPressure.decoded!!
            assertEquals(it.getValue("displayed_in_DiegoG3"), String.format(Locale.ROOT, "%.2f", result))
        }
    }

    @Test fun allTenStateAndIconMeasurementsPreserveCodes() {
        val rows = csv("DiegoG3_measurements(1).csv")
        assertEquals(10, rows.size)
        rows.forEach { row ->
            val t = decode(hex(row.getValue("payload_hex")))
            if (row.getValue("field") == "raw8:9") {
                assertEquals(row.getValue("requested_value").toInt(16) shr 5, t.stateCode)
                assertEquals(2.0, t.petrol[0].decoded!!, 0.0)
            } else {
                assertEquals(row.getValue("requested_value").toInt(16) and 0x8000 != 0, t.gasIconRed)
                assertEquals(2.0, t.petrol[1].decoded!!, 0.0)
            }
        }
    }

    @Test fun exactLutAndFiftyFiveTemperatureMeasurements() {
        assertEquals(256, TemperatureTable.values.size)
        csv("temperature_lut.csv").forEach { row ->
            assertEquals(row.getValue("temperature_c_exact").toDouble(), TemperatureTable.values[row.getValue("raw_dec").toInt()], 0.0)
        }
        val measurements = csv("tred_measurements_from_conversation.csv")
        assertEquals(55, measurements.size)
        measurements.forEach { row ->
            assertEquals(row.getValue("gui_tred_c").toDouble(), TemperatureTable.values[row.getValue("raw_dec").toInt()], 0.051)
        }
    }

    @Test fun masksAreSpecificToPetrolChannelsOneAndTwo() {
        val payload = baseline()
        (DiegoProtocol.petrolOffsets + DiegoProtocol.gasOffsets).forEach { payload[it] = 0xFF.toByte(); payload[it + 1] = 0xFF.toByte() }
        val t = decode(payload)
        assertEquals(0x1FFF / 187.0, t.petrol[0].decoded!!, 0.0)
        assertEquals(0x7FFF / 187.0, t.petrol[1].decoded!!, 0.0)
        (t.petrol.drop(2) + t.gas).forEach { assertEquals(65535 / 187.0, it.decoded!!, 0.0) }
        assertEquals(7, t.stateCode)
    }

    @Test fun pressureBoundaryAndUnknownModesDoNotInventValues() {
        for (raw in listOf(0, 249, 250, 255)) {
            val p = baseline().apply { this[1] = raw.toByte() }
            if (raw >= 250) {
                assertNull(decode(p).gasPressure.decoded)
                assertEquals(1.9072107438, decode(p, r0d = 142, r0f = 26).gasPressure.decoded!!, 0.000001)
                assertEquals(0.0, decode(p, r0d = 1, r0f = 2).gasPressure.decoded!!, 0.0)
            } else assertNotNull(decode(p).gasPressure.decoded)
            assertNotNull(decode(p, DecoderConfig(gasPressureMode = GasPressureMode.STANDARD)).gasPressure.decoded)
        }
        val t = decode(config = DecoderConfig(MapMode.UNKNOWN, GasPressureMode.UNKNOWN))
        assertNull(t.mapPressure.decoded)
        assertNull(t.gasPressure.decoded)
        assertNotNull(decode(config = DecoderConfig(analogMode = 6, thirdTemperatureFromPayload = true)).analog.decoded)
        assertNotNull(decode(config = DecoderConfig(thirdTemperatureFromPayload = true)).thirdTemperature.decoded)
    }

    @Test fun zeroRpmAndTiesUseEvenRounding() {
        assertEquals(0.0, decode(baseline().apply { this[7] = 0; this[8] = 0 }).rpm.decoded!!, 0.0)
        for ((word, expected) in listOf(64 to 350620.0, 192 to 116880.0)) {
            val payload = baseline().apply { this[7] = (word shr 8).toByte(); this[8] = word.toByte() }
            assertEquals(expected, decode(payload).rpm.decoded!!, 0.0)
        }
        // Finite words, including 65535, stay valid.
        for (word in listOf(1, 128, 65535)) {
            val payload = baseline().apply { this[7] = (word shr 8).toByte(); this[8] = word.toByte() }
            assertTrue(decode(payload).rpm.decoded!! >= 0)
        }
    }

    @Test fun arbitraryChunkBoundariesAndMarkerBytesInsidePayload() {
        val payload = baseline().apply { this[42] = 0xD0.toByte(); this[43] = 0xC0.toByte() }
        val wire = DiegoProtocol.frame(payload)
        for (split in 0..wire.size) {
            val parser = ReplyAccumulator(50, 0x0B)
            val frames = parser.feed(wire.copyOfRange(0, split)) + parser.feed(wire.copyOfRange(split, wire.size))
            assertEquals(1, frames.size)
            assertArrayEquals(payload, frames[0])
        }
        val parser = ReplyAccumulator(50, 0x0B)
        val frames = wire.flatMap { parser.feed(byteArrayOf(it)) }
        assertEquals(1, frames.size)
        assertArrayEquals(payload, frames[0])
    }

    @Test fun badChecksumNoiseAndConcatenatedFramesResynchronize() {
        val payload = baseline()
        val valid = DiegoProtocol.frame(payload)
        val bad = valid.copyOf().apply { this[lastIndex] = (this[lastIndex].toInt() xor 1).toByte() }
        val parser = ReplyAccumulator(50, 0x0B)
        val frames = parser.feed(byteArrayOf(0x55, 0xD0.toByte(), 0x01) + bad + valid + valid)
        assertEquals(2, frames.size)
        assertTrue(parser.rejectedCandidates > 0)
        frames.forEach { assertArrayEquals(payload, it) }
    }

    @Test fun rejectsWrongRegisterReplyAndInvalidInput() {
        val parser = ReplyAccumulator(3, 0x07, 0x21)
        assertTrue(parser.feed(DiegoProtocol.frame(byteArrayOf(0x07, 0x22, 0x00))).isEmpty())
        assertEquals(1, parser.feed(DiegoProtocol.frame(byteArrayOf(0x07, 0x21, 0x03))).size)
        assertThrows(IllegalArgumentException::class.java) { DiegoProtocol.request(-1, 0) }
        assertThrows(IllegalArgumentException::class.java) { DiegoProtocol.request(0, 256) }
        assertThrows(IllegalArgumentException::class.java) { decode(ByteArray(49)) }
        assertThrows(IllegalArgumentException::class.java) { decode(ByteArray(50)) }
        assertThrows(IllegalArgumentException::class.java) { DiegoProtocol.decode(baseline(), DoubleArray(255), DecoderConfig()) }
        assertThrows(IllegalArgumentException::class.java) { decode(r0d = -1) }
    }

    @Test fun requestAndBaselineReplyChecksumMatchReference() {
        assertEquals("C0 0B 0B D6", DiegoProtocol.request(0x0B, 0x0B).hex())
        assertEquals(0x77, DiegoProtocol.checksum(baseline()))
    }

    @Test fun startupSequenceMatchesAllSeventySevenCapturedRequests() {
        val rows = resource("startup.tsv").trim().lines()
        assertEquals(77, StartupSequence.commands.size)
        rows.zip(StartupSequence.commands).forEach { (row, command) ->
            val cells = row.split('\t')
            assertArrayEquals(hex(cells[1]), command.wire)
            assertEquals(cells[2].toInt(), command.payloadSize)
        }
        val noReply = StartupSequence.commands.filter { it.payloadSize == 0 }
        assertEquals(4, noReply.size)
        noReply.forEach { assertEquals("C2 01 01 6B 87", it.wire.hex()); assertEquals(450, it.pauseMs) }
    }

    @Test fun completeInitializationWorksAgainstFragmentedOriginalEmulatorReplies() {
        val fixtures = resource("startup_replies.tsv").trim().lines().associate {
            val cells = it.split('\t'); cells[0] to hex(cells[1])
        }
        var pending = byteArrayOf()
        var reads = 0
        var time = 0L
        val transport = object : SerialTransport {
            override fun write(bytes: ByteArray, timeoutMs: Int) {
                assertTrue(pending.isEmpty())
                pending = fixtures[bytes.hex()]?.let { DiegoProtocol.frame(it) } ?: byteArrayOf()
            }
            override fun read(buffer: ByteArray, timeoutMs: Int): Int {
                time += 1_000_000
                val length = minOf(pending.size, listOf(1, 3, 2, 7)[reads++ % 4])
                pending.copyInto(buffer, 0, 0, length)
                pending = pending.copyOfRange(length, pending.size)
                return length
            }
        }
        val session = DiegoSession(transport, nanoTime = { time })
        StartupSequence.commands.forEach { command ->
            val reply = session.exchange(command)
            if (command.payloadSize == 0) assertNull(reply)
            else assertArrayEquals(fixtures[command.wire.hex()], reply)
        }
        assertEquals(0, session.rejectedFrames)
    }

    @Test fun timeoutDoesNotDecodeTruncatedResponseAndCanDrain() {
        var time = 0L
        var pending = byteArrayOf(0xD0.toByte(), 0x0B, 0x79)
        val transport = object : SerialTransport {
            override fun write(bytes: ByteArray, timeoutMs: Int) {}
            override fun read(buffer: ByteArray, timeoutMs: Int): Int {
                time += timeoutMs * 1_000_000L
                val count = pending.size
                pending.copyInto(buffer)
                pending = byteArrayOf()
                return count
            }
        }
        val session = DiegoSession(transport, nanoTime = { time })
        assertThrows(ReplyTimeout::class.java) { session.exchange(StartupCommand(DiegoProtocol.request(0x0B, 0x0B), 50), 300) }
        session.drain()
    }
}
