package com.blazej222.lpgdashboard

import com.blazej222.lpgdashboard.protocol.Reading
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class InjectionSummaryTest {
    private fun readings(vararg values: Double?) = values.map {
        Reading(0, it, "ms", "test", source = "test")
    }

    @Test fun unusedChannelsDoNotLowerTheAverage() {
        assertEquals(2.22, readings(2.21, 2.23, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).averageInjectionTimeMs()!!, 1e-12)
    }

    @Test fun allEightChannelsContributeWithoutRoundingIndividualReadings() {
        val values = (1..8).map { it / 187.0 }
        assertEquals(4.5 / 187.0, readings(*values.toTypedArray()).averageInjectionTimeMs()!!, 1e-12)
    }

    @Test fun stoppedInjectionIsZeroAndMissingDataIsAbsent() {
        assertEquals(0.0, readings(0.0, 0.0).averageInjectionTimeMs()!!, 0.0)
        assertNull(readings().averageInjectionTimeMs())
        assertNull(readings(null, null).averageInjectionTimeMs())
    }
}
