package com.blazej222.lpgdashboard

import com.blazej222.lpgdashboard.protocol.Reading
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class InjectionSummaryTest {
    private fun readings(vararg values: Double?) = values.map {
        Reading(0, it, "ms", "test", source = "test")
    }

    @Test fun onlyConfiguredChannelsContributeEvenIfOthersHaveNonZeroValues() {
        assertEquals(2.22, readings(2.21, 2.23, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0).averageInjectionTimeMs(2)!!, 1e-12)
    }

    @Test fun zeroTimeOfAnInstalledInjectorContributesToTheAverage() {
        assertEquals(1.5, readings(2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0).averageInjectionTimeMs(4)!!, 1e-12)
    }

    @Test fun allEightChannelsContributeWithoutRoundingIndividualReadings() {
        val values = (1..8).map { it / 187.0 }
        assertEquals(4.5 / 187.0, readings(*values.toTypedArray()).averageInjectionTimeMs(8)!!, 1e-12)
    }

    @Test fun stoppedInjectionIsZeroAndMissingDataIsAbsent() {
        assertEquals(0.0, readings(0.0, 0.0).averageInjectionTimeMs(2)!!, 0.0)
        assertNull(readings().averageInjectionTimeMs(4))
        assertNull(readings(null, null).averageInjectionTimeMs(2))
        assertNull(readings(2.0, null).averageInjectionTimeMs(2))
        assertNull(readings(2.0).averageInjectionTimeMs(4))
    }

    @Test fun singleInstalledInjectorRetainsItsValue() {
        assertEquals(2.23, readings(2.23, 8.0).averageInjectionTimeMs(1)!!, 0.0)
    }
}
