package com.blazej222.lpgdashboard

import com.blazej222.lpgdashboard.protocol.Reading

/** Presentation only: zero channels are omitted, without inferring the cylinder count. */
internal fun List<Reading>.averageInjectionTimeMs(): Double? {
    val values = mapNotNull { it.decoded }
    if (values.isEmpty()) return null
    val nonZero = values.filter { it > 0.0 }
    return if (nonZero.isEmpty()) 0.0 else nonZero.average()
}
