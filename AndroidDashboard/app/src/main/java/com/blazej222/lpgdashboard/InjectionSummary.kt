package com.blazej222.lpgdashboard

import com.blazej222.lpgdashboard.protocol.Reading

/** Average installed channels, including stopped injectors; never infer count from times. */
internal fun List<Reading>.averageInjectionTimeMs(injectorCount: Int): Double? {
    require(injectorCount in 1..8)
    val values = take(injectorCount).mapNotNull { it.decoded }
    return if (values.size == injectorCount) values.average() else null
}
