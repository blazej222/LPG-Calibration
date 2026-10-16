package com.blazej222.lpgdashboard

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import org.junit.Rule
import org.junit.Test

/** Emulator checks for the real activity and USB-absent flow; opens no ECU port. */
class DashboardSmokeTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    @Test fun dashboardOpensAndSettingsListMissingUsbDevice() {
        compose.onNodeWithText("GAS ECU").assertIsDisplayed()
        compose.onNodeWithText("● BRAK DANYCH").assertIsDisplayed()
        compose.onNodeWithText("Ustawienia").performClick()
        compose.onNodeWithText("Urządzenie USB").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("Nie wykryto urządzeń USB. Sprawdź kabel i port USB host radia.").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("Dashboard").performClick()
        compose.onNodeWithText("● BRAK DANYCH").assertIsDisplayed()
    }

    @Test fun connectWithoutSelectedDeviceShowsUsefulError() {
        compose.onNodeWithText("Połącz").performClick()
        compose.onNodeWithText("Wybierz obsługiwany port USB w ustawieniach").assertIsDisplayed()
        compose.onNodeWithText("● BRAK DANYCH").assertIsDisplayed()
    }
}
