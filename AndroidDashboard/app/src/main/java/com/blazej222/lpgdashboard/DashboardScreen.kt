package com.blazej222.lpgdashboard

import android.os.SystemClock
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.blazej222.lpgdashboard.protocol.*
import kotlinx.coroutines.delay
import java.io.File
import java.util.Locale

@Composable
fun DashboardApp(state: DashboardState, model: ConnectionViewModel, shareLog: (File) -> Unit) {
    var settingsOpen by rememberSaveable { mutableStateOf(false) }
    var now by remember { mutableLongStateOf(SystemClock.elapsedRealtime()) }
    LaunchedEffect(Unit) { while (true) { delay(500); now = SystemClock.elapsedRealtime() } }
    val age = state.lastFrameMs?.let { (now - it).coerceAtLeast(0) }
    val live = state.fresh && age != null && age < 2000
    Scaffold { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(horizontal = 16.dp, vertical = 8.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(Modifier.weight(1f)) {
                    Text("GAS ECU", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                    Text("KME Diego G3", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                OutlinedButton(onClick = { settingsOpen = !settingsOpen }) { Text(if (settingsOpen) "Dashboard" else "Ustawienia") }
                Spacer(Modifier.width(8.dp))
                Button(onClick = { if (state.busy) model.disconnect() else model.connect() },
                    enabled = state.phase != ConnectionPhase.STOPPING) {
                    Text(if (state.busy) "Rozłącz" else "Połącz")
                }
            }
            Text(state.message, color = when {
                state.phase == ConnectionPhase.ERROR -> MaterialTheme.colorScheme.error
                live -> MaterialTheme.colorScheme.primary
                else -> MaterialTheme.colorScheme.onSurfaceVariant
            }, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(vertical = 8.dp))
            if (settingsOpen) SettingsScreen(state, model, shareLog)
            else LiveDashboard(state, live, age)
        }
    }
}

@Composable
private fun LiveDashboard(state: DashboardState, live: Boolean, age: Long?) {
    val t = state.telemetry
    val width = LocalConfiguration.current.screenWidthDp
    val columns = if (width >= 900) 6 else if (width >= 600) 3 else 2
    LazyVerticalGrid(columns = GridCells.Fixed(columns),
        horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item(span = { GridItemSpan(maxLineSpan) }) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(if (live) "● NA ŻYWO" else if (t != null) "● DANE NIEAKTUALNE" else "● BRAK DANYCH",
                    color = if (live) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant)
                Text("${t?.stateLabel ?: "—"} · stan ${t?.stateCode ?: "—"}", style = MaterialTheme.typography.titleMedium)
            }
        }
        item { MetricCard("Obroty silnika", t?.rpm, live, 0) }
        item { MetricCard("Ciśnienie gazu", t?.gasPressure, live, 2) }
        item { MetricCard("Ciśnienie MAP", t?.mapPressure, live, 2) }
        item { MetricCard("Temperatura reduktora", t?.reducerTemperature, live, 1) }
        item { MetricCard("Temperatura gazu", t?.gasTemperature, live, 1) }
        item { MetricCard("Napięcie · bajt 4*", t?.voltageCandidate, live, 2) }
        item(span = { GridItemSpan(maxLineSpan) }) {
            if (state.settings.averageInjectionTimes) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    AverageInjectionCard("Czas wtrysku benzyny", t?.petrol?.averageInjectionTimeMs(), live, Modifier.weight(1f))
                    AverageInjectionCard("Czas wtrysku gazu", t?.gas?.averageInjectionTimeMs(), live, Modifier.weight(1f))
                }
            } else {
                Card {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Czasy wtrysku · ms", style = MaterialTheme.typography.titleMedium)
                        InjectionTable(t, live)
                    }
                }
            }
        }
        item(span = { GridItemSpan(maxLineSpan) }) {
            Text("Ramki: ${state.frames} · timeouty: ${state.timeouts} · błędne ramki: ${state.rejectedFrames}" +
                (age?.let { " · ostatnia: %.1f s temu".format(Locale.getDefault(), it / 1000.0) } ?: "") +
                "\n* Przelicznik napięcia potwierdzony; znaczenie jako napięcie zasilania pozostaje hipotezą.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun MetricCard(label: String, reading: Reading?, live: Boolean, decimals: Int) {
    Card(Modifier.fillMaxWidth().heightIn(min = 134.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(if (live) reading?.decoded.number(decimals) else "—", style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.SemiBold, color = if (live) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant)
            Text(reading?.unit ?: "", style = MaterialTheme.typography.bodyMedium)
            if (live && reading?.decoded == null) Text("RAW ${reading?.raw ?: "—"} · brak konfiguracji/źródła", style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun AverageInjectionCard(label: String, averageMs: Double?, live: Boolean, modifier: Modifier = Modifier) {
    Card(modifier.heightIn(min = 160.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(label, style = MaterialTheme.typography.titleMedium)
            Text(if (live) averageMs.number(2) else "—", style = MaterialTheme.typography.displaySmall,
                fontWeight = FontWeight.SemiBold,
                color = if (live) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant)
            Text("ms · średnia", style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun InjectionTable(t: Telemetry?, live: Boolean) {
    BoxWithConstraints {
        val columns = if (maxWidth < 600.dp) 4 else 8
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            for (start in 0 until 8 step columns) {
                Row(Modifier.fillMaxWidth()) {
                    Column(Modifier.width(70.dp)) {
                        Text("Kanał", style = MaterialTheme.typography.labelMedium)
                        Text("Benzyna", style = MaterialTheme.typography.bodyMedium)
                        Text("Gaz", style = MaterialTheme.typography.bodyMedium)
                    }
                    for (index in start until start + columns) {
                        Column(Modifier.weight(1f)) {
                            Text("${index + 1}", style = MaterialTheme.typography.labelMedium)
                            Text(if (live) t?.petrol?.get(index)?.decoded.number(2) else "—")
                            Text(if (live) t?.gas?.get(index)?.decoded.number(2) else "—", color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SettingsScreen(state: DashboardState, model: ConnectionViewModel, shareLog: (File) -> Unit) {
    val settings = state.settings
    val enabled = !state.busy
    LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item {
            Text("Wyświetlanie", style = MaterialTheme.typography.titleLarge)
            SettingSwitch("Średnie czasy wtrysku", "Dwa duże kafelki zamiast tabeli kanałów. Średnia osobno dla benzyny i gazu, z pominięciem zerowych czasów; same zera dają 0,00 ms.",
                settings.averageInjectionTimes, true, model::setAverageInjectionTimes)
        }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Urządzenie USB", style = MaterialTheme.typography.titleLarge)
                OutlinedButton(onClick = model::refreshDevices) { Text("Odśwież") }
            }
            Text("9600 bodów · 8N1 · bez kontroli przepływu. Wybierz port kabla LPG.")
            if (state.devices.isEmpty()) Text("Nie wykryto urządzeń USB. Sprawdź kabel i port USB host radia.")
        }
        items(state.devices, key = { it.key }) { device ->
            Card {
                Row(Modifier.fillMaxWidth().padding(8.dp)) {
                    RadioButton(selected = state.selectedKey == device.key,
                        onClick = { model.selectDevice(device.key) }, enabled = enabled && device.portIndex != null)
                    Column(Modifier.padding(top = 8.dp)) {
                        Text(device.title)
                        Text(device.details, style = MaterialTheme.typography.bodySmall)
                        if (device.portIndex == null) Text("To urządzenie nie udostępnia obsługiwanego portu szeregowego", color = MaterialTheme.colorScheme.error)
                    }
                }
            }
        }
        item {
            Text("Profil czujników", style = MaterialTheme.typography.titleLarge)
            Text("Domyślnie wybrano wariant instalacji z pomiarów w dokumentacji. Typów czujników nie wykrywamy automatycznie. Zmiana ustawień wymaga rozłączenia.")
        }
        item {
            ChoiceRow("MAP", listOf("Z pomiarów" to MapMode.OBSERVED, "Alternatywny" to MapMode.ALTERNATE, "Nieznany" to MapMode.UNKNOWN),
                settings.decoder.mapMode, enabled) { model.updateSettings(settings.copy(decoder = settings.decoder.copy(mapMode = it))) }
        }
        item {
            ChoiceRow("Ciśnienie gazu", listOf("FA → 0D/0F" to GasPressureMode.REGISTER_FALLBACK, "Standardowy" to GasPressureMode.STANDARD, "Nieznany" to GasPressureMode.UNKNOWN),
                settings.decoder.gasPressureMode, enabled) { model.updateSettings(settings.copy(decoder = settings.decoder.copy(gasPressureMode = it))) }
        }
        item {
            ChoiceRow("Odstęp odczytu 0B", listOf("100 ms" to 100L, "200 ms" to 200L, "500 ms" to 500L, "1 s" to 1000L),
                settings.pollIntervalMs, enabled) { model.updateSettings(settings.copy(pollIntervalMs = it)) }
            Text("Odstęp minimalny; rzeczywista częstotliwość zależy od czasu odpowiedzi ECU.", style = MaterialTheme.typography.bodySmall)
        }
        item {
            SettingSwitch("DTR i RTS włączone", "Tak jak w przechwyconej sesji producenta.", settings.assertControlLines, enabled) {
                model.updateSettings(settings.copy(assertControlLines = it))
            }
            SettingSwitch("Pełny cykl odczytu producenta", "Dodatkowe zapytania 40/29, 64, 40/28, 53, 40/0D, 40/0F, 3F. Dashboard korzysta z 0B.", settings.originalPolling, enabled) {
                model.updateSettings(settings.copy(originalPolling = it))
            }
            SettingSwitch("Wejście analogowe: przeliczaj × 0,02 V", "Włącz dla potwierdzonego trybu 1–6; fizyczne znaczenie wejścia nieustalone.", settings.decoder.analogMode != null, enabled) {
                model.updateSettings(settings.copy(decoder = settings.decoder.copy(analogMode = if (it) 1 else null)))
            }
            SettingSwitch("Temperatura 3 z bajtu 47", "Włącz po potwierdzeniu wariantu czujnika; domyślnie pokazywany jest RAW.", settings.decoder.thirdTemperatureFromPayload, enabled) {
                model.updateSettings(settings.copy(decoder = settings.decoder.copy(thirdTemperatureFromPayload = it)))
            }
        }
        item {
            Text("Diagnostyka", style = MaterialTheme.typography.titleLarge)
            Text("Inicjalizacja: PORTMAN.LOG · 77 żądań. Firmware: ${state.firmware.ifEmpty { "—" }}")
            Text("X = kod producenta, M = pomiary GUI, H = hipoteza, N = nieznane. Warianty warunkowe wymagają konfiguracji.", style = MaterialTheme.typography.bodySmall)
            state.telemetry?.let { t ->
                for ((label, reading) in listOf("Poziom LPG" to t.levelRaw, "Analog" to t.analog,
                    "Temperatura 3" to t.thirdTemperature, "Napięcie (hipoteza)" to t.voltageCandidate,
                    "MAP" to t.mapPressure, "Ciśnienie gazu" to t.gasPressure)) {
                    Text("$label: RAW ${reading.raw} · ${reading.decoded.number(3)} ${reading.unit} · [${reading.confidence}] · ${reading.configMode ?: ""}",
                        style = MaterialTheme.typography.bodySmall)
                    Text(reading.source, style = MaterialTheme.typography.bodySmall)
                }
                Text("Ikona G: ${if (t.gasIconRed) "czerwona" else "szara"} [M] · stan ${t.stateCode}")
                Text("RAW 42–46/48–49: " + listOf(42,43,44,45,46,48,49).joinToString(" · ") { "$it=%02X".format(t.rawPayload[it]) },
                    style = MaterialTheme.typography.bodySmall)
                Text(t.rawPayload.map { it.toByte() }.toByteArray().hex(), fontFamily = FontFamily.Monospace,
                    style = MaterialTheme.typography.bodySmall)
            }
            state.logFile?.let { log ->
                OutlinedButton(onClick = { shareLog(log) }) { Text("Udostępnij log TX/RX") }
                Text("Log sesji: ${log.name} · limit 4 MiB · zachowane ostatnie 5 sesji.", style = MaterialTheme.typography.bodySmall)
            }
            Text(state.trace.joinToString("\n"), fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun <T> ChoiceRow(label: String, options: List<Pair<String, T>>, selected: T, enabled: Boolean, choose: (T) -> Unit) {
    Column {
        Text(label, style = MaterialTheme.typography.titleMedium)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            options.forEach { (name, value) ->
                FilterChip(selected = selected == value, onClick = { choose(value) }, label = { Text(name) }, enabled = enabled)
            }
        }
    }
}

@Composable
private fun SettingSwitch(label: String, description: String, checked: Boolean, enabled: Boolean, change: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Column(Modifier.weight(1f).padding(end = 12.dp)) {
            Text(label, style = MaterialTheme.typography.titleMedium)
            Text(description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = change, enabled = enabled)
    }
}

private fun Double?.number(decimals: Int): String = this?.let { String.format(Locale.getDefault(), "%.${decimals}f", it) } ?: "—"
