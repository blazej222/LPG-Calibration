# DiegoG3 ECU — gotowy handoff reverse engineeringu do Codex / VS Code

**Wersja badań: 2026-10-14.** W tym folderze jest komplet dotychczasowych ustaleń oraz działający referencyjny dekoder offline. Agent nie potrzebuje oryginalnej rozmowy w ChatGPT.

## Pliki na początek

- **[`PROTOCOL.md`](PROTOCOL.md)** — szczegółowa specyfikacja protokołu, katalog 83 zapytań, ramka `0B`, wzory, maski, testy i zastrzeżenia.
- **[`REVERSE_ENGINEERING.md`](REVERSE_ENGINEERING.md)** — materiał dowodowy z EXE, VA/disassembly, historia błędnych hipotez, wnioski eksperymentalne, lista zadań badawczych.
- **[`AGENTS.md`](AGENTS.md)** — obowiązujące instrukcje dla Codexa (nazwa **AGENTS.md** to standard). Na Windows jest to ten sam plik, o który proszono jako „agents.MD”.
- **[`TASKS.md`](TASKS.md)** — priorytety i kryteria odbioru.
- **[`reference/known_commands.json`](reference/known_commands.json)** — pełny machine-readable katalog żądań i przykładowych odpowiedzi, włącznie z wielobajtowymi payloadami.
- **[`reference/temperature_lut.csv`](reference/temperature_lut.csv)** — pełna oryginalna tablica 256 temperatur.
- **`reference/DiegoG3_measurements*.csv`** — trzy źródłowe CSV eksperymentalne (5 RPM, 10 stanów/ikony, 15 MAP).
- **`reference/tred_measurements_from_conversation.csv`** — 55 unikalnych historycznych pomiarów Tred przepisanych z rozmowy; zgodność 55/55 z odzyskaną LUT.
- **[`src/diego_protocol.py`](src/diego_protocol.py)** — Python 3.10+, referencyjny dekoder i inkrementalny składacz odpowiedzi.
- **[`tools/emulator_cli.py`](tools/emulator_cli.py)** — samodzielny emulator laboratoryjny (nie wymaga starego `emulator.py`); tryb `--dry-run` działa offline, z COM wymaga opcjonalnego `pyserial`.
- **[`tests/test_protocol.py`](tests/test_protocol.py)** — testy regresji, bez zewnętrznych zależności.
- **`tools/extract_temperature_lut.py`** — ekstrakcja danych `double` z dokładnie tej wersji EXE, z kontrolą SHA.
- **`tools/make_catalog.py`, `tools/generate_protocol_md.py`** — odtworzenie katalogu i tabeli w specyfikacji.

## VS Code / Codex: jak rozpocząć

1. Rozpakuj ZIP do nowego katalogu, np. `DiegoG3-RE/` i otwórz **ten folder** w VS Code (`File > Open Folder`). `AGENTS.md` powinien znajdować się w **root workspace**.
2. Otwórz terminal PowerShell w katalogu projektu, sprawdź Python `3.10+`.
3. Wykonaj:

   ```powershell
   python -m unittest discover -s tests -v
   python src/diego_protocol.py
   python tools/emulator_cli.py --dry-run
   ```

4. Otwórz panel Codex w VS Code; jako pierwsze zadanie użyj opisu z [`HANDOFF_PROMPT.md`](HANDOFF_PROMPT.md), jeśli agent nie odczyta automatycznie plików.
5. Jeżeli masz oryginalny EXE, umieść go lokalnie w `vendor/DiegoG3-3.0.8.2.exe` (ten katalog jest ignorowany przez Git) i sprawdź:

   ```powershell
   python tools/extract_temperature_lut.py vendor/DiegoG3-3.0.8.2.exe --out reference/temperature_lut_check.csv
   ```

   Plik `temperature_lut.csv` już jest w paczce; oryginalny EXE **nie jest wymagany do dekodowania ani testów**.

## Opcjonalna komunikacja z prawdziwym ECU

Ten handoff zawiera **emulator odpowiadający aplikacji PC**, lecz **nie zawiera klienta automatycznie wysyłającego komendy** do rzeczywistego sterownika — to świadoma decyzja. Część katalogu to statyczne odpowiedzi dawnego emulatora, bez potwierdzenia bezpieczeństwa komend. Przed implementacją portu i schedulera należy przechwycić pasywny trace i oddzielić read-only od funkcji zapisujących/konfiguracyjnych. Do przyszłej warstwy I/O prawdopodobnie przyda się `pyserial` (`python -m pip install pyserial`), ale testy offline go nie wymagają.

## Co pochodzi z EXE, a co z eksperymentów

- **EXE [X]:** LUT, parser `0B`, maski flag, `/187`, wzór RPM, skalowania ciśnienia/MAP, wybrane warunkowe ścieżki GUI.
- **Eksperyment [M]:** RPM 5/5; mapa MAP 15/15; stany LPG 8/8; ikonka G 2/2; seria temperatur potwierdza LUT.
- **Emulator [E]:** 83 requesty + statyczne payloady, port COM18, 9600, checksum TX/RX.
- **Hipotezy [H]/nieznane [N]:** bezpieczność i znaczenie większości komend, poziom LPG, pełna semantyka 45 i 42–49, fizyczne nazwy 4/6/47 i inne wersje konfiguracji.

**Ważne:** wszystkie długie odpowiedzi w JSON pochodzą z ręcznej transkrypcji wklejonego kodu emulatora, nie z lokalnego pliku źródłowego. Jeśli posiadasz w swoim projekcie oryginalny `emulator.py`, zrób z nim porównanie bajt po bajcie przed wykorzystaniem całego katalogu w aplikacji produkcyjnej.
