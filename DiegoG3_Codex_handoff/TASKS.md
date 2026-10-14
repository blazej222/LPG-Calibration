# TASKS.md — backlog techniczny dla Codexa

## Skończone / punkt startowy

- [x] Komunikaty 4-bajtowe `C0` + suma i statyczna odpowiedź `D0` + payload + suma w emulatorze.
- [x] 50-bajtowa odpowiedź `0B` bajt-po-bajcie; parser Big Endian; mapa 8+8 wtryskiwaczy.
- [x] LUT 256 temperatur z EXE (`temperature_lut.csv`), zweryfikowana także 55/55 historycznymi pomiarami użytkownika.
- [x] RPM `10*round(2_244_000/N)` potwierdzone 5/5 pomiarów.
- [x] MAP `payload[41]`, aktywny wzór potwierdzony 15/15.
- [x] Maski benzyna1 `1FFF`, benzyna2 `7FFF`, stany LPG 0..7 i flaga ikony „G”.
- [x] Ciśnienie gazu standardowe i specjalne przy RAW>=FA, rejestry 0D/0F.
- [x] Katalog historycznych odpowiedzi emulatora i testy ich ramkowania.

## Wykonaj w tej kolejności

1. **P1 — Zbadanie maski z bajtu 45 w GUI.** Xref do `[ebp+0x1A]`, warunek możliwości z `0x0056409C`, potwierdzenie wybranej gałęzi. Użytkownik nie zaobserwował `C0 3A 00 FA` podczas `trace all`. Rozstrzygnij, czy zmiana RAW45 jest w ogóle używana w tej konfiguracji.
2. **P2 — Napięcie/poziom LPG/analog**. Zidentyfikuj status kalibracji bajtu 5 i rodzaj wejścia bajtu 6 (`0x0057BEB0`). Opracuj jednoznaczne testy GUI dla kilku RAW, z konfiguracją czujnika.
3. **P3 — Dodatkowe pola** `42..44`, `46`, `48..49`: xrefy przed skanami, z dokumentacją `bit->źródło->ekran`. Bez sztucznych nazw w GUI.
4. **P4 — Tryb temperatury3 i ciśnienia MAP**: zweryfikuj bramki i fallback rejestru 0x1B, wskaż etykietę czujnika, wykonaj testy w obu trybach jeśli dostępne.
5. **P5 — Realizacja klienta tylko do odczytu**: pyserial, incremental frame accumulator, konfiguracja COM, polling **wyłącznie komend potwierdzonych jako read-only**, logi, GUI po późniejszej decyzji użytkownika.
6. **P6 — Mapa pozostałych komend**: pasywny traffic capture z oprogramowania, klasy `read_only/mutation/handshake/unknown`; semantyka rodziny 07,12,40,51,53,60,61 itd.

## Kryterium ukończenia każdego punktu

- Zapisano dowód `[X] VA + instrukcje` lub `[M] RAW→GUI + parametry konfiguracji` do `FINDINGS.md`.
- Przygotowano funkcję czystą i test reprodukujący ustalenie.
- Zaktualizowano tabelę w `PROTOCOL.md` i status `[H]/[N]`.
- Żaden test z `python -m unittest discover -s tests -v` nie uległ regresji.
- Nie wykonano żadnego działania wpływającego na pracującą instalację LPG.
