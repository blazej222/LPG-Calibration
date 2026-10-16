# FINDINGS.md — nowe ustalenia po przekazaniu projektu

Zasady wpisu: `data, źródło ([X]/[M]/[E]/[H]), weryfikacja, VA/offset, wersja EXE, wniosek, test, następny krok`.

## 2026-10-14 — Stan początkowy

Wszystkie badania ukończone przed przekazaniem znajdują się w `PROTOCOL.md` i `REVERSE_ENGINEERING.md`. Od tej chwili dodawaj tu tylko nowe dowody. Nie powtarzaj bez analizy starych hipotez.

## 2026-10-14 — Klient Android, porównanie z lokalnymi logami i oryginalnym emulatorem

- **[M: pasywny trace]** `../GasECU Reverse Engineering/data/PORTMAN.LOG`, wpisy ID 10–15: 9600, 8N1, SET_RTS, SET_DTR. Potwierdza ustawienia aplikacji PC w zarejestrowanej sesji; nie ustala poziomów elektrycznych złącza ECU ani wymagania tych linii przez każdy adapter.
- Przed pierwszym `C0 0B 0B D6` (linia pliku 249, ID 248) są **77 transmisji**, w tym dwie `01/01`, `02/02`, odczyty identyfikacji/parametrów oraz cztery `C2 01 01 6B 87`. W `DESKTOP-B2HLV5I-new3.LOG` początek zawiera trzy `01/01`, więc liczba powtórzeń identyfikacji nie jest uniwersalna.
- **[M: trace]** Każde z czterech C2 w tym starcie kończy się READ TIMEOUT ok. 446–480 ms, długość 0. **[H]** jego rola pozostaje nieznana; obecność w starcie nie stanowi dowodu klasyfikacji read-only. Klient Android odtwarza przechwycony start w zakresie żądanym przez użytkownika, z odstępem 450 ms.
- Cyklicznie występują `0B/0B`, `40/29`, `64/A5`, `40/28`, `53/00`, `40/0D`, `40/0F`, `3F/00`. Klient domyślnie odczytuje 0B, uzupełnia 0D/0F dla specjalnej gałęzi ciśnienia i udostępnia opcję pełnego cyklu.
- **Ograniczenie dowodu:** długie odpowiedzi Portmon często mają większy deklarowany `Length` niż liczba widocznych bajtów (np. 0B: Length 51, 21 bajtów tekstu). Nie wolno dopisać zer do uciętego zrzutu ani uznać go za pełny pomiar sumy odpowiedzi.
- **[E]** Porównano `reference/known_commands.json` z AST słownika `responses` w lokalnym `../GasECU Reverse Engineering/src/emulator.py`: 83 wpisy w obu źródłach. Różnice payloadów dotyczą `95/01`, `D4/01`, `61/10`, `61/20`, `61/11`, `61/21`; C2 ma różną reprezentację pustej odpowiedzi (JSON null, oryginał pusty bytes). Archiwum i oryginał pozostają niezmienione. Długości **wszystkich 77** odpowiedzi startowych są zgodne między oryginalnym emulatorem a deklarowanymi długościami logu.
- Implementacja: `../AndroidDashboard/app/src/main/java/com/blazej222/lpgdashboard/protocol/`; generator `../AndroidDashboard/tools/prepare_reference.py`; regresje `DiegoProtocolTest.kt`. Zachowano RAW i kontekst profilu. Nie używa się stałych rejestrów emulatora do obliczania żywego ciśnienia. Fizyczna komunikacja z ECU nie była wykonywana podczas tych prac.
- Dokładne różnice długości [E]: payloady `95/01` i `D4/01` mają w katalogu 40 bajtów, w oryginale 41; cztery payloady `61/*` mają w katalogu 50, w oryginale 52. Żadna z tych odpowiedzi nie należy do użytego startu przed pierwszym 0B.
- Weryfikacja klienta Android: 15/15 testów protokołu, 3/3 instrumentacyjnych na emulatorze API 37, poprawny build debug APK. Testy Python 18 OK / 1 skipped (brak EXE). Lint pozostał niezweryfikowany z powodu timeoutów dwóch zależności narzędzia z repozytorium Google; szczegóły w README AndroidDashboard. Następny krok: pomiar TX/RX z K706 i rzeczywistym kablem USB, porównanie wskazań oraz test odłączenia/ponownego połączenia.
