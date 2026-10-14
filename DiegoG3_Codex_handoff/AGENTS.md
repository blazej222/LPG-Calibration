# AGENTS.md — instrukcje pracy dla Codexa / lokalnego agenta VS Code

**Projekt:** Diego G3 (GAS ECU) — odtwarzanie diagnostycznego protokołu szeregowego i budowa własnego klienta odczytowego. **Język komunikacji z użytkownikiem: polski; kod, nazwy typów i testów: czytelny Python/angielski zgodnie z kontekstem.**

Ten plik ma działać **bez dostępu do historii rozmowy ChatGPT**. Został przygotowany po eksperymentach zakończonych 2026-10-14.

## Pierwsze 5 minut pracy (obowiązkowo)

1. Przeczytaj w tej kolejności: **`PROTOCOL.md` → `REVERSE_ENGINEERING.md` → `README.md` → `TASKS.md`**. Sprawdź katalogi `reference/`, `src/`, `tests/`.
2. Uruchom `python -m unittest discover -s tests -v`. Nie zmieniaj niczego w potwierdzonych algorytmach, jeśli testy przechodzą, bez nowego dowodu.
3. Jeżeli istnieje lokalnie `vendor/DiegoG3-3.0.8.2.exe`, sprawdź jego SHA-256. Oczekiwane: `f182bbf3476fdd68678670de87235329e36f4be8beda2b4c6f88cb147794bd4a`. Nigdy nie analizuj obcej wersji tak, jakby miała identyczne adresy.
4. Domyślnym celem jest **klient odczytowy (read-only)**, nie modyfikowanie map kalibracyjnych ani wyjść instalacji. Nie wysyłaj poleceń do prawdziwego ECU bez jawnego potwierdzenia, że są tylko do odczytu.
5. Prowadź przyrostowy dziennik odkryć w `FINDINGS.md` (jeśli go nie ma, utwórz) i testy dla każdej nowej reguły.

## Twarde ograniczenia jakości i bezpieczeństwa

- **Rozdzielaj pewność**: [X] dowód z EXE, [M] pomiar GUI, [E] tylko historyczny emulator, [H] hipoteza, [N] nieznane. Nie przedstawiaj [H] jako [X], [E] jako [M] ani pola RAW jako temperatury, jeśli tryb konfiguracji jest nieznany.
- **Nigdy nie wracaj do obalonych założeń:** RPM `22_500_000/N` (zamiast właściwego zaokrąglenia); ogólna maska `&0x1FFF` na wszystkie wtryski; bajty 25–40 jako śmieci; temperatury z modelu NTC Beta zamiast fabrycznej LUT; byte41 jako nieznany; bezwarunkowe działanie bajtu 45; komenda `C0 3A 00 FA` jako potwierdzona.
- **Zachowuj surowe wartości i kontekst** w modelach danych: `raw`, `decoded`, `unit`, `confidence`, `config_mode`, `source`. Jeśli tryb czujnika jest nieznany, pole warunkowe powinno zwracać `None` lub oba możliwe warianty z wyraźnymi etykietami, nie arbitralnie wybraną wartość.
- **Nie edytuj** binarnego EXE, oryginalnych CSV ani odzyskanej tablicy LUT w miejscu. Jeżeli ekstrakcja się zmieni, zrób osobny nowy artefakt, udokumentuj hash i diff.
- **Nie wysyłaj sterujących lub zapisujących komend LPG do prawdziwego pojazdu.** Wszelkie testy niepotwierdzonych bajtów wykonuj tylko z wirtualnym/izolowanym emulatorem, a komunikację z realnym ECU ograniczaj do pasywnego monitoringu lub zweryfikowanych poleceń odczytowych w bezpiecznych warunkach.
- **Nie implementuj parsera zależnego od granic `pyserial.read()`**. Odczyt COM to strumień bajtów. Nie rozdzielaj ramek po każdym wystąpieniu `C0`/`D0` wewnątrz payloadu. Znana odpowiedź `0B` ma 50 bajtów payloadu, 52 bajty `D0+payload+checksum`.
- **Nie nazywaj sumy kontrolnej polinomialnym CRC**. Żądanie `C0`: `sum(first_three)&0xFF`; odpowiedź emulatora: `(sum(payload)-0x30)&0xFF`.
- **Nie utożsamiaj** `payload[41]` (MAP) i `payload[1]` (ciśnienie gazu), nawet gdy funkcje konwersji mają podobne stałe.
- **Maski**: benzyna1 = `0x1FFF` i 3-bitowy status; benzyna2 = `0x7FFF` i flaga G; benzyna3–8 i gaz1–8 = pełne `uint16`; skala `/187.0`.
- **Stany gazu** `0..7` zachowaj jako osobne kody; `1 != 2`, `6 != 7`, chociaż tekst GUI się pokrywa.

## Repo i konwencje zmian

- `PROTOCOL.md` — **specyfikacja wire format + wszystkie znane odpowiedzi + evidence tags**. Gdy odkryjesz pole/komendę, uaktualnij ten dokument.
- `REVERSE_ENGINEERING.md` — adresy VA i funkcje w Delphi, log eksperymentów i plan. Nowy dowód: `VA, instrukcje, format wejścia/wyjścia, przykład RAW, zachowanie GUI, status dowodu`.
- `reference/known_commands.json` — **katalog statycznych odpowiedzi 83 zapytań**; nie oznaczaj tych danych „real ECU capture”. Długie payloady zostały przepisane z historycznego kodu; w razie dostępności oryginalnego `emulator.py` porównaj byte-by-byte.
- `reference/temperature_lut.csv` — **256 rzeczywistych wartości double**, wyekstrahowanych z konkretnego EXE, indeksowanych `raw_byte`.
- `reference/DiegoG3_measurements*.csv` — **pomiarowe fixture'y**; pozostają niezmienione. 5 RPM, 10 status/ikona G, 15 MAP.
- `reference/tred_measurements_from_conversation.csv` — 55 pomiarów Tred odtworzonych z historii (źródło tekstowe, nie oryginalny CSV); zgodność 55/55 z LUT.
- `src/diego_protocol.py` — czyste funkcje dekodujące i granice odpowiedzi. Kod nie wykonuje automatycznych operacji na serialu.
- `tests/test_protocol.py` — dodawaj testy regresyjne przed/razem ze zmianą dekodera; całość musi przechodzić.
- `tools/` — ekstraktory danych oraz opcjonalny `emulator_cli.py` (wyłącznie emulator laborator.; nie klient fizycznego ECU).
- `vendor/` — opcjonalny lokalny plik EXE (w `.gitignore`), **nigdy nie commituj** binariów producenta, ustawień prywatnych czy dumpów prywatnych urządzeń bez świadomej zgody.

## Zasady implementacji

- Python 3.10+; do warstwy protokołu standard library wystarczy. `pyserial` dopiero w warstwie I/O. Unikaj ciężkich zależności bez uzasadnienia.
- Publiczne funkcje mają mieć typy, walidację `uint8/uint16`, konwencję BE w payloadzie, testy na 0, 255, 0xFA, błędne długości, negatywne czasy, maksymalne kanały i brakujący tryb czujnika.
- Widoczny wynik w bar/°C/ms może być zaokrąglony **wyłącznie na warstwie GUI**, przechowuj wartości `float` pełnej precyzji. Np. temperatura `0x69=33.53631112`, MAP `0x17≈0.40605`.
- Dla obliczenia RPM z danym słowem: `0 if N==0 else 10*round(2_244_000/N)`. Zaokrąglanie w punktach połowicznych na x87 może różnić się w sytuacjach brzegowych; osobno przetestuj, jeśli to istotne.
- Interaktywny emulator nie powinien nadpisywać wszystkich statycznych odpowiedzi. Zachowuj oryginalne wartości RAW nieznanych pól. Wszelkie przełączanie stanu ma zachować bity czasu.
- Nowy plik pomiarów ma być tworzony pod **nową nazwą**; nie dopisuj do archiwalnych fixture'ów bez zgody i kontroli schematu.

## Najbliższy zakres prac (priorytet)

Zacznij od `TASKS.md`: **(1)** parser `0B` i testy już istnieją, **(2)** rozpoznanie bajtu 45 z bramką capabilities i negatywnym trace `C0 3A 00 FA`, **(3)** semantyka bajtu 5 (poziom gazu), **(4)** nieznane 42–44/46/48–49, **(5)** wyjaśnienie konfiguracji trzeciego czujnika temperatury i trybu MAP, **(6)** własny klient tylko do odczytu.

Zawsze zgłaszaj w odpowiedzi: co jest [X]/[M], co jest [H], jakie testy uruchomiłeś i co pozostaje nieustalone. Jeżeli nie możesz zweryfikować hipotezy ze statycznego EXE lub CSV, zaproponuj **konkretny skan bajtów i oczekiwane rozróżnienie**, nie swobodną interpretację.
