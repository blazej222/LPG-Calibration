# REVERSE_ENGINEERING.md — DiegoG3 3.0.8.2 / udokumentowana ścieżka badawcza

> **Stan na 2026-10-14.** Wersja programu: `DiegoG3-3.0.8.2.exe`, 32-bitowy natywny Borland Delphi 5. Reverse engineering programowo i eksperymentalnie, nie z dokumentacji producenta. Dokument jest samodzielnym protokołem badań i listą otwartych zadań; implementacyjne szczegóły ramek są w [`PROTOCOL.md`](PROTOCOL.md).

## 1. Dowody, ich pochodzenie i stopnie pewności

| Oznaczenie | Co potwierdza | Co **nie** wynika |
|---|---|---|
| **[X] EXE** | Konkretne instrukcje x86, maski, tablice, stałe w podanym SHA EXE | Że każdy wariant ECU używa tej samej gałęzi |
| **[M] POMIAR** | Zachowanie programu DiegoG3 dla celowo zmienionej ramki emulatora | Wierność parametrów fizycznego silnika |
| **[E] EMULATOR** | Statyczna odpowiedź dawnego Python `process_data` | Wartość w rzeczywistym ECU, bezpieczeństwo polecenia |
| **[H] HIPOTEZA** | Prawdopodobna interpretacja na podstawie kontekstu/GUI | Pewna semantyka bitu/sygnału |
| **[N] NIEZNANE** | Wyłącznie przechowywanie wartości / jej offset | Właściwy algorytm przeliczenia |

Wszystkie ustalenia dotyczą konkretnej wersji pliku; po zmianie EXE koniecznie ponowić statyczną analizę. **Domyślnie koduj tylko to, co [X] lub [M]; hipotezy mają być jawnie oznaczone także w kodzie.**

## 2. Artefakty źródłowe dostępne do odtworzenia

| Artefakt | Rola | Gdzie w handoffie |
|---|---|---|
| `DiegoG3-3.0.8.2.exe` | Binarna implementacja klienta diagnostycznego (użytkownik ma oryginał) | **NIE jest dystrybuowany w ZIP**; skopiować lokalnie do `vendor/` poza kontrolą wersji |
| `DiegoG3_measurements.csv` | 5 rzeczywistych pomiarów emulator → GUI RPM | `reference/` (kopia dostarczonego CSV) |
| `DiegoG3_measurements(1).csv` | 8 kodów stanu ECU + 2 pomiary ikony „G” | `reference/` |
| `DiegoG3_measurements(2).csv` | 15 par RAW41 → wskazania MAP bar | `reference/` |
| `temperature_lut.csv` | Wszystkie 256 wartości temperatury IEEE754 double z EXE | `reference/`, procedura odtworzenia niżej |
| `tred_measurements_from_conversation.csv` | 55 unikalnych pomiarów Tred przepisanych z tekstu rozmowy | `reference/` (historyczna transkrypcja, brak oryginalnego CSV) |
| `known_commands.json` | Pełny katalog statycznych odpowiedzi emulatora | `reference/`; transkrypcja historycznego słownika Pythona; długie ciągi warto porównać z oryginalnym `emulator.py` |
| `src/diego_protocol.py` | Referencyjny dekoder/konwertery/warstwa framingu | Implementacja testowalna bez portu COM |
| `tests/test_protocol.py` | Testy regresyjne używające CSV i katalogu | Uruchomić jako pierwsze |

### 2.1 Tożsamość EXE

```text
File: DiegoG3-3.0.8.2.exe
SHA256: f182bbf3476fdd68678670de87235329e36f4be8beda2b4c6f88cb147794bd4a
Format: PE32 / i386 / Windows GUI / Borland Delphi 5 (sygnatura Detect It Easy)
ImageBase: 0x00400000
EntryPoint VA: 0x005B33BC
CODE  VA 0x00401000, file offset 0x00000400, virtual size 0x001B263C
DATA  VA 0x005B4000, file offset 0x001B2C00, virtual size 0x00006090
.rsrc VA 0x005E0000, file offset 0x001D8C00, file raw size 0x00149C00
```

Zagrożenie „PKLITE32” wyświetlone przez DiE było oznaczone tylko jako **suspicion**. Sekcje (`CODE`, `DATA`, `BSS`, `.idata`, `.tls`, `.rdata`, `.reloc`, `.rsrc`) są typowe dla Delphi; brak potwierdzenia spakowania, brak podstaw do modyfikowania EXE. Importy `ReadFile`/`WriteFile` umożliwiają tracing I/O w debuggerze.

Przy konwersji **VA → file offset**:

```python
# Tylko w zakresach sekcji pokazanych powyżej:
file_offset_CODE = VA - 0x00401000 + 0x00000400
file_offset_DATA = VA - 0x005B4000 + 0x001B2C00
```

Uwaga: używaj offsetu z nagłówków PE aktualnej wersji; nie stosuj stale jednej formuły dla całego EXE.

### 2.2 Narzędzia i bezpieczne uruchomienie

- **DiE** — identyfikacja kompilatora/packera; **IDR (Interactive Delphi Reconstructor)** — formularze, RTTI, `published` methods; **Ghidra** — x86/dekompilacja/xref; **x64dbg** — śledzenie dynamiczne tylko w odizolowanej maszynie Windows. Można używać `objdump -D -Mintel` do szybkiego zrzutu statycznego.
- Tylko kopia EXE oraz pliki DLL wymagane przez klienta; nie uruchamiać nieznanego EXE ze specjalnymi uprawnieniami na podstawowym komputerze. Nie podłączać podczas testów reverse engineeringu do **pracującej instalacji LPG**.
- Do pracy z portem COM korzystać z pary portów wirtualnych / emulacji offline; wtyczka Codex w VS Code ma wykonywać testy parsera offline przed testami I/O.

## 3. Lokalizacje faktycznie przeanalizowanych procedur x86

| VA | Funkcja / dane | Czego dowodzi | Uwaga |
|---|---|---|---|
| `0x00564E90` | **Parser odpowiedzi `0B`** | kolejność i zapisy payloadu w strukturze, 8+8 kanałów | Najważniejszy punkt kontynuacji RE |
| `0x00564F4D–0x00564F5B` | Składanie RPM | `payload[7]<<8 | payload[8]` → słowo BE | Zapisywane np. do `[ebp+0x48E]` |
| `0x00564F67–0x00564F99` | Status i czas benzyny 1 | `and 0xE0`, `shr 5`, `and 0x1F`, zapisy statusu i czasu osobno | `[ebp+0x48C]` status, `[ebp+0x490]` odmaskowany licznik |
| `0x00564F7C–0x00564FB0` | Flaga i czas benzyny 2 | `test 0x80`, maska `0x7F`, konstrukcja BE | `[ebp+0x520]` flaga, `[ebp+0x494]` czas |
| `0x005650DC–0x0056511F` | Końcówka payloadu | RAW41→`[ebp+0x486]`, 42..49→poszczególne bajty obiektu | Dokładne storage offsets opisane niżej |
| `0x00563D58` | Odczyt temperatury LUT | `table[255-raw]` | [X] |
| `0x005B8B44` | Tablica temperatur | `double[256]`, offset pliku `0x001B7744` | Data array, indeks odwrotny |
| `0x00563DEC` | Czas wtrysku | `raw_uint16 / 187.0` | `187.0` pod `0x00563E0C` |
| `0x00563EA0` | RPM | `10*round(2_244_000/N)`, specjalny przypadek N=0 | Stałe `187`, `1000`, `120`, `10` |
| `0x00563950` | Ciśnienie gazu | tryb standardowy i gałąź RAW>=`FA` z rejestrami `0D/0F` | Wybór zależy od konfiguracji |
| `0x00563BC8` | Ciśnienie MAP | dwa wzory skalowania zależne od typu czujnika | Badana konfiguracja używa `(x/256+0.00842)/0.242` |
| `0x005638EC` | Sygnał napięciowy | `raw * 5/256 /0.215` | **[H]** fizyczna nazwa „zasilanie” |
| `0x0057BEB0` | Analog 6 | 0.02 V/LSB w wybranych konfiguracjach | **[H]** lambda |
| `0x005826C2` | Dodatkowe temperatury | `payload[47]` do procedury temperatury przy spełnionej bramce | Fallback na alternatywny parametr `0x1B`, gdy bramka nieaktywna |
| `0x00582570–0x005825F3` | Bajt 45 | osiem testów bitów i powiązanie z kontrolkami GUI | Źródło danych warunkowe, brak widocznego efektu w emulatorze |
| `0x0056409C` | Porównanie wariantu/możliwości | warunek przełączenia ścieżek GUI, np. byte45 i byte47 | Dokładna publiczna nazwa trybu [H] |
| `0x0056ADF0` | Możliwy generator komendy `3A` | trop w ścieżce alternatywnych statusów | `C0 3A 00 FA` **niezaobserwowane** na `trace all` |
| `0x0057F97C` | `ReadOscValues` | zbieranie odczytów/wywołania dekoderów | Odzyskana nazwa Delphi |
| `0x00580348` | `TimerekTimer` | callback GUI | Nazwa i VA |
| `0x0058F638` | `TimerMapaTimer` | callback GUI/mapa | Nazwa i VA |
| `0x00599874` | `UpdateComPortInfo` | GUI portu | Nazwa i VA |

Pełny wykaz metod Delphi można odzyskać z tablic `published methods`. W poprzedniej analizie odczytano **662 metody**. Plik z tą listą nie jest niezbędny do implementacji protokołu, więc nie traktować tych adresów jako stabilnego API.

### 3.1 Fragment krytyczny statusów (rekonstrukcja semantyczna)

```asm
; wejście parsera wskazuje bufor o offset przesunięty względem payloadu:
; benzyna 1: starszy bajt payload[9]
0x00564F67 and dl, 0E0h
0x00564F70 shr edx, 5
0x00564F73 mov [ebp+048Ch], dl     ; status 0..7
0x00564F88 and al, 01Fh          ; starszy bajt czasu: dolne 5 bitów
...
; benzyna 2: starszy bajt payload[11]
0x00564F7C test dl, 80h
0x00564F7F seta cl
0x00564F82 mov [ebp+0520h], cl     ; bool dla GUI
0x00564F9F and dl, 07Fh          ; starszy bajt czasu: dolne 7 bitów
```

Nie stosować maski do „wszystkich wtryskiwaczy”. Parser rozróżnia dokładnie **benzynę 1**, **benzynę 2** i **pozostałe słowa**.

### 3.2 Dokładne przypisanie końcówki `0B` do obiektu

Odczytano w `0x005650DC..0x0056511F`:

```text
payload[41] -> [ebp+0x486]  MAP
payload[42] -> [ebp+0x0B5]  nieznane (nie mylić z 0B)
payload[43] -> [ebp+0x018]  nieznane
payload[44] -> [ebp+0x019]  nieznane
payload[45] -> [ebp+0x01A]  maska 8 bitów
payload[46] -> [ebp+0x01B]  boolean (0 albo 1)
payload[47] -> [ebp+0x01C]  surowa dodatkowa temperatura w jednym wariancie
payload[48] -> [ebp+0x016]  nieznane
payload[49] -> [ebp+0x01D]  nieznane
```

Jeżeli Codex bada xrefy: nie sugerować się samym numerem offsetu w `esi` (jest przesunięty w stosunku do naszej indeksacji). Korelować bajty z rzeczywistym payloadem i wywołaniami.

## 4. Wyniki eksperymentów i ich ograniczenia

### 4.1 RPM — pięć **zweryfikowanych** pomiarów

**Źródło:** `reference/DiegoG3_measurements.csv` (5 wierszy). Dekodowanie `uint16be(payload[7:9])` i `10*round(2_244_000/N)`:

| RAW | N | GUI | Wzór | Wniosek |
|---|---:|---:|---:|---|
| `65BA` | 26042 | 860 | 860 | identyczny |
| `659A` | 26010 | 860 | 860 | zmiana LSB nie zmienia wskazania przez zaokrąglenie |
| `65DA` | 26074 | 860 | 860 | jak wyżej |
| `45BA` | 17850 | 1260 | 1260 | zmiana starszego bajtu daje +400 |
| `85BA` | 34234 | 660 | 660 | odwrotność okresu |

Dawny komentarz „-32 w młodszym bajcie = +10 RPM” został **obalony dla tej zarejestrowanej serii**. Nie wymuszać takiego przesunięcia w dekoderze.

### 4.2 Stany instalacji i flaga „G”

**Źródło:** `reference/DiegoG3_measurements(1).csv` (10 wierszy).

- Bajt `9`: `01,21,41,61,81,A1,C1,E1` → kody `0..7`, teksty `off,zaplon,zaplon,benzyna(czerwone ciśnienie),czekam,gaz,benzyna(nieczerwone),benzyna(nieczerwone)` [M].
- Słowo `11..12`: `01 76` → ikonka G **wyszarzona**; `81 76` → **czerwona**, obie z czasem `2.0ms` [M].
- To jest potwierdzony efekt na GUI, **nie** potwierdzony bit fizycznego elektrozaworu.

### 4.3 MAP — 15/15 punktów

**Źródło:** `reference/DiegoG3_measurements(2).csv` (15 wierszy). RAW: `00,08,10,17,20,30,40,50,60,80,A0,C0,E0,F0,FF`; wartości GUI: `0.03,0.16,0.29,0.41,0.55,0.81,1.07,1.33,1.58,2.10,2.62,3.13,3.65,3.91,4.15` bar.

Wszystkie 15 wyników = `round((raw/256+0.00842)/0.242,2)` w bieżącej konfiguracji. Największy błąd bez wcześniejszego zaokrąglenia ~0.0048 bar. Wcześniejsza próba linearyzacji NTC nie dotyczy MAP, nie powinna być używana.

### 4.4 Temperatury — 256-elementowa LUT, niezależna od regresji

1. Pierwotnie zmierzono kilkadziesiąt par RAW/Tred, złożona, nieliniowa charakterystyka NTC.
2. W regresjach dopasowywano Beta ~3900–3950K / model czteroparametrowy. W przybliżeniu błędy wynosiły 0.1–0.5°C, występowały nierówne różnice kolejnych RAW.
3. Zamiast dalszego fitu przeanalizowano EXE: pod `0x00563D58` następuje bezpośrednie odwołanie do tabeli `double[255-raw]`.
4. Reprodukowalny ekstraktor `tools/extract_temperature_lut.py` potwierdza 256 wartości (np. `69→33.53631112`, `80→24.92070832`, `88→21.99276177`, `F0→-27.948317662...`).
5. Wszystkie 55 unikalnych kodów z ręcznej transkrypcji pomiarów użytkownika `reference/tred_measurements_from_conversation.csv` zgadzają się z LUT po zaokrągleniu do 0.1°C.
6. Wcześniejszy przelicznik liniowy `T=C-0.5*raw`, model wielomianowy czy globalny sztywny krok 0.9°C to **nie implementacja rzeczywistej funkcji EXE**.

### 4.5 Ciśnienie gazu — anomalia `FA` wyjaśniona

**Pierwsze eksperymenty:** `0xF9` ≈ 2.53 bar, `0xFA,0xFE,0xFF` stale 1.91 bar. Dawny model liniowy pasował tylko do `F9`.

**Kod EXE `0x00563950`:** porównanie z `0xFA`, przejście do gałęzi z różnicą rejestrów 0D/0F zależne od typu czujnika. Dla historycznych rejestrów `008E` i `001A`: różnica `116` → `1.9072107...` bar ⇒ GUI 1.91. Nie traktować `FA` jako „overflow” kodu prostego pomiaru w każdej konfiguracji; uruchamia specjalny sposób obliczania w **danym trybie**.

## 5. Lista błędnych lub przestarzałych interpretacji — NIE PRZYWRACAĆ

| Starsza hipoteza | Dlaczego nieaktualna | Poprawna postać |
|---|---|---|
| `RPM≈22_500_000/N` bez zaokrąglania | Znaleziono rzeczywistą funkcję i pomiary 5/5 | `10*round(2_244_000/N)` |
| `-32` w młodszym bajcie = zawsze +10 RPM | CSV potwierdza `659A/65BA/65DA` → każde 860 | Big Endian 16-bit i zaokrąglenie |
| `time_ms=(raw-base)/187` dla każdej pary | `A000`,`8000` to bity statusu, a nie stałe kalibracyjne | `&1FFF` tylko B1, `&7FFF` tylko B2, reszta bez maski |
| `&1FFF` dla wszystkich wtryskiwaczy | EXE maskuje tylko szczególne dwa słowa benzyny | 8+8 słów zgodnie z maskami w PROTOCOL |
| Bajty `25–40` są niezidentyfikowane/zerowe | Parser zapisuje czas wtrysku kanałów 5..8 | osiem dodatkowych uint16 BE/187 |
| Ciśnienie `0.01058*raw-0.107` w każdej wartości | EXE ma dokładniejsze stałe i specjalny warunek `FA` | `(raw/256-0.04)/0.369`, alternatywnie rejestry 0D/0F |
| Temperatura globalną prostą/NTC Beta fit | Zidentyfikowano gotową LUT | `double[255-raw]`, skopiowane jako raw→°C |
| Bajt 45 zawsze steruje widocznymi 8 ikonami | Test GUI negatywny, kod ma gałąź zależną od możliwości ECU | 8-bit pole z **warunkowym** użyciem; semantyka [N] |
| Komenda `C0 3A 00 FA` musi być wysyłana | W `trace all` jej **nie było** | nadal tylko trop statyczny warunkowej ścieżki |
| „Bajt 41 nieznany” | Użytkownik zidentyfikował MAP, 15/15 pomiarów | `(raw/256+0.00842)/0.242` w danej konfiguracji |
| „T3 na pewno zawsze byte47” | EXE wybiera inne źródło w części konfiguracji | warunkowe `payload[47]`, fallback rejestr `0x1B` |

## 6. Co nadal pozostaje otwarte i jak je rozpracować (priorytety)

### P0 — Zweryfikuj warstwę transportu na realnym ECU, bez zapisu

**Cel:** oddzielić działanie emulatora od działania ECU. Zapisz pasywny trace `timestamp, direction, port settings, bytes, checksum, request correlation, UI action` dla inicjalizacji i cyklicznego odczytu. Podłącz klienta wyłącznie w bezpiecznym, nieruchomym zestawie laboratoryjnym. Nie używaj nierozpoznanych komend do modyfikowania map LPG, ciśnienia ani wyjść.

**Rezultat:** lista komend read-only i dokładne warunki długości/timeoutów, potwierdzenie że `D0+PAYLOAD+CKS` i suma minus `0x30` są właściwe również poza emulatorem.

### P1 — Bajt 45, maska 8 bitów (bramka możliwości)

W `0x00582570` wyszukać xrefy do `[esi+0x1A]` i dekompilować obie gałęzie po wywołaniu `0x0056409C`. Jeśli GUI korzysta z `payload[45]`, przetestować `00,01,02,04,08,10,20,40,80,FF`, obserwując **konkretny ekran i stan**. Jeśli bramka nieaktywna, osobno sprawdzić z jakiego źródła aplikacja pobiera dane (wskazówka `0x0056ADF0`, możliwe `3A`; nie wysyłać do realnego ECU bez potwierdzenia).

**Rezultat:** tabela `bit → funkcja UI / warunek`, a nie tylko niezmienione kontrolki.

### P2 — Fizyczny sens bajtu 5 (poziom LPG) i bajtu 6 (analog/lambda)

Z xref `0x005879F3` i funkcji analogowej `0x0057BEB0` odzyskać pełny warunek kalibracji czujnika. W emulacji offline skan byte5: `00,20,40,6B,80,C0,FF`, byte6 analogicznie; zapisać konfigurację typu czujnika, GUI wskaźnik procentów/litrów/volts, warianty saturacji. Nie zakładać stałej liniowej jednostki poziomu LPG.

**Rezultat:** odpowiednie LUT/skale + mapowanie konfiguracji, oba rozdzielone od spekulacji.

### P3 — Fizyczna nazwa temperatury 3 i konfiguracja MAP

W `0x005826A0..0x00582707` jest ścieżka dla `payload[47]` oraz fallback parametru `0x1B`. Odtworzyć przełączający warunek i odczytać rzeczywistą etykietę GUI. Podobnie dla `0x00563BC8` odtworzyć `sensor_mode`, zaznaczać w dekoderze tryb podany/ustalony, nigdy wyliczać trybu z samego RAW.

**Rezultat:** `map_mode`, `third_temp_source`, precyzyjne nazwy fizyczne i status konfiguracji.

### P4 — Końcówka `42..44`, `46`, `48..49` + semantyka kodów statusu

Zacząć od xrefów do obiektu `[ebp+0xB5]`, `[ebp+0x18]`, `[ebp+0x19]`, `[ebp+0x1B]`, `[ebp+0x16]`, `[ebp+0x1D]`. Dopiero potem skany pojedynczego bajtu (`00,01,7F,80,FF`) z kontrolą innych pól. Przy statusach `1/2` i `6/7` sprawdzić różnicę oprócz identycznej nazwy GUI; dla `gas_icon` określić, czy odzwierciedla stan wyjścia czy tylko prezentacji. Obserwować wirtualne kontrolki/oscyloskop/ikonę poziomu LPG i logować brak zmian.

**Rezultat:** klasyfikacja każdego bitu pola wraz z konkretnym dowodem, nie ogólna etykieta „status”.

### P5 — Mapa pozostałych komend z historycznego emulatora

Priorytet: `01`,`02`,`53 xx`,`40 xx`,`07 xx`,`51 xx`,`12 xx`,`60/61/64` oraz pozostałe identyfikatory. Zarejestrować moment wysłania w oryginalnym GUI (startup / cykliczny odczyt / otwarcie zakładki / zapis). Nie inferować odczytu/zapisu tylko z długości odpowiedzi; nie wysyłać komend niepotwierdzonych do pracującego ECU. Zbudować enum komend dopiero po potwierdzeniu semantyki.

**Rezultat:** rozdział komend na `read_only`, `mutation`, `handshake`, `unknown` wraz z parametrami i rozmiarem odpowiedzi.

## 7. Architektura docelowego własnego klienta

1. `transport`: port COM, 9600 8N1 **jako zweryfikowany punkt wyjścia**, konfiguracja przez CLI/pliki; timeouts, log bajtów TX/RX; obsługa odłączeń.
2. `framing`: dekoder strumienia inkrementalny, checksums, resynchronizacja, oczekiwane długości per zapytanie, bez nadinterpretacji chunków `pyserial.read()`.
3. `codec`: czyste funkcje z `src/diego_protocol.py`; wszystkie dane surowe dostępne obok przeliczonych, konfiguracja jawnie.
4. `service`: bezpieczny scheduler **potwierdzonych read-only** zapytań + reakcja na timeouty/CRC błędy, interfejs do mocka; odkładać komendy mapująco-konfiguracyjne.
5. `model/UI`: wskazanie temperatur, MAP, ciśnienia gazu, RPM, czasów 1..8, stanu LPG, znaczników jakości/pewności; pole nieznane jako RAW nie „fake human-readable”.
6. `tests`: fixture podstawowego `0B`, CSV, checksum request/reply, granice `FA`, wycinki `read()`, 8+8 słów, maski obu pól, brak pomyłek offsetów i żadnego automatycznego nadawania w testach.
7. `trace/analysis`: osobny tryb offline do wczytywania logów binarnych, porównywania ramek i definiowania eksperymentów; nie scalać z kodem sterującym autem.

## 8. Kryteria gotowości do przejścia na nowy etap

- Wszystkie testy `python -m unittest discover -s tests -v` przechodzą.
- `src/diego_protocol.py` dekoduje bazowy `0B` z dokładnymi wskazaniami i danymi RAW.
- 5/5 testów RPM, 15/15 testów MAP, 10/10 testów stanu/ikony, 55/55 Tred, pełne 256 elementów LUT, katalog request-reply ma poprawne sumy.
- Nie ma błędnego globalnego `&0x1FFF`, `round(22_500_000/N)` ani nieopisanej zamiany MAP z ciśnieniem gazu.
- Nieznane pola i tryby są `None`/RAW/[H], a nie sztucznie przyjętym pomiarem.
- Każda nowa odkryta reguła ma **dowód**: VA+instrukcje EXE albo minimum kilka niezależnych par `raw→GUI`, preferencyjnie oba.
- Zero komend zmieniających parametry ECU bez własnej, potwierdzonej specyfikacji i izolowanego środowiska testowego.

## 9. Powtarzalne polecenia badawcze (przykłady)

```powershell
# PowerShell, katalog repo
python tools/extract_temperature_lut.py vendor/DiegoG3-3.0.8.2.exe --out reference/temperature_lut.csv
python -m unittest discover -s tests -v
python src/diego_protocol.py
```

```bash
# Linux / WSL - podgląd x86 z konkretnym VA
objdump -D -Mintel --start-address=0x00564E90 --stop-address=0x00565125 vendor/DiegoG3-3.0.8.2.exe
objdump -D -Mintel --start-address=0x00563D58 --stop-address=0x00563D82 vendor/DiegoG3-3.0.8.2.exe
objdump -D -Mintel --start-address=0x00563EA0 --stop-address=0x00563EE6 vendor/DiegoG3-3.0.8.2.exe
```

**Dla Codexa:** kontynuuj od `PROTOCOL.md` sekcje 2–3, `REVERSE_ENGINEERING.md` sekcja 6 (P1). Bez nowego dowodu nie zmieniaj potwierdzonych konwersji ani sposobu składania Big Endian.
