# PROTOCOL.md — DiegoG3 ECU / komunikacja diagnostyczna

> **Zakres:** program DiegoG3 **3.0.8.2**, 32-bitowy Delphi 5, emulowany protokół na łączu szeregowym. Aktualizacja 2026-10-14. Dokument samodzielny do wdrożenia dekodera **read-only**, nie jest specyfikacją sterowania instalacją LPG. **Czytaj równolegle `REVERSE_ENGINEERING.md` i `AGENTS.md`.**

## 0. Słownik wiarygodności (obowiązkowy)

- **[X] KOD EXE** — odczytana instrukcja albo stała w *konkretnym* EXE. Potwierdza zachowanie tej wersji oprogramowania, niekoniecznie prawdziwego ECU.
- **[M] POMIAR** — użytkownik zmieniał payload emulatora i zapisał wskazanie fabrycznego GUI; podajemy liczbę punktów i zakres.
- **[E] EMULATOR** — historyczny słownik `process_data` dostarczony przez użytkownika; jest to **przykład odpowiedzi**, nie dowód kompletnej implementacji komendy w ECU.
- **[H] HIPOTEZA** — nazwa fizyczna, semantyka bitu, konfiguracja albo kierunek transmisji nie zostały potwierdzone.
- **[N] NIEZNANE** — tylko offset i surowa wartość.

**Nie interpretować `E` jako `M`.** Pola potwierdzone w kodzie EXE mogą być wykorzystywane **warunkowo**, zależnie od konfiguracji/wariantów ECU.

## 1. Transport i kierunki

- Port RS-232 lub wirtualny COM; użyty emulator: `COM18`, **9600 bodów** [E]. Liczba bitów danych, parzystość, stop: ustawienia `pyserial.Serial` domyślnie **8N1** [E]. Sterowanie przepływem nie było jawnie ustawiane; domyślnie `xonxoff=False`, `rtscts=False`, `dsrdtr=False` [E]. Rzeczywiste poziomy elektryczne i sprzętowe złącze ECU **nieustalone**.
- **Żądania programu (PC → ECU):** typowo 4 bajty `C0 CMD ARG CHECK`, gdzie `CHECK=(0xC0+CMD+ARG) mod 256` [E]; wiele konkretnych przykładów jest zgodnych. Nie mylić sumy z wielomianowym CRC.
- **Wyjątek:** `C2 01 01 6B 87` to specjalny 5-bajtowy komunikat w historycznym emulatorze, nie wywołuje odpowiedzi [E]; znaczenie protokołowe [H]. Nie przykładać automatycznie składni `C0` do `C2`.
- **Odpowiedź ECU → PC (format odtworzony przez emulator):** `D0 PAYLOAD CKS`, gdzie `CKS=(sum(PAYLOAD)-0x30) & 0xFF` [E]. `D0` jest znacznikiem odpowiedzi, **nie wlicza się** do CKS. Weryfikacja tej reguły niezależnie na rzeczywistym ECU i dla wszystkich wariantów pozostaje zadaniem.
- **Brak pola długości** widocznego w powyższej warstwie dla typowych odpowiedzi; długość wyznaczana z kontekstu zapytania / mapy komend [E/H]. Dla `0B` payload to **50 bajtów**, więc całkowita odpowiedź na przewodzie: **52 bajty** (`D0 + 50 + CKS`).
- Port szeregowy jest **strumieniem bajtów, nie strumieniem ramek**. `read(in_waiting)` może zwrócić `C0 53 00` i dopiero później `13`. Buforować i parsować po długości; nie bazować na granicach `read()` ani na `sleep()`.
- Payload może zawierać bajty `C0`/`D0` (np. komendy `C0 C0 00 80`, odpowiedzi na `D0`, `0B` mogą w teorii zawierać dowolne bajty). Nie rozcinać ramek po wystąpieniu bajtu znacznika **wewnątrz** znanej długości. Bufor resynchronizacji powinien walidować sumy i długości.
- Dla danych wejściowych z nieznaną długością należy rejestrować surowy strumień, a nie zgadywać granice. Zadawanie poleceń zmian konfiguracji rzeczywistemu ECU bez potwierdzenia read-only jest zabronione w tej ścieżce pracy.

### 1.1 Wzory i przykład rzeczywistego przesyłu emulatora

```python
request = bytes([0xC0, command, argument, (0xC0 + command + argument) & 0xFF])
reply_wire = bytes([0xD0]) + payload + bytes([(sum(payload) - 0x30) & 0xFF])
```

- `C0 53 00 13` → `D0 53 4A 6D` (`0x53 + 0x4A - 0x30 = 0x6D`).
- `C0 40 0D 0D` → payload `40 00 8E` → `D0 40 00 8E 9E`.
- `C0 40 0F 0F` → payload `40 00 1A` → `D0 40 00 1A 2A`.
- `C0 0B 0B D6` → `D0` + **50-bajtowy payload** (poniżej) + suma **`77`** (52 bajty na przewodzie).

Ostatnie dwa pakiety (`40 0D`, `40 0F`) są **zrzutami emulatora**. W EXE wartości odpowiadają rejestrom/parametrom `0D` i `0F` używanym w specjalnym trybie przeliczania ciśnienia gazu.

## 2. Cała odpowiedź `0x0B`: struktura, indeksy, bitowość

### Uzupełnienie transportu 2026-10-14 — lokalny trace i klient Android

Lokalne `../GasECU Reverse Engineering/data/PORTMAN.LOG` i `DESKTOP-B2HLV5I-new3.LOG` pokazują 9600 8N1 oraz SET_DTR/SET_RTS **[M: pasywny trace]**, uzupełniając wcześniejszy dowód wyłącznie z emulatora. Start przed pierwszą ramką 0B w PORTMAN zawiera 77 żądań, w tym cztery `C2 01 01 6B 87` z timeoutami ok. 450 ms. Semantyka C2 pozostaje **[H]**. Długie wpisy logów mają często ucięty tekst HEX; deklarowany `Length` nie oznacza dostępności pełnej odpowiedzi do weryfikacji sumy.

Klient Android w `../AndroidDashboard/` odtwarza ten konkretny start, serializuje request/reply i potem odczytuje głównie 0B. Warunkowo pobiera aktualne 0D/0F. Szczegóły źródeł, długości i różnic katalogu względem oryginalnego emulatora zapisano w `FINDINGS.md` oraz `../AndroidDashboard/README.md`. Nie zmienia to klasyfikacji pozostałych komend na read-only/mutation/unknown ani nie potwierdza znaczenia C2.

**Zapytanie:** `C0 0B 0B D6`.

**Przykład referencyjny**: payload z oryginalnego emulatora; ten sam payload bazowy zapisany w trzech CSV pomiarowych. Kolejne linie mają po 10 bajtów (offsety dziesiętne w nawiasach):

```text
[00..09]  0B 79 69 93 9B 6B 17 65 BA A1
[10..19]  76 81 76 01 76 01 76 01 76 01
[20..29]  76 01 76 01 76 00 00 00 00 00
[30..39]  00 00 00 00 00 00 00 00 00 00
[40..49]  00 17 72 00 00 FF 01 88 00 02
```

**Uwaga na offsety:** indeks `0` to `0B` (ID wiadomości) **po** `D0`, nigdy `D0`. Parser w EXE miewa przesunięcie wskaźnika `esi` o jeden: przykładowo instrukcja `mov al,[esi+0x2A]` odpowiada **`payload[41]`**, a nie `payload[42]`. Mapa poniżej została uzgodniona z przykładowymi pomiarami i strukturą parsera.

| Offset w payloadzie | Bajty przykładu | Pole, interpretacja | Pewność / ograniczenia |
|---|---|---|---|
| `0` | `0B` | ID payloadu/status zbiorczy `0B` | [E][X] |
| `1` | `79` | Ciśnienie gazu (bar), standardowa gałąź lub `FA..FF` → rejestry `0D/0F` | [X], standardowy wzór zgodny z wcześniejszymi pomiarami; aktywna gałąź zależy od konfiguracji |
| `2` | `69` | Temperatura reduktora (°C), `LUT[raw]` | [X][M] |
| `3` | `93` | Temperatura gazu (°C), ta sama LUT | [X], warto sprawdzić odrębnie wyświetlenie w GUI |
| `4` | `9B` | Wartość napięciowa `5*raw/256/0.215` → ~14,08 V; prawdopodobnie zasilanie 12V | [X] wzór, **[H]** nazwa fizyczna |
| `5` | `6B` | Pole wykorzystywane przy wskaźniku poziomu LPG | [X] użycie, **[H]** jednostki/przekształcenie |
| `6` | `17` | Sygnał analogowy, `raw*0.02 V` **w obsługiwanych trybach 1..6**; możliwa lambda | [X] zależność warunkowa, **[H]** interpretacja fizyczna |
| `7–8` | `65 BA` | Słowo Big Endian `N`; `RPM=10*round(2_244_000/N)` | [X][M] 5/5 |
| `9–10` | `A1 76` | Czas benzyny 1: 13-bit (`word&1FFF`) /187; stan ECU (`word>>13)&7` | [X][M] |
| `11–12` | `81 76` | Czas benzyny 2: 15-bit (`word&7FFF`) /187; najwyższy bit = czerwony stan ikony „G” | [X][M] 2/2 |
| `13–14` | `01 76` | Czas benzyny 3: pełne uint16 BE /187 | [X] |
| `15–16` | `01 76` | Czas benzyny 4: pełne uint16 BE /187 | [X] |
| `17–18` | `01 76` | Czas gazu 1: pełne uint16 BE /187 | [X] |
| `19–20` | `01 76` | Czas gazu 2: pełne uint16 BE /187 | [X] |
| `21–22` | `01 76` | Czas gazu 3: pełne uint16 BE /187 | [X] |
| `23–24` | `01 76` | Czas gazu 4: pełne uint16 BE /187 | [X] |
| `25–26` | `00 00` | Czas benzyny 5 | [X] (obsługa protokołu, fizyczny kanał opcjonalny) |
| `27–28` | `00 00` | Czas benzyny 6 | [X] |
| `29–30` | `00 00` | Czas benzyny 7 | [X] |
| `31–32` | `00 00` | Czas benzyny 8 | [X] |
| `33–34` | `00 00` | Czas gazu 5 | [X] |
| `35–36` | `00 00` | Czas gazu 6 | [X] |
| `37–38` | `00 00` | Czas gazu 7 | [X] |
| `39–40` | `00 00` | Czas gazu 8 | [X] |
| `41` | `17` | **MAP**, ciśnienie w kolektorze ssącym; w badanym wariancie `(raw/256 +0.00842)/0.242` bar | [X][M] 15/15 |
| `42` | `72` | Osobny bajt stanu w obiekcie (`[ebp+0xB5]`) | [X] storage; **[N]** semantyka |
| `43` | `00` | Bajt w obiekcie (`[ebp+0x18]`) | [X] storage; [N] |
| `44` | `00` | Bajt w obiekcie (`[ebp+0x19]`) | [X] storage; [N] |
| `45` | `FF` | Maska 8 bitów (`[ebp+0x1A]`), użycie w ośmiu kontrolkach **warunkowe** | [X] maska; [M] zmiany nie miały widocznego efektu w testowanym GUI; [N] znaczenie bitów |
| `46` | `01` | Flaga logiczna, `raw != 0` (`[ebp+0x1B]`) | [X]; [N] znaczenie |
| `47` | `88` | Dodatkowy czujnik temperatury, LUT; gdy dany wariant programu aktywny, inaczej odczyt alternatywnego parametru `0x1B` | [X] warunkowe; [H] nazwa czujnika |
| `48` | `00` | Bajt w obiekcie (`[ebp+0x16]`) | [X] storage; [N] |
| `49` | `02` | Bajt w obiekcie (`[ebp+0x1D]`) | [X] storage; [N] |

**Krytyczna poprawka względem wczesnych notatek:** bajty `25..40` to **wtryskiwacze 5–8**, a nie „szesnaście niezidentyfikowanych zer”. Kod w ramce `0B` obsługuje 8+8 kanałów, co nie oznacza 8 cylindrów w każdym sterowniku.

### 2.1 Mapa słów czasu bez błędnego maskowania

```python
PETROL_STARTS = [9,11,13,15,25,27,29,31]
GAS_STARTS    = [17,19,21,23,33,35,37,39]
PETROL_MASKS  = [0x1FFF,0x7FFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF]
GAS_MASKS     = [0xFFFF] * 8
word = (payload[offset] << 8) | payload[offset+1]
petrol_ms_i = (word & PETROL_MASKS[i]) / 187.0
gas_ms_i = word / 187.0
```

Tylko **pierwsze dwa słowa benzynowe** mają maski statusu. Pierwsze trzy bity `A0` nie są „stałą bazową” do matematycznego odejmowania; stan `5` jest rzeczywistym kodem pracy. Druga flaga `0x80` nie jest częścią czasu. Nie stosować `&0x1FFF` do gazu ani kanałów 3..8.

### 2.2 3-bitowy status instalacji (starsze 3 bity `payload[9]`)

Status: `(payload[9] >> 5) & 0x07`, albo równoważnie `(u16be(payload,9) >> 13)&0x07`.

| Kod | Bajt 9 w pomiarze (dolne 5 bitów stałe = `01`) | Wskazanie GUI, **zmierzone** | Interpretacja wewnętrzna |
|---:|---|---|---|
| 0 | `01` | `off` | [M] OFF |
| 1 | `21` | `zaplon` | [M] IGNITION_1, różnica od 2 nieustalona |
| 2 | `41` | `zaplon` | [M] IGNITION_2, różnica od 1 nieustalona |
| 3 | `61` | `benzyna cisnienie_na_czerwono` | [M] PETROL_PRESSURE_RED |
| 4 | `81` | `czekam` | [M] WAITING |
| 5 | `A1` | `gaz` | [M] GAS |
| 6 | `C1` | `benzyna cisnienie_nie_na_czerwono` | [M] PETROL_6 |
| 7 | `E1` | `benzyna cisnienie_nie_na_czerwono` | [M] PETROL_7 |

Kody 1/2 i 6/7 są **różne na przewodzie**, mimo identycznego tekstu. Nie scalać ich bez zbadania pozostałego zachowania programu.

### 2.3 Flaga ikony „G” (`payload[11] & 0x80`)

- `01 76`: wyświetlona ikonka „G” wskaźnika LPG jest **wyszarzona** [M].
- `81 76`: ikonka **czerwona** [M].
- W obu przypadkach licznik czasu to `0x0176=374`, czyli `2.0 ms`.
- Nie utożsamiać bitu z fizycznym elektrozaworem, przełącznikiem ani aktywnym wtryskiwaczem: fizyczna semantyka **niepotwierdzona**.

## 3. Przeliczniki — dokładne i warunkowe

### 3.1 Temperatury Tred, Tgas, opcjonalne T3 — tablica z EXE [X][M]

Oprogramowanie korzysta z 256 liczb **IEEE754 double (little-endian)** zapisanych w pliku EXE w kolejności **odwrotnej** do kodu RAW.

- Funkcja VA `0x00563D58`; tablica VA `0x005B8B44`, offset pliku `0x001B7744`.
- Indeks tabeli w EXE = `255 - RAW`; `temperature_c = doubles[255-raw]`.
- W `reference/temperature_lut.csv` zapisano **pełne 256 temperatur** z oryginalną precyzją `double` i od razu w układzie `RAW -> °C`. Dodatkowo `reference/tred_measurements_from_conversation.csv` zawiera **55** ręcznie zarejestrowanych kodów Tred (przepisanych z historii rozmowy), zgodnych z LUT **55/55** po zaokrągleniu do 0.1°C.
- Narzędzie `tools/extract_temperature_lut.py` odzyskuje tablicę z tego konkretnego EXE z walidacją SHA-256.
- GUI wyświetla najczęściej **jedno miejsce po przecinku**, np. `0x69→33.53631112→33.5°C`, `0x80→24.92070832→24.9°C`, `0x88→21.99276177→22.0°C`, `0xF0→-27.94831766→-27.9°C`.
- Wcześniejsze dopasowania NTC/Beta/Steinhart–Hart i hipoteza stałej kwantyzacji **są przestarzałe** jako implementacja dekodera. W EXE odnaleziono gotową tablicę. Charakterystyka jest zgodna z fizyką NTC, lecz GUI nie oblicza jej każdorazowo równaniem Beta w tej funkcji.
- Dokładne wyświetlenie `T3` z `payload[47]` jest **warunkowe**: pod `0x005826C2` w określonym wariancie wywoływana jest funkcja formatująca LUT; w drugim wariancie aplikacja pobiera osobny parametr/rejestr `0x1B`. Jeżeli UI nie reaguje na zmianę bajtu 47, nie uznawać tego za błąd LUT.

### 3.2 RPM [X][M: 5/5]

```python
N = (payload[7] << 8) | payload[8]  # BE
rpm = 0 if N == 0 else 10 * round(2_244_000 / N)
```

- Funkcja EXE `0x00563EA0`, składanie Big Endian w parserze `0x00564F4D..0x00564F5B`.
- Stałe w kodzie: `187.0`, `1000.0`, `120.0`, `10.0`; wzór odpowiada `10 * round((120 / (N/(187*1000))) / 10)`.
- Oryginalna konwersja używa x87 i funkcji zaokrąglającej do najbliższej liczby całkowitej; Python `round` ma zwykle zgodne *ties-to-even*, ale w wartościach granicznych należy sprawdzić wpływ precyzji x87.
- **Nie jest to** bezpośrednio `RPM=N`; odczyt stanowi okres lub wielkość do niego proporcjonalną. Wcześniejsze `22_500_000/N` było przybliżeniem i zostaje odrzucone.

| Bajty 7–8 | N | RPM z GUI | RPM wzór |
|---|---:|---:|---:|
| `65 BA` | 26042 | 860 | 860 |
| `65 9A` | 26010 | 860 | 860 |
| `65 DA` | 26074 | 860 | 860 |
| `45 BA` | 17850 | 1260 | 1260 |
| `85 BA` | 34234 | 660 | 660 |

### 3.3 MAP / kolektor ssący (`payload[41]`) [X][M: 15/15]

**Bieżący wariant czujnika w badanym ECU/GUI** (użytkownik potwierdził znaczenie fizyczne MAP):

```python
map_bar = max(0.0, (raw / 256.0 + 0.00842) / 0.242)
```

- Funkcja EXE `0x00563BC8`; wybór formuły zależy od ustawienia typu czujnika w aplikacji/ECU.
- Rozdzielczość: `1/(256*0.242)=~0.0161415289256 bar / LSB`.
- Dla `raw=0x17 (23)`, `P=0.406...bar` → **0,41 bar**.
- Dla innych konfiguracji EXE ma drugą gałąź `max(0,(raw/256 - 0.04)/0.369)`. Ta konfiguracja nie była zweryfikowana tym CSV.
- Nie mylić z „podciśnieniem” jako `P_atm - P_MAP` bez niezależnego pomiaru/referencji. Etykieta w oryginalnym GUI może używać „podciśnienie”, ale wzór produkuje bar.

| RAW | GUI bar | Wzór bar (do wyświetlenia zaokrąglony do 2 cyfr) |
|---|---:|---:|
| `00` | 0.03 | 0.03 |
| `08` | 0.16 | 0.16 |
| `10` | 0.29 | 0.29 |
| `17` | 0.41 | 0.41 |
| `20` | 0.55 | 0.55 |
| `30` | 0.81 | 0.81 |
| `40` | 1.07 | 1.07 |
| `50` | 1.33 | 1.33 |
| `60` | 1.58 | 1.58 |
| `80` | 2.10 | 2.10 |
| `A0` | 2.62 | 2.62 |
| `C0` | 3.13 | 3.13 |
| `E0` | 3.65 | 3.65 |
| `F0` | 3.91 | 3.91 |
| `FF` | 4.15 | 4.15 |

### 3.4 Ciśnienie gazu (`payload[1]`) [X] + częściowe [M]

**Tryb standardowy (`raw<0xFA`, a w konfiguracji czujnika `mode=1` również powyżej):**

```python
pressure_bar = max(0.0, (raw/256.0 - 0.04)/0.369)
```

- Funkcja EXE `0x00563950`. Stałe z programu: `5`, `256`, `0.04`, `0.00369`, `100` (czynnik `5` znosi się).
- `0x79=121` → `1.172510... bar` → GUI około **1.17**; `0xF9` → około **2.53**.
- W tym samym typie konfiguracji **`raw>=0xFA`** przełącza źródło na 16-bitowe parametry `R0D` i `R0F`:

```python
if R0F > R0D:
    pressure_bar = 0.0
else:
    pressure_bar = ((R0D-R0F)/256.0 + 0.00842)/0.242
```

- Historyczny emulator zwraca na `C0 40 0D 0D` wartość `40 00 8E` (`R0D=142`) i na `C0 40 0F 0F` wartość `40 00 1A` (`R0F=26`). Wtedy `pressure=1.9072107...bar` → `1.91 bar`, dokładnie jak GUI przy `0xFA..0xFF` [M] w tym teście.
- **Wniosek:** `0xFA..FF` w tej konfiguracji nie jest liniowym przedłużeniem RAW. Dawny empiryczny wzór `0.01058*raw-0.107` odrzucić jako implementację.
- „Sensor mode 1” to skrót naszej analizy warunku programu, **nie** publicznie potwierdzona nazwa ustawienia GUI.

### 3.5 Czas wtrysku i statusy [X][M]

Czas bez znaczników: `time_ms = N / 187.0`; funkcja EXE `0x00563DEC`, stała `187.0` pod VA `0x00563E0C`.

- `0x0176=374` → dokładnie `2.0 ms`.
- Dla benzyny 1: `0xA176 & 0x1FFF == 0x0176`, status `(0xA176>>13)&7 == 5`.
- Dla benzyny 2: `0x8176 & 0x7FFF == 0x0176`, ikonka-G = 1.
- Dla benzyny 3..8 i gazu 1..8: całe uint16/187 bez tych masek.
- Co do rzeczywistej maksymalnej liczby aktywnych kanałów oraz dopuszczalnych wartości czasów — zależne od ECU [H].

### 3.6 Napięcie i wejście analogowe

- Bajt 4: `U=(raw*5/256)/0.215`, dla `9B` około `14.0807 V` [X] wzór; nazwa „napięcie akumulatora/zasilania” [H].
- Bajt 5: poziom LPG [H], aktywne GUI może wymagać typu czujnika i tabeli skalowania — **nie stosować dowolnej liniowej funkcji**.
- Bajt 6: w pewnych trybach wejścia `U=raw*0.02 V`, dla `17` → `0.46 V` [X] warunkowy przelicznik; „lambda” [H]. Funkcja EXE `0x0057BEB0`.
- Bajt 45: bitfield 8 bitów. GUI nie reagowało w danym układzie pomiarowym [M]. W EXE `0x00582570` występuje **bramka możliwości/wariant sterownika**. Bez potwierdzenia przejścia przez tę bramkę nie nazywać tych bitów stanami 8 wtryskiwaczy.

## 4. Referencyjne wartości dla bazowej ramki `0B`

Po dekodowaniu bazowego payloadu:

| Pole | Oczekiwany odczyt |
|---|---|
| Ciśnienie gazu `79` | ~1.17 bar (gałąź standardowa) |
| Tred `69` | `33.53631112°C` → GUI `33.5°C` |
| Tgas `93` | `18.23821472°C` → GUI `18.2°C` w formacie 1dp |
| Bajt 4 `9B` | ~14.08067 V, nazwa zasilanie [H] |
| Bajt 5 `6B` | 107 RAW, nieznana fizyczna skala |
| Bajt 6 `17` | 23 RAW, `0.46 V` tylko po właściwej konfiguracji |
| RPM `65 BA` | **860 RPM** |
| Stan `(A1 >> 5)&7` | 5 = **GAZ** |
| Ikona G `(81&80)!=0` | **czerwona** |
| Benzyna 1..4 | `2.0, 2.0, 2.0, 2.0 ms` |
| Benzyna 5..8 | `0.0, 0.0, 0.0, 0.0 ms` |
| Gaz 1..4 | `2.0, 2.0, 2.0, 2.0 ms` |
| Gaz 5..8 | `0.0, 0.0, 0.0, 0.0 ms` |
| MAP `17` | `0.406... bar` → **0.41 bar** |
| Temp3 `88` | `21.99276177°C` → `22.0°C`, tylko przy aktywnym wariancie GUI |
| 42–46 / 48–49 | Surowe bajty; brak potwierdzonych nazw fizycznych |

Testy offline w `tests/test_protocol.py` wykonują regresję dla rzeczywistych CSV i sprawdzają kompletność katalogu ramek.

## 5. Katalog wszystkich znanych zapytań i statycznych odpowiedzi

**Źródło:** historyczny słownik `responses` oryginalnego emulatora dostarczony w rozmowie, nie pełny trace ECU. Łącznie **83 wpisy** (82 z odpowiedzią i jeden specjalny bez odpowiedzi). Dla każdego wpisu archiwum zawiera:

- `reference/known_commands.json`: **pełny** `request_hex`, `payload_hex`, długości, gotową ramkę `wire_reply_hex` i wyliczoną sumę `payload_checksum_hex`. Jest **źródłem prawdy** do zaimplementowania emulatora lub fixture testów. Wpisy z `provenance="emulator_snapshot"` nie powinny być automatycznie wysyłane do żywego ECU.
- `tools/make_catalog.py`: generator katalogu z transkrypcji historycznego słownika (dla długich, ręcznie przepisanych odpowiedzi należy porównać z lokalnym oryginalnym `emulator.py`, jeśli jest dostępny).
- `H`: `C0 07 18 DF` dodano eksperymentalnie **bez odczytu z prawdziwego urządzenia**. `C2 01 01 6B 87`: wpis emulacyjny bez odpowiedzi.

Kolumna „Bajtów na przewodzie” obejmuje marker `D0` i CRC, **nie obejmuje** zapytania. Długie payloady są w tabeli skrócone do prefiksu, natomiast **pełne bajty są w JSON**.

| # | Zapytanie (HEX) | Payload odpowiedzi (HEX; bez `D0`, CRC) | Bajtów payloadu | Bajtów na przewodzie | CRC odp. | Pochodzenie |
|---:|---|---|---:|---:|---|---|
| 1 | `C0 01 01 C2` | `01 30` | 2 | 4 | `01` | E |
| 2 | `C0 02 02 C4` | `02 31 36 3A 32 36 3A 31 31 4D 61 72 20 31 32 20 32 30 31 32` | 20 | 22 | `FF` | E |
| 3 | `C0 53 00 13` | `53 4A` | 2 | 4 | `6D` | E |
| 4 | `C0 53 01 14` | `53 04` | 2 | 4 | `27` | E |
| 5 | `C0 53 02 15` | `53 00` | 2 | 4 | `23` | E |
| 6 | `C0 53 03 16` | `53 04` | 2 | 4 | `27` | E |
| 7 | `C0 53 04 17` | `53 02` | 2 | 4 | `25` | E |
| 8 | `C0 53 05 18` | `53 00` | 2 | 4 | `23` | E |
| 9 | `C0 53 06 19` | `53 FF` | 2 | 4 | `22` | E |
| 10 | `C0 40 1A 1A` | `40 00 5B` | 3 | 5 | `6B` | E |
| 11 | `C0 3F 00 FF` | `3F F0` | 2 | 4 | `FF` | E |
| 12 | `C0 07 22 E9` | `07 22 9C` | 3 | 5 | `95` | E |
| 13 | `C0 07 21 E8` | `07 21 03` | 3 | 5 | `FB` | E |
| 14 | `C0 40 1C 1C` | `40 00 01` | 3 | 5 | `11` | E |
| 15 | `C0 D8 D8 70` | `D8 03 0F 80 0A 72 DB 24 1E 07 CD 00 82 01 F9 01 00 1C 01 F4 01 F4 04 39 96 03` | 26 | 28 | `00` | E |
| 16 | `C0 35 00 F5` | `35 C8 8F 5C 28` | 5 | 7 | `E0` | E |
| 17 | `C0 36 00 F6` | `36 40 4E 5B 68` | 5 | 7 | `57` | E |
| 18 | `C0 07 23 EA` | `07 23 C3` | 3 | 5 | `BD` | E |
| 19 | `C0 51 0B 1C` | `51 0B 01 90` | 4 | 6 | `BD` | E |
| 20 | `C0 51 0C 1D` | `51 0C 1F 40` | 4 | 6 | `8C` | E |
| 21 | `C0 07 1F E6` | `07 1F 20` | 3 | 5 | `16` | E |
| 22 | `C0 07 1D E4` | `07 1D 5A` | 3 | 5 | `4E` | E |
| 23 | `C0 51 0E 1F` | `51 0E 00 07` | 4 | 6 | `36` | E |
| 24 | `C0 07 20 E7` | `07 20 01` | 3 | 5 | `F8` | E |
| 25 | `C0 07 25 EC` | `07 25 41` | 3 | 5 | `3D` | E |
| 26 | `C0 12 4A 1C` | `12 4A 00 03` | 4 | 6 | `2F` | E |
| 27 | `C0 07 2A F1` | `07 2A 9C` | 3 | 5 | `9D` | E |
| 28 | `C0 07 28 EF` | `07 28 01` | 3 | 5 | `00` | E |
| 29 | `C0 51 10 21` | `51 10 02 31` | 4 | 6 | `64` | E |
| 30 | `C0 07 26 ED` | `07 26 4D` | 3 | 5 | `4A` | E |
| 31 | `C0 07 27 EE` | `07 27 32` | 3 | 5 | `30` | E |
| 32 | `C0 07 13 DA` | `07 13 03` | 3 | 5 | `ED` | E |
| 33 | `C0 07 2B F2` | `07 2B 32` | 3 | 5 | `34` | E |
| 34 | `C0 07 2C F3` | `07 2C E2` | 3 | 5 | `E5` | E |
| 35 | `C0 93 00 53` | `93 00 00 00 00 00 00 00 00` | 9 | 11 | `63` | E |
| 36 | `C0 07 2D F4` | `07 2D 02` | 3 | 5 | `06` | E |
| 37 | `C0 07 2E F5` | `07 2E 02` | 3 | 5 | `07` | E |
| 38 | `C0 51 12 23` | `51 12 09 C4` | 4 | 6 | `00` | E |
| 39 | `C0 51 13 24` | `51 13 02 EC` | 4 | 6 | `22` | E |
| 40 | `C0 07 31 F8` | `07 31 00` | 3 | 5 | `08` | E |
| 41 | `C0 07 32 F9` | `07 32 00` | 3 | 5 | `09` | E |
| 42 | `C0 51 16 27` | `51 16 12 43` | 4 | 6 | `8C` | E |
| 43 | `C0 56 00 16` | `56 3C 27 10` | 4 | 6 | `99` | E |
| 44 | `C0 54 00 14` | `54 3B C5 15 07 7E 91 92 D4 3B C5 15 07 7E 8B A4 34 3B C5 15 07 7E 8C A3 89 3B C5 15 0 …` (pełne w JSON) | 42 | 44 | `11` | E |
| 45 | `C0 07 34 FB` | `07 34 00` | 3 | 5 | `0B` | E |
| 46 | `C0 07 2F F6` | `07 2F 00` | 3 | 5 | `06` | E |
| 47 | `C0 51 11 22` | `51 11 00 00` | 4 | 6 | `32` | E |
| 48 | `C0 51 18 29` | `51 18 01 F4` | 4 | 6 | `2E` | E |
| 49 | `C0 51 19 2A` | `51 19 FF FF` | 4 | 6 | `38` | E |
| 50 | `C0 51 14 25` | `51 14 01 76` | 4 | 6 | `AC` | E |
| 51 | `C0 51 15 26` | `51 15 0E 9C` | 4 | 6 | `E0` | E |
| 52 | `C0 40 28 28` | `40 00 00` | 3 | 5 | `10` | E |
| 53 | `C0 07 18 DF` | `07 18 00` | 3 | 5 | `EF` | H |
| 54 | `C0 0B 0B D6` | `0B 79 69 93 9B 6B 17 65 BA A1 76 81 76 01 76 01 76 01 76 01 76 01 76 01 76 00 00 00 0 …` (pełne w JSON) | 50 | 52 | `77` | E |
| 55 | `C0 40 29 29` | `40 00 05` | 3 | 5 | `15` | E |
| 56 | `C0 64 A5 C9` | `64 03 6A 0D 60 7A 6B 18 B8 00 00 00` | 12 | 14 | `C3` | E |
| 57 | `C0 40 0D 0D` | `40 00 8E` | 3 | 5 | `9E` | E |
| 58 | `C0 40 0F 0F` | `40 00 1A` | 3 | 5 | `2A` | E |
| 59 | `C0 07 35 FC` | `07 35 3C` | 3 | 5 | `48` | E |
| 60 | `C0 10 10 E0` | `10 04 19 00 F3 03 A7 07 4E 0A F5 0E 9C 10 00 10 00 10 00 10 00` | 21 | 23 | `D8` | E |
| 61 | `C0 13 13 E6` | `13 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00` | 17 | 19 | `E3` | E |
| 62 | `C0 95 00 55` | `95 00 00 00 F3 01 B6 03 1A 01 F9 03 1B 02 D9 04 8B 04 4F 06 13 05 8D 07 1F 06 A6 08 0 …` (pełne w JSON) | 41 | 43 | `AD` | E |
| 63 | `C0 95 01 56` | `95 FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF F …` (pełne w JSON) | 40 | 42 | `3E` | E |
| 64 | `C0 07 1E E5` | `07 1E 68` | 3 | 5 | `5D` | E |
| 65 | `C0 D4 00 94` | `D4 00 00 00 F3 01 B6 03 1A 01 F9 03 1B 02 D9 04 8B 04 4F 06 13 05 8D 07 1F 06 A6 08 0 …` (pełne w JSON) | 41 | 43 | `EC` | E |
| 66 | `C0 D4 01 95` | `D4 FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF F …` (pełne w JSON) | 40 | 42 | `7D` | E |
| 67 | `C0 12 01 D3` | `12 01 04 19` | 4 | 6 | `00` | E |
| 68 | `C0 12 02 D4` | `12 02 00 F3` | 4 | 6 | `D7` | E |
| 69 | `C0 12 0B DD` | `12 0B 04 00` | 4 | 6 | `F1` | E |
| 70 | `C0 12 0C DE` | `12 0C 00 00` | 4 | 6 | `EE` | E |
| 71 | `C0 12 0D DF` | `12 0D 00 00` | 4 | 6 | `EF` | E |
| 72 | `C0 C0 00 80` | `C0 01 F9 02 EC 02 FF 04 E5` | 9 | 11 | `62` | E |
| 73 | `C0 D0 00 90` | `D0 66 50 3C 2F 21 17 0F 09 00 FA F6 F4 F0 EE EA E8 E6 E4 E1 E0` | 21 | 23 | `30` | E |
| 74 | `C0 D2 00 92` | `D2 F2 F4 F6 F7 F8 FB FC FE FF 00 01 01 01 02 02 02 03 03 04 05` | 21 | 23 | `79` | E |
| 75 | `C0 DB 00 9B` | `DB 00 00 00 00 00 00 01 02 03 04 05 08 0A 0B 0C 0D 0E 0F 10 11` | 21 | 23 | `2E` | E |
| 76 | `C0 97 00 57` | `97 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00` | 21 | 23 | `67` | E |
| 77 | `C0 60 A5 C5` | `60 0A 14 1E 28 32 3C 46 50 5A 64 01 F4 03 E8 05 DC 07 D0 09 C4 0B B8 0D AC 0F A0 11 94 13 88` | 31 | 33 | `26` | E |
| 78 | `C0 61 10 31` | `61 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 F …` (pełne w JSON) | 50 | 52 | `50` | E |
| 79 | `C0 61 20 41` | `61 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0 …` (pełne w JSON) | 50 | 52 | `51` | E |
| 80 | `C0 61 11 32` | `61 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0 …` (pełne w JSON) | 50 | 52 | `42` | E |
| 81 | `C0 61 21 42` | `61 21 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0 …` (pełne w JSON) | 50 | 52 | `52` | E |
| 82 | `C0 12 68 3A` | `12 68 00 00` | 4 | 6 | `4A` | E |
| 83 | `C2 01 01 6B 87` | — (brak odpowiedzi) | 0 | 0 | `—` | E / brak odpowiedzi |

### 5.1 Komenda `C0 3A 00 FA` — **NIE** dodawać do potwierdzonego katalogu

W analizie jednej gałęzi kodu Delphi zidentyfikowano możliwe generowanie zapytania `C0 3A 00 FA` [X/H]. Użytkownik włączył `trace all` w emulatorze i **nie zaobserwował** tej sekwencji [M, wynik negatywny w badanej sesji]. Nie znamy poprawnej odpowiedzi. Traktować jako **niesprawdzony trop zależny od konfiguracji**, a nie polecenie do bezpiecznego wysyłania do ECU.

### 5.2 Identyfikatory i potencjalne klasy komend

W historycznym katalogu występują m.in. rodziny: `01`, `02`, `07 xx`, `0B`, `10`, `12 xx`, `13`, `35`, `36`, `3F`, `40 xx`, `51 xx`, `53 xx`, `54`, `56`, `60`, `61`, `64`, `93`, `95`, `97`, `C0`, `D0`, `D2`, `D4`, `D8`, `DB`. Ich **semantyka, read/write, sekcje map pamięci i parametrów kalibracji nie są w pełni potwierdzone**. Część dużych odpowiedzi wygląda jak bloki map kalibracyjnych, lecz **nie oznacza to bezpiecznych operacji odczytu**.

### 5.3 Niezbadane aspekty warstwy komunikacji

- Komendy inicjalizacji identyfikujące model, wersję, liczbę kanałów, tryb czujników i przejścia stanu.
- Czy nadawanie odbywa się półdupleksowo i jakie są wymagane odstępy/opóźnienia; stary emulator wysyłał `ser.write(b"\xD0")`, a następnie `ser.write(payload+CRC)`, ale port szeregowy nie gwarantuje granicy między wywołaniami.
- Timeouty, retry, możliwości sesji, komendy zapisu i odpowiedzi błędów oraz inne warianty sumy kontrolnej.
- Obsługa asynchronicznych ramek, odpowiedzi niezwiązanych z ostatnim zapytaniem i uzgodnienia długości przy nieznanych komendach.
- Pełna zgodność długości i zawartości długich odpowiedzi w historycznej transkrypcji z plikiem użytkownika `emulator.py` (nie był załączony jako plik w tym przekazaniu).

**Nie implementuj aktywnego zapisu do ECU na podstawie samych odpowiedzi emulatora.**
