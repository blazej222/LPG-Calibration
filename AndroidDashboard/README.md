# GAS ECU — dashboard Diego G3 dla Androida 10

Aplikacja w `app/` obsługuje kabel USB–UART w trybie USB host, inicjalizację sesji Diego G3 i cykliczny odczyt `C0 0B 0B D6`. Dashboard pokazuje RPM, temperatury reduktora/gazu, ciśnienie gazu, MAP, napięcie jako pole o niepotwierdzonej nazwie fizycznej, stan instalacji oraz czasy wtrysku 8+8 kanałów. Kanały opcjonalne pozostają widoczne; zera nie oznaczają wykrycia liczby cylindrów.

## Uruchomienie na K706

1. Zbuduj i zainstaluj `app/build/outputs/apk/debug/app-debug.apk` (Android Studio lub polecenia poniżej).
2. Podłącz kabel LPG do portu USB host radia.
3. Otwórz **Ustawienia → Urządzenie USB**, wybierz urządzenie i port. Lista zawiera nazwę sterownika, VID/PID i ścieżkę USB. Nieobsługiwane urządzenia są widoczne, ale nie można ich wybrać.
4. Wybierz **Połącz** i zaakceptuj systemowe okno dostępu do USB. Inicjalizacja trwa kilka sekund; jej postęp jest widoczny.
5. Wróć na **Dashboard**. Domyślny minimalny odstęp między odczytami to 200 ms. Ustawienia profilu czujników można zmieniać po rozłączeniu.

Wybrane VID/PID/port i ustawienia są zapamiętywane. Jeśli podłączono kilka identycznych adapterów, po ponownym wykryciu potrzebny jest jednoznaczny wybór. Nie ma automatycznego łączenia po podłączeniu kabla. Ekran pozostaje włączony podczas połączenia. Przejście aplikacji w tło zatrzymuje transmisję; obrót ekranu zachowuje sesję. Wróciwszy do aplikacji, wybierz **Połącz**.

## Przebieg komunikacji i źródła

- **Transport:** 9600, 8 bitów danych, brak parzystości, 1 stop bit; DTR/RTS domyślnie włączone. Te ustawienia występują w `../GasECU Reverse Engineering/data/PORTMAN.LOG`, wpisy 10–15. Nie włączamy RTS/CTS ani XON/XOFF. Sterownik USB pochodzi z [usb-serial-for-android 3.11.0](https://github.com/mik3y/usb-serial-for-android/tree/3.11.0).
- **Start:** dokładne 77 żądań przed pierwszym `0B` z `PORTMAN.LOG`, linie 22–246 (numery linii pliku, nie kolumna ID). Odpowiedzi są pobierane ze sterownika, weryfikowane i konsumowane po kolei. Zwracane wartości konfiguracji pozostają RAW; aplikacja nie wywodzi z nich nieznanych trybów czujników.
- **C2:** cztery zaobserwowane `C2 01 01 6B 87`; oryginalna aplikacja czekała ok. 450 ms i otrzymywała timeout. Klient powtarza komunikat i odstęp, nie oczekuje ramki `D0`. Semantyka tej specjalnej wiadomości pozostaje **[H]**. Odtworzenie przechwyconego startu nie dowodzi, że każda jego wiadomość jest odczytem. Nie dodano poleceń zapisu map, kalibracji, przełączania paliwa ani sterowania wyjściami.
- **Odczyt:** głównie `0B`, 50 bajtów payloadu / 52 bajty na przewodzie. Gdy RAW ciśnienia wynosi `FA..FF` w wybranym wariancie, pobieramy aktualne rejestry `40/0D` i `40/0F`. Brak tych odpowiedzi daje nieustalone ciśnienie, bez podstawiania stałych emulatora.
- **Opcja zgodności:** „Pełny cykl odczytu producenta” dodaje po `0B` zapytania `40/29`, `64/A5`, `40/28`, `53/00`, `40/0D`, `40/0F`, `3F/00`, występujące w powtarzającym się cyklu logów. Ich payloady nie zastępują danych `0B` na dashboardzie. Opcja domyślnie wyłączona.
- **Ramkowanie:** jeden aktywny request; strumień buforowany do oczekiwanej długości, kontrola sumy `(sum(payload)-0x30)&255`, ID komendy i echa argumentu tam, gdzie występuje. Bajty `C0`/`D0` w payloadzie nie rozcinają ramki. Trzy kolejne timeouty odczytu zatrzymują połączenie; przed ponowieniem odbiór jest opróżniany do przerwy. Protokół nie ma numeru transakcji, więc spóźnione odpowiedzi tej samej komendy nie mają dodatkowego identyfikatora.

Zawartość długich odpowiedzi w logach Portmon jest często **ucięta do 21 widocznych bajtów**, mimo większego pola `Length`. Długości można wykorzystać, ale takich wpisów nie można traktować jako pełnych fixture'ów `0B` ani jako niezależnego potwierdzenia ich sumy. Pełne odpowiedzi użyte w testach inicjalizacji pochodzą z oryginalnego `emulator.py` **[E]**; długości wszystkich 77 wpisów są zgodne z przechwyconym startem.

## Dekodowanie i niepewność

Implementacja przenosi potwierdzone algorytmy z `../DiegoG3_Codex_handoff/src/diego_protocol.py`:

- RPM: `10 * round(2244000/N)`, zero dla `N=0`, zaokrąglanie do parzystej **[X/M]**.
- Temperatury: niezmieniona, 256-elementowa LUT z EXE **[X]**, Tred zgodne z 55 pomiarami **[M]**.
- Wtryski: BE `/187.0`; maski tylko dla benzyny 1 (`1FFF`) i 2 (`7FFF`).
- Stan: osiem osobnych kodów; flaga czerwonej ikony G osobno.
- Profil domyślny: MAP „Z pomiarów” oraz ciśnienie „FA → 0D/0F”, odpowiadające instalacji z handoffu. To **jawny wybór profilu**, nie automatyczny odczyt konfiguracji ECU. „Nieznany” wyłącza przeliczanie danego ciśnienia; alternatywny MAP pochodzi z EXE, bez pomiarowego potwierdzenia tego wariantu.
- Poziom gazu: RAW, bez wymyślonych procentów. Analog i dodatkowa temperatura domyślnie RAW, z opcją jawnego wskazania gałęzi. Nazwy fizyczne bajtów 4/6/47 pozostają hipotezami. Pola 42–46 i 48–49 są dostępne jako RAW w diagnostyce.

Modele `Reading` zachowują `raw`, `decoded`, `unit`, `confidence`, `configMode`, `source`; pełny payload pozostaje w `Telemetry`. Dokładne wartości nie są zaokrąglane przed UI. Po timeoutach lub rozłączeniu dashboard pokazuje kreski i „DANE NIEAKTUALNE”; ostatni payload w diagnostyce pozostaje dostępny do analizy.

## Logi i sprawdzenie ze sprzętem

Ustawienia zawierają ostatnie fragmenty TX/RX i przycisk **Udostępnij log TX/RX**. Pliki są prywatne dla aplikacji, maks. 4 MiB na sesję, ostatnie pięć sesji. Log zapisuje czas monotoniczny, kierunek i wszystkie odebrane fragmenty, także złą sumę. Limit zapisu nie zatrzymuje odczytu ECU.

Weryfikacja sprzętowa wymaga K706, kabla i ECU: wybór USB → zgoda → koniec inicjalizacji → stabilny licznik ramek → porównanie wskazań z programem producenta → odłączenie kabla i ponowne połączenie. Jeśli inicjalizacja zatrzyma się, udostępniony log wskaże konkretną komendę i ostatnią odpowiedź. W testach offline nie jest otwierany rzeczywisty port ECU.

## Budowanie i testy

Wymagane lokalne SDK/JDK są takie jak w template: SDK 37, JDK 25, Gradle Wrapper 9.6.0. `local.properties` powinien wskazywać lokalne SDK. Minimalna wersja urządzenia: Android 10 / API 29.

```powershell
cd AndroidDashboard
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug :app:lintDebug
```

`DiegoProtocolTest` sprawdza komplet pomiarów CSV, LUT, maski, gałęzie ciśnienia, unknown modes, fragmentację, sumy, odzyskiwanie synchronizacji, timeouty i cały start na odpowiedziach historycznego emulatora. Oryginalne fixture'y nie są modyfikowane.

Ponowne wygenerowanie LUT, sekwencji i kopii fixture'ów z oryginalnych danych:

```powershell
python AndroidDashboard/tools/prepare_reference.py
```

Generator sprawdza również długości startowych odpowiedzi względem oryginalnego emulatora. Pełny katalog handoffu zawiera kilka innych długich odpowiedzi różniących się od oryginału; nie są one wykorzystywane w tej inicjalizacji. Szczegóły w `../DiegoG3_Codex_handoff/FINDINGS.md`.

## Wynik weryfikacji — 2026-10-14

- `:app:testDebugUnitTest`, `:app:assembleDebug` i `:app:assembleDebugAndroidTest`: zakończone poprawnie. 15 testów protokołu oraz test przykładowy template'u bez błędów.
- Testy instrumentacyjne uruchomione bezpośrednio przez ADB na istniejącym emulatorze API 37: **3/3 OK**, w tym dwa testy dashboardu/ustawień bez urządzenia USB. Podgląd obu ekranów sprawdzony przy 1024×600, 160 dpi. Testy Espresso aktualizowano do 3.7.0 / AndroidX JUnit 1.3.0, ponieważ stare wersje template'u wywoływały usunięte API InputManager emulatora.
- Referencyjne testy Python handoffu: 19 przypadków, 18 OK, 1 pominięty z powodu braku opcjonalnego EXE.
- **Lint nie został ukończony:** pobieranie `intellij-core-32.4.1.jar` i `kotlin-compiler-32.4.1.jar` z `dl.google.com` kończyło się timeoutami. Ponowienie zatrzymano po kolejnym braku postępu pobierania. Nie jest to wynik analizy lintu; kontrolę trzeba powtórzyć po udostępnieniu tych zależności.
- Nie testowano fizycznego kabla, okna zgody USB na K706 ani odpowiedzi rzeczywistego ECU. Przeniesione przeliczniki mają dowody [X/M] opisane powyżej; skuteczność odtwarzanej inicjalizacji na danym urządzeniu wymaga sprawdzenia ze sprzętem.
