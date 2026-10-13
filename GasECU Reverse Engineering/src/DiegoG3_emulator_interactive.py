"""Interaktywny emulator DiegoG3 z kodowaniem parametrów w jednostkach fizycznych.

Nie zastępuje słownika pozostałych odpowiedzi: importuje go z oryginalnego
emulator.py (funkcje process_data i calculate_outgoing_crc).

Wymagane obok pliku:
  DiegoG3_decoder.py
  DiegoG3_temperature_table.csv

Uruchomienie:
  python DiegoG3_emulator_interactive.py --source emulator.py --port COM18

Wpisz 'help' w konsoli, żeby zobaczyć wszystkie polecenia.
Wersja przeznaczona do laboratoryjnej emulacji, nie do sterowania instalacją LPG.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.util
from pathlib import Path
import threading
import time

import serial

from DiegoG3_decoder import (
    decode_temperature,
    decode_pressure,
    decode_rpm,
    experimental_injection_time_from_frame,
)

REQUEST_0B = bytes.fromhex('C0 0B 0B D6')
SPECIAL_NO_REPLY = bytes.fromhex('C2 01 01 6B 87')
PAYLOAD_SIZE = 50
REG_REQUESTS = {
    bytes.fromhex('C0 40 0D 0D'): 0x0D,
    bytes.fromhex('C0 40 0F 0F'): 0x0F,
}


def load_original_program(path: str):
    path_obj = Path(path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(
            f'Nie znaleziono pliku {path_obj}. Zachowaj stary emulator jako emulator.py'
        )
    spec = importlib.util.spec_from_file_location('diego_original_emulator', path_obj)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Nie mogę załadować: {path_obj}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('process_data', 'calculate_outgoing_crc'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'Brak funkcji {name} w {path_obj}')
    return module


def parse_float(s: str) -> float:
    value = float(s.replace(',', '.'))
    if not (-float('inf') < value < float('inf')):
        raise ValueError('Wartość musi być liczbą skończoną')
    return value


def parse_hex(s: str) -> int:
    return int(s.removeprefix('0x').removeprefix('0X'), 16)


def find_temperature_byte(celsius: float, previous: int) -> int:
    """Odwrócenie prawdziwej tablicy LUT: najbliższa osiągalna temperatura."""
    candidates = range(256)
    lower = min(decode_temperature(x) for x in candidates)
    upper = max(decode_temperature(x) for x in range(256))
    if not lower <= celsius <= upper:
        raise ValueError(f'Temperatura poza zakresem LUT: {lower:.1f}..{upper:.1f}°C')
    return min(
        candidates,
        key=lambda x: (abs(decode_temperature(x) - celsius), abs(x - previous)),
    )


def find_pressure_byte(bar: float, previous: int) -> int:
    """Odwrócenie standardowego trybu dla bajtów 00..F9 (bez 0xFA..FF)."""
    if bar < 0:
        raise ValueError('Ciśnienie nie może być ujemne')
    maximum = decode_pressure(0xF9)
    if bar > round(maximum, 2) + 1e-9:
        raise ValueError(
            f'Standardowa gałąź obsługuje najwyżej {maximum:.3f} bar. '
            'Dla RAW FA..FF konieczne są dodatkowe rejestry 0D/0F.'
        )
    return min(range(0xFA), key=lambda x: (
        abs(decode_pressure(x) - bar), abs(x - previous)
    ))


def find_rpm_word(rpm: float, previous: int) -> int:
    """Dobiera uint16 z uwzględnieniem fabrycznego zaokrąglenia do 10 RPM.

    Poszukujemy wartości dającej najbliższy wynik. Przy wielu trafieniach
    preferujemy dotychczasowy kod RAW, by nie zmieniać go bez potrzeby.
    """
    if rpm < 0:
        raise ValueError('RPM nie może być ujemne')
    if rpm == 0:
        return 0
    if rpm > decode_rpm(1):
        raise ValueError('RPM powyżej największej wartości reprezentowalnej')
    if rpm < decode_rpm(0xFFFF):
        raise ValueError(
            f'Najmniejszy niezerowy wynik kodera to {decode_rpm(0xFFFF)} RPM; '
            'alternatywą jest 0 RPM.'
        )
    # 65 536 prób to niewielki koszt i zapewnia prawidłowy wybór także
    # w pobliżu dyskontynuacji zaokrągleń i ograniczeń uint16.
    return min(range(1, 65536), key=lambda n: (
        abs(decode_rpm(n) - rpm), abs(n - previous)
    ))


class FrameState:
    def __init__(self, payload: bytes, *, log_path: Path, original_registers=None):
        if len(payload) != PAYLOAD_SIZE or payload[0] != 0x0B:
            raise ValueError(
                f'Zły payload 0B: długość={len(payload)}, '
                f'pierwszy bajt={payload[:1].hex()} (oczekiwano 50 bajtów, 0B)'
            )
        self.original = bytes(payload)
        self._payload = bytearray(payload)
        self._lock = threading.Lock()
        self.scan: tuple[str, list[str]] | None = None
        self.scan_pos = 0
        self.trace = 'off'
        self.log_path = log_path
        self.original_registers = dict(original_registers or {0x0D: 0x008E, 0x0F: 0x001A})
        self.registers = dict(self.original_registers)
        self.register_overrides = set()

    def get(self) -> bytes:
        with self._lock:
            return bytes(self._payload)

    def _update(self, start: int, raw: bytes, *, clear_scan: bool = True):
        if start < 1 or start + len(raw) > PAYLOAD_SIZE:
            raise ValueError('Nieprawidłowy offset/zakres (musi mieścić się w bajtach 1..49)')
        with self._lock:
            self._payload[start:start + len(raw)] = raw
            if clear_scan:
                self.scan = None

    def set_raw(self, offset: int, raw: int, width: int, *, clear_scan: bool = True):
        if width not in (1, 2):
            raise ValueError('Obsługiwane są pola 8 i 16 bitów')
        if not (0 <= raw < 1 << (8 * width)):
            raise ValueError(f'RAW musi mieścić się w {8 * width} bitach')
        self._update(offset, raw.to_bytes(width, 'big'), clear_scan=clear_scan)

    def set_physical(self, field: str, value: float, *, clear_scan: bool = True):
        f = field.lower()
        payload = self.get()
        if f in ('tred', 'tgas'):
            i = 2 if f == 'tred' else 3
            raw = find_temperature_byte(value, payload[i])
            self.set_raw(i, raw, 1, clear_scan=clear_scan)
            print(f'{f}: żądano {value:.2f}°C -> RAW={raw:02X} -> '
                  f'osiągalne {decode_temperature(raw):.2f}°C')
        elif f in ('pressure-special', 'press-special'):
            self.set_pressure_special(value, clear_scan=clear_scan)
        elif f in ('pressure', 'press'):
            raw = find_pressure_byte(value, payload[1])
            self.set_raw(1, raw, 1, clear_scan=clear_scan)
            print(f'Ciśnienie: żądano {value:.3f} bar -> RAW={raw:02X} -> '
                  f'osiągalne {decode_pressure(raw):.3f} bar')
        elif f == 'rpm':
            previous = int.from_bytes(payload[7:9], 'big')
            raw = find_rpm_word(value, previous)
            self.set_raw(7, raw, 2, clear_scan=clear_scan)
            print(f'RPM: żądano {value:.1f} -> RAW={raw:04X} -> '
                  f'osiągalne {decode_rpm(raw)} RPM (mapowanie pola eksperymentalne)')
        elif f.startswith(('petrol', 'gas')):
            self.set_injection_by_field(f, value, clear_scan=clear_scan)
        else:
            raise ValueError('Pole nieznane. Dostępne: tred, tgas, pressure, rpm, '
                             'petrol1..4, gas1..4')

    def set_injection_by_field(self, field: str, ms: float, *, clear_scan: bool = True):
        if field.startswith('petrol'):
            fuel, index_text, base = 'benzyna', field[6:], 9
        elif field.startswith('gas'):
            fuel, index_text, base = 'gaz', field[3:], 17
        else:
            raise ValueError('Podaj petrol1..4 lub gas1..4')
        if index_text not in ('1', '2', '3', '4'):
            raise ValueError('Numer wtryskiwacza musi wynosić 1..4')
        ch = int(index_text)
        if ms < 0:
            raise ValueError('Czas wtrysku nie może być ujemny')
        low_bits = round(ms * 187)
        if low_bits > 0x1FFF:
            raise ValueError('Wartość nie mieści się w 13 dolnych bitach; '
                             'nie chcemy nadpisać niepotwierdzonych flag')
        offset = base + 2 * (ch - 1)
        original = int.from_bytes(self.get()[offset:offset + 2], 'big')
        flags = original & 0xE000
        raw = flags | low_bits
        self.set_raw(offset, raw, 2, clear_scan=clear_scan)
        print(f'{fuel} {ch}: żądano {ms:.3f} ms -> RAW={raw:04X}, '
              f'wyliczone {low_bits / 187:.3f} ms; '
              f'flagi {flags:04X} zachowane [HIPOTEZA flag]')

    def get_register_override(self, request: bytes):
        register = REG_REQUESTS.get(request)
        if register is None:
            return None
        with self._lock:
            if register not in self.register_overrides:
                return None
            return b'\x40' + self.registers[register].to_bytes(2, 'big')

    def set_register(self, register: int, word: int, *, clear_scan=True):
        if register not in (0x0D, 0x0F):
            raise ValueError('Można nadpisywać tylko rejestry 0D i 0F')
        if not 0 <= word <= 65535:
            raise ValueError('Wartość rejestru musi być uint16')
        with self._lock:
            self.registers[register] = word
            self.register_overrides.add(register)
            if clear_scan:
                self.scan = None
        print(f'Rejestr {register:02X} = {word:04X} ({word})')

    def set_pressure_special(self, bar: float, *, clear_scan=True):
        """Emuluje gałąź RAW>=FA + rejestry 0D/0F (sensor_mode != 1).

        Odtworzony wzór: P=((r0d-r0f)/256 + 0.00842)/0.242.
        Rejestr r0f pozostaje niezmieniony, poza przypadkiem P=0.
        """
        if bar < 0:
            raise ValueError('Ciśnienie nie może być ujemne')
        if bar == 0:
            r0d, r0f = 0, 1
        else:
            difference = round((bar * 0.242 - 0.00842) * 256)
            if difference < 0:
                raise ValueError('Najmniejszy niezerowy wynik tej gałęzi to '
                                 f'{0.00842 / 0.242:.3f} bar. Możesz użyć 0.')
            with self._lock:
                r0f = self.registers[0x0F]
            r0d = r0f + difference
            if r0d > 65535:
                raise ValueError('Wartość przekracza zakres rejestrów uint16')
        with self._lock:
            self._payload[1] = 0xFA
            self.registers[0x0D] = r0d
            self.registers[0x0F] = r0f
            self.register_overrides.update((0x0D, 0x0F))
            if clear_scan:
                self.scan = None
        achieved = decode_pressure(0xFA, sensor_mode=0,
                                   register_0d=r0d, register_0f=r0f)
        print(f'SPECJALNE ciśnienie: żądano {bar:.3f} bar -> '
              f'0B[1]=FA, R0D={r0d:04X}, R0F={r0f:04X} -> '
              f'{achieved:.3f} bar [zakładamy sensor_mode != 1]')

    def reset(self):
        with self._lock:
            self._payload[:] = self.original
            self.registers = dict(self.original_registers)
            self.register_overrides.clear()
            self.scan = None
        print('Przywrócono oryginalną odpowiedź 0B.')
        self.show()

    def show(self):
        d = self.get()
        with self._lock:
            r0d = self.registers[0x0D]
            r0f = self.registers[0x0F]
        print('\nAKTUALNA ODPOWIEDŹ 0B (50 bajtów, bez D0/CRC):')
        print(' '.join(f'{x:02X}' for x in d))
        try:
            if d[1] >= 0xFA:
                p = decode_pressure(d[1], sensor_mode=0,
                                    register_0d=r0d, register_0f=r0f)
                pressure = f'{p:.3f} bar (tryb specjalny, sensor_mode != 1)'
            else:
                pressure = f'{decode_pressure(d[1]):.3f} bar'
        except ValueError:
            pressure = 'TRYB SPECJALNY FA..FF (wymagane rejestry 0D/0F)'
        print(f'  Cisnienie [01] {d[1]:02X} = {pressure}')
        print(f'  Rejestry 0D={r0d:04X}, 0F={r0f:04X} '
              f'(odpowiedzi 40 {r0d>>8:02X} {r0d&255:02X} / '
              f'40 {r0f>>8:02X} {r0f&255:02X})')
        print(f'  Tred     [02] {d[2]:02X} = {decode_temperature(d[2]):.2f} °C')
        print(f'  Tgas     [03] {d[3]:02X} = {decode_temperature(d[3]):.2f} °C')
        raw_rpm = int.from_bytes(d[7:9], 'big')
        print(f'  RPM    [07:08] {raw_rpm:04X} = {decode_rpm(raw_rpm)} obr/min '
              '[przypisanie pola do potwierdzenia]')
        for i in range(4):
            braw = int.from_bytes(d[9 + 2*i:11 + 2*i], 'big')
            graw = int.from_bytes(d[17 + 2*i:19 + 2*i], 'big')
            print(f'  Wtrysk {i+1}: benzyna {braw:04X} ~'
                  f'{experimental_injection_time_from_frame(braw):.3f} ms, '
                  f'gaz {graw:04X} ~'
                  f'{experimental_injection_time_from_frame(graw):.3f} ms '
                  '[flagi niepotwierdzone]')
        print('  Unknown [04:06]:', d[4:7].hex(' ').upper())
        print('  Unknown [25:40]:', d[25:41].hex(' ').upper())
        print('  Unknown [41:49]:', d[41:50].hex(' ').upper())
        print()

    def start_scan(self, command: str, values: list[str]):
        if not values:
            raise ValueError('Brak wartości do skanowania')
        # Sprawdź wartości przed rozpoczęciem, ale nie modyfikuj finalnej ramki
        # jeśli przygotowanie skanu się nie powiedzie.
        for val in values:
            if command.startswith('raw8:') or command.startswith('raw16:'):
                offset = int(command.split(':')[1])
                width = 1 if command.startswith('raw8:') else 2
                rv = parse_hex(val)
                if not (1 <= offset and offset + width <= PAYLOAD_SIZE and
                        0 <= rv < 1 << (width * 8)):
                    raise ValueError('Nieprawidłowy offset lub RAW w skanowaniu')
            else:
                v = parse_float(val)
                if command in ('tred', 'tgas'):
                    arr = [decode_temperature(i) for i in range(256)]
                    if not min(arr) <= v <= max(arr):
                        raise ValueError('Temperatura poza zakresem LUT')
                elif command in ('pressure-special', 'press-special'):
                    if v < 0:
                        raise ValueError('Ciśnienie nie może być ujemne')
                elif command in ('pressure', 'press'):
                    if not 0 <= v <= round(decode_pressure(0xF9), 2):
                        raise ValueError('Ciśnienie poza standardowym zakresem RAW 00..F9')
                elif command == 'rpm':
                    if v < 0 or v > decode_rpm(1) or (0 < v < decode_rpm(0xFFFF)):
                        raise ValueError('RPM poza zakresem')
                elif command.startswith(('petrol', 'gas')):
                    if v < 0 or round(v * 187) > 0x1FFF:
                        raise ValueError('Czas nie mieści się w 13 bitach')
                else:
                    raise ValueError('Nieznane pole skanowania')
        self.scan = (command, values)
        self.scan_pos = 0
        self.apply_scan()

    def apply_scan(self):
        if self.scan is None:
            print('Brak aktywnego skanu. Użyj sweep albo scan/scan16.')
            return
        field, vals = self.scan
        val = vals[self.scan_pos]
        print(f'\n>>> WARIANT {self.scan_pos+1}/{len(vals)}: {field} = {val}')
        if field.startswith('raw8:'):
            self.set_raw(int(field.split(':')[1]), parse_hex(val), 1, clear_scan=False)
        elif field.startswith('raw16:'):
            self.set_raw(int(field.split(':')[1]), parse_hex(val), 2, clear_scan=False)
        else:
            self.set_physical(field, parse_float(val), clear_scan=False)
        self.show()

    def advance(self, delta: int):
        if self.scan is None:
            print('Najpierw uruchom sweep, scan lub scan16.')
            return
        self.scan_pos = (self.scan_pos + delta) % len(self.scan[1])
        self.apply_scan()

    def record(self, displayed: str, notes: str = ''):
        """Dodaj odczyt z fabrycznego GUI do pliku CSV (do dalszego RE)."""
        d = self.get()
        with self._lock:
            r0d, r0f = self.registers[0x0D], self.registers[0x0F]
        field, target, index = ('manual', '', '')
        if self.scan is not None:
            field, vals = self.scan
            target = vals[self.scan_pos]
            index = str(self.scan_pos + 1)
        row = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'field': field,
            'variant': index,
            'requested_value': target,
            'displayed_in_DiegoG3': displayed,
            'notes': notes,
            'pressure_raw': f'{d[1]:02X}',
            'reg_0d': f'{r0d:04X}',
            'reg_0f': f'{r0f:04X}',
            'tred_raw': f'{d[2]:02X}',
            'tgas_raw': f'{d[3]:02X}',
            'rpm_raw': d[7:9].hex().upper(),
            'payload_hex': d.hex(' ').upper(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.log_path.exists() and self.log_path.stat().st_size > 0
        with self.log_path.open('a', newline='', encoding='utf-8-sig' if not exists else 'utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f'Zapisano wynik do {self.log_path}')


HELP_TEXT = '''
KOMENDY (offsety dziesiętne, RAW w HEX, wartości fizyczne w DEC):
  show                              pokaż ramkę i zdekodowane wartości
  tred 35.0                         ustaw reduktor na ~35°C (fabryczna tablica)
  tgas 24.9                         ustaw gaz na ~24,9°C
  pressure 1.25                     ustaw ciśnienie na ~1,25 bar (RAW < FA)
  pressure-special 1.91             ustaw RAW=FA i rejestry 0D/0F (~1,91 bar)
  rpm 1500                          ustaw ~1500 obr./min
  inj b 1 2.50                      czas benzyny wtryskiwacza 1 = ~2,50 ms
  inj g 2 3.10                      czas gazu wtryskiwacza 2 = ~3,10 ms

  set 1 F9                          wymuś bajt RAW 0xF9, offset 1
  set16 7 65BA                      wymuś uint16 RAW 0x65BA, offset 7
  reg 0D 008E                       ustaw ręcznie 16-bitowy rejestr 0D
  reg 0F 001A                       ustaw ręcznie 16-bitowy rejestr 0F
  flags b 1 A0                      ustaw flagi starszego bajtu benzyny 1
                                     (A0 / 80 / 00 ...; eksperymentalne)

  sweep tred 20 25 30 35 40         skan temperatur w stopniach C
  sweep pressure 0.5 1.0 1.5 2.0    skan ciśnienia w barach
  sweep pressure-special 1 1.5 1.91  skan specjalnej gałęzi
  sweep rpm 800 900 1000 1500      skan obrotów
  sweep petrol1 0 1.5 2 3.5         skan czasu benzyny 1 (ms)
  sweep gas1 1 2 3 4               skan czasu gazu 1 (ms)
  scan 2 60 65 69 70               skan RAW bajtu 2
  scan16 7 65BA 659A 45BA          skan RAW dwóch bajtów RPM
  next / prev                       następny / poprzedni wariant skanu
  record 33.5 [komentarz]           zapisz wskazanie GUI do CSV
  stop                              zakończ skan, pozostaw ostatni wariant
  trace on / off / all              loguj odpowiedzi 0B / nic / wszystkie
  reset                             przywróć całą bazową ramkę 0B
  help                              wyświetl pomoc
  quit                              zamknij program

Wszystkie inne żądania obsługiwane są przez Twój oryginalny process_data().
Uwaga: maska bitów statusu wtrysku 0xE000 to nadal HIPOTEZA.
'''


def command_loop(state: FrameState, stop: threading.Event):
    print(HELP_TEXT)
    state.show()
    while not stop.is_set():
        try:
            line = input('lpg> ').strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd == 'help' and len(parts) == 1:
                print(HELP_TEXT)
            elif cmd == 'show' and len(parts) == 1:
                state.show()
            elif cmd in ('tred', 'tgas', 'pressure', 'press', 'pressure-special', 'press-special', 'rpm') and len(parts) == 2:
                state.set_physical(cmd, parse_float(parts[1]))
                state.show()
            elif cmd == 'inj' and len(parts) == 4:
                fuel = {'b': 'petrol', 'g': 'gas'}.get(parts[1].lower())
                if fuel is None:
                    raise ValueError('Użyj inj b/g numer czas_ms')
                state.set_physical(fuel + parts[2], parse_float(parts[3]))
                state.show()
            elif cmd == 'set' and len(parts) == 3:
                state.set_raw(int(parts[1]), parse_hex(parts[2]), 1)
                state.show()
            elif cmd == 'set16' and len(parts) == 3:
                state.set_raw(int(parts[1]), parse_hex(parts[2]), 2)
                state.show()
            elif cmd == 'reg' and len(parts) == 3:
                state.set_register(parse_hex(parts[1]), parse_hex(parts[2]))
                state.show()
            elif cmd == 'flags' and len(parts) == 4:
                fuel = parts[1].lower()
                ch = int(parts[2])
                flag_high = parse_hex(parts[3])
                if fuel not in ('b', 'g') or ch not in range(1,5):
                    raise ValueError('Flagi: flags b/g kanał(1..4) bajtHEX')
                if flag_high & 0x1F:
                    raise ValueError('Flagi ograniczone do trzech wysokich bitów (E0/...).')
                pos = (9 if fuel == 'b' else 17) + (ch-1)*2
                word = int.from_bytes(state.get()[pos:pos+2], 'big')
                word = (flag_high << 8) | (word & 0x1FFF)
                state.set_raw(pos, word, 2)
                print('UWAGA: interpretacja bitów statusu niepotwierdzona!')
                state.show()
            elif cmd == 'sweep' and len(parts) >= 3:
                state.start_scan(parts[1].lower(), parts[2:])
            elif cmd == 'scan' and len(parts) >= 3:
                state.start_scan(f'raw8:{int(parts[1])}', parts[2:])
            elif cmd == 'scan16' and len(parts) >= 3:
                state.start_scan(f'raw16:{int(parts[1])}', parts[2:])
            elif cmd == 'next' and len(parts) == 1:
                state.advance(1)
            elif cmd == 'prev' and len(parts) == 1:
                state.advance(-1)
            elif cmd == 'stop' and len(parts) == 1:
                state.scan = None
                print('Skan zakończony, bieżący wariant pozostaje aktywny.')
            elif cmd == 'record' and len(parts) >= 2:
                state.record(parts[1], ' '.join(parts[2:]))
            elif cmd == 'trace' and len(parts) == 2 and parts[1] in ('on', 'off', 'all'):
                state.trace = parts[1]
                print(f'Trace: {state.trace}')
            elif cmd == 'reset' and len(parts) == 1:
                state.reset()
            elif cmd in ('quit', 'exit') and len(parts) == 1:
                break
            else:
                print('Nieznane polecenie. Wpisz help.')
        except (ValueError, OverflowError) as exc:
            print(f'Błąd: {exc}')
    stop.set()


def serial_loop(ser, original, state: FrameState, stop: threading.Event):
    buffer = bytearray()
    try:
        while not stop.is_set():
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buffer.extend(chunk)
            while buffer:
                if buffer[0] not in (0xC0, 0xC2):
                    del buffer[0]
                    continue
                n = 5 if buffer[0] == 0xC2 else 4
                if len(buffer) < n:
                    break
                frame = bytes(buffer[:n])
                if frame == SPECIAL_NO_REPLY:
                    del buffer[:n]
                    continue
                if frame[0] == 0xC0 and (sum(frame[:3]) & 0xFF) != frame[3]:
                    if state.trace != 'off':
                        print(f'[SYNC] CRC żądania nie pasuje: {frame.hex(" ").upper()}')
                    del buffer[0]
                    continue

                if frame == REQUEST_0B:
                    response = state.get()
                else:
                    response = state.get_register_override(frame)
                    if response is None:
                        response = original.process_data(frame)

                if response:
                    del buffer[:n]
                    crc = original.calculate_outgoing_crc(response)
                    # Zachowujemy semantykę starego emulatora: osobny marker D0
                    # i dopiero payload z jego bajtem kontrolnym.
                    ser.write(b'\xD0')
                    ser.write(response + bytes([crc]))
                    if state.trace == 'all' or (state.trace == 'on' and frame == REQUEST_0B):
                        print(f'RX {frame.hex(" ").upper()} | '
                              f'TX D0 {response.hex(" ").upper()} {crc:02X}')
                else:
                    # Prawidłowa 4-bajtowa komenda nieznanego typu.
                    if state.trace != 'off':
                        print(f'Nieznane żądanie: {frame.hex(" ").upper()}')
                    del buffer[:n]
    except (serial.SerialException, OSError) as exc:
        print(f'Błąd portu szeregowego: {exc}')
        stop.set()


def main():
    p = argparse.ArgumentParser(description='DiegoG3 laboratoryjny emulator z kodowaniem parametrów')
    p.add_argument('--source', default='emulator.py', help='Oryginalny emulator z process_data')
    p.add_argument('--port', default='COM18')
    p.add_argument('--baudrate', type=int, default=9600)
    p.add_argument('--log', default='DiegoG3_measurements.csv')
    args = p.parse_args()

    original = load_original_program(args.source)
    baseline_registers = {}
    for request, address in REG_REQUESTS.items():
        reply = original.process_data(request)
        if len(reply) != 3 or reply[0] != 0x40:
            raise ValueError(f'Nieprawidłowa statyczna odpowiedź na rejestr {address:02X}: {reply!r}')
        baseline_registers[address] = int.from_bytes(reply[1:], 'big')
    state = FrameState(original.process_data(REQUEST_0B),
                       log_path=Path(args.log), original_registers=baseline_registers)
    stop = threading.Event()
    with serial.Serial(args.port, args.baudrate, timeout=0.1, write_timeout=1.0) as port:
        thread = threading.Thread(target=serial_loop, args=(port, original, state, stop), daemon=True)
        thread.start()
        try:
            command_loop(state, stop)
        finally:
            stop.set()
            thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
