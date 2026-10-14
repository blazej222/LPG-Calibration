"""Interaktywny emulator protokołu DiegoG3 — wersja 2026-10-14.

UWAGA: LABORATORYJNA emulacja komunikacji z GUI, bez podłączania sterownika LPG.

W tym samym katalogu musi być:
    DiegoG3_decoder.py
    DiegoG3_temperature_table.csv
    emulator.py  (Twój stary oryginalny program ze słownikiem process_data)

Uruchom:
    python DiegoG3_emulator_interactive.py --source emulator.py --port COM18
    python DiegoG3_emulator_interactive.py --source emulator.py --dry-run

Główne różnice względem starej wersji: 8+8 czasów wtrysku, właściwe
maski dwóch pól benzynowych, LPG state 0..7, G icon, MAP byte 41,
trzecia temperatura byte 47, napięcie byte 4 i poprawiona diagnostyka.
"""
from __future__ import annotations

import argparse
import csv
import io
from contextlib import redirect_stdout
from datetime import datetime
import importlib.util
from pathlib import Path
import threading

try:
    import serial
except ImportError:
    serial = None

from DiegoG3_decoder import (
    LPG_STATES, PAYLOAD_SIZE, _offset,
    decode_0b, decode_pressure, decode_map, decode_temperature,
    decode_rpm, decode_voltage,
    encode_temperature, encode_pressure, encode_map, encode_rpm, encode_voltage,
    decode_injection_word, encode_injection_word,
)

REQUEST_0B = bytes.fromhex('C0 0B 0B D6')
SPECIAL_NO_REPLY = bytes.fromhex('C2 01 01 6B 87')
SERIAL_ERRORS = (OSError,) if serial is None else (OSError, serial.SerialException)

REG_REQUESTS = {
    bytes.fromhex('C0 40 0D 0D'): 0x0D,
    bytes.fromhex('C0 40 0F 0F'): 0x0F,
}


def load_original_program(filename: str):
    path = Path(filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Brak oryginalnego emulatora: {path}. Uzyj --source PLIK')
    spec = importlib.util.spec_from_file_location('diego_original_emulator', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Nie mozna odczytac modulu: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('process_data', 'calculate_outgoing_crc'):
        if not callable(getattr(module,name,None)):
            raise ValueError(f'W {path} brakuje funkcji {name}()')
    return module


def parse_float(value: str) -> float:
    v = float(value.replace(',', '.'))
    if not -float('inf') < v < float('inf'):
        raise ValueError('Podaj skonczona liczbe')
    return v


def parse_hex(value: str) -> int:
    return int(value, 16)


class FrameState:
    def __init__(self, payload: bytes, *, log_path: Path,
                 original_registers: dict[int,int] | None = None):
        if len(payload) != PAYLOAD_SIZE or payload[0] != 0x0B:
            raise ValueError('Odpowiedz 0B musi miec 50 bajtow i zaczynac sie od 0B')
        self.original = bytes(payload)
        self._payload = bytearray(payload)
        self._lock = threading.RLock()
        self.log_path = Path(log_path)
        self.scan: tuple[str,list[str]] | None = None
        self.scan_pos = 0
        self.trace = 'off'
        self.original_registers = dict(original_registers or {0x0D:0x008E, 0x0F:0x001A})
        self.registers = dict(self.original_registers)
        self.register_overrides = set()

    def get(self) -> bytes:
        with self._lock:
            return bytes(self._payload)

    def set_raw(self, offset: int, raw: int, width: int = 1, *, clear_scan=True):
        if width not in (1,2) or not 1 <= offset <= PAYLOAD_SIZE-width:
            raise ValueError('Nieprawidlowy offset; zakres 1..49, bez ID 0B')
        if not 0 <= raw < 1 << (8*width):
            raise ValueError(f'RAW musi byc uint{width*8}')
        with self._lock:
            self._payload[offset:offset+width] = raw.to_bytes(width,'big')
            if clear_scan:
                self.scan = None

    def set_injection_by_field(self, field: str, ms: float, *, clear_scan=True):
        fuel = 'b' if field.startswith('petrol') else 'g'
        channel = int(field[6:] if fuel == 'b' else field[3:])
        offset, _mask = _offset(fuel,channel)
        raw_prev = int.from_bytes(self.get()[offset:offset+2],'big')
        raw_new = encode_injection_word(ms,fuel=fuel,channel=channel,previous=raw_prev)
        self.set_raw(offset,raw_new,2,clear_scan=clear_scan)
        print(f'Wtrysk {fuel.upper()}{channel}: {ms:.3f} ms -> '
              f'RAW={raw_new:04X}, osiągalne '
              f'{decode_injection_word(raw_new,fuel=fuel,channel=channel):.4f} ms')

    def set_state(self, status: int, *, clear_scan=True):
        if status not in LPG_STATES:
            raise ValueError('Stan LPG musi byc liczba 0..7')
        raw = self.get()[9]
        self.set_raw(9,(raw & 0x1F) | (status << 5),clear_scan=clear_scan)
        print(f'Stan LPG {status}: {LPG_STATES[status]}; zachowano bity czasu benzyny 1')

    def set_gas_indicator(self, enabled: bool, *, clear_scan=True):
        raw = self.get()[11]
        self.set_raw(11, (raw & 0x7F) | (0x80 if enabled else 0), clear_scan=clear_scan)
        print('Ikona G: ' + ('czerwona' if enabled else 'wyszarzona'))

    def set_injection_flags_legacy(self, fuel: str, ch: int, byte: int):
        """Zgodność z dawnym 'flags b 1 A0'; brak flag w innych kanałach."""
        if fuel.lower() != 'b' or ch not in (1,2):
            raise ValueError('Flagi rozpoznano tylko dla b 1 (E0) i b 2 (80)')
        if ch == 1:
            if byte & ~0xE0:
                raise ValueError('Benzyna 1: dozwolone tylko bity 0xE0')
            self.set_state(byte >> 5)
        else:
            if byte not in (0,0x80):
                raise ValueError('Benzyna 2: bit flagi to 0x80 lub 0x00')
            self.set_gas_indicator(bool(byte))

    def set_register(self, register: int, word: int, *, clear_scan=True):
        if register not in (0x0D,0x0F) or not 0 <= word < 65536:
            raise ValueError('Dozwolone: rejestry 0D/0F, wartosc uint16')
        with self._lock:
            self.registers[register]=word
            self.register_overrides.add(register)
            if clear_scan:
                self.scan = None
        print(f'Rejestr {register:02X} ustawiony na {word:04X}')

    def get_register_override(self, request: bytes) -> bytes | None:
        reg = REG_REQUESTS.get(request)
        if reg is None:
            return None
        with self._lock:
            return b'\x40' + self.registers[reg].to_bytes(2,'big') if reg in self.register_overrides else None

    def set_pressure_special(self, bar: float, *, clear_scan=True):
        """Dla konfig. pressure_sensor_mode!=1, x>=FA: par. 0D/0F."""
        if bar < 0:
            raise ValueError('Ciśnienie nie moze byc ujemne')
        if bar == 0:
            reg0d, reg0f = 0,1
        else:
            delta = round((bar*0.242-0.00842)*256)
            if delta < 0:
                raise ValueError('Nieosiagalne dodatnie cisnienie w tej galezi')
            with self._lock:
                reg0f = self.registers[0x0F]
            reg0d = reg0f + delta
            if reg0d > 65535:
                raise ValueError('Wartosc przekracza uint16')
        with self._lock:
            self._payload[1] = 0xFA
            self.registers[0x0D],self.registers[0x0F] = reg0d,reg0f
            self.register_overrides.update((0x0D,0x0F))
            if clear_scan:
                self.scan=None
        achieved=decode_pressure(0xFA,register_0d=reg0d,register_0f=reg0f)
        print(f'Pressure SPECIAL: {bar:.3f} bar -> {achieved:.3f} bar '
              f'(RAW FA, rejestry 0D={reg0d:04X},0F={reg0f:04X})')

    def set_physical(self, field: str, value: float, *, clear_scan=True):
        name = field.lower()
        d = self.get()
        if name in ('tred','tgas','temp3','temperature3'):
            idx={'tred':2,'tgas':3,'temp3':47,'temperature3':47}[name]
            raw=encode_temperature(value,d[idx])
            self.set_raw(idx,raw,clear_scan=clear_scan)
            print(f'{name}: {value:.3f}°C -> RAW={raw:02X} -> '
                  f'{decode_temperature(raw):.3f}°C')
        elif name in ('pressure','press'):
            raw=encode_pressure(value,d[1])
            self.set_raw(1,raw,clear_scan=clear_scan)
            print(f'Cisnienie gazu: {value:.3f} bar -> RAW={raw:02X} -> '
                  f'{decode_pressure(raw):.3f} bar')
        elif name in ('pressure-special','press-special'):
            self.set_pressure_special(value,clear_scan=clear_scan)
        elif name in ('map','manifold'):
            raw=encode_map(value,d[41])
            self.set_raw(41,raw,clear_scan=clear_scan)
            print(f'MAP: {value:.3f} bar -> RAW={raw:02X} -> '
                  f'{decode_map(raw):.3f} bar (wariant potwierdzony 15/15)')
        elif name in ('voltage','vbat'):
            raw=encode_voltage(value,d[4])
            self.set_raw(4,raw,clear_scan=clear_scan)
            print(f'Napiecie [hipoteza nazwy pola]: {value:.2f} V -> RAW={raw:02X} '
                  f'-> {decode_voltage(raw):.3f} V')
        elif name=='rpm':
            prev=int.from_bytes(d[7:9],'big')
            raw=encode_rpm(value,prev)
            self.set_raw(7,raw,2,clear_scan=clear_scan)
            print(f'RPM: {value:.0f} -> RAW={raw:04X} -> {decode_rpm(raw)} RPM')
        elif name.startswith('petrol') or (name.startswith('gas') and name[3:].isdigit()):
            self.set_injection_by_field(name,value,clear_scan=clear_scan)
        elif name=='state':
            if not value.is_integer() or not 0 <= value <= 7:
                raise ValueError('state musi byc liczba calkowita 0..7')
            self.set_state(int(value),clear_scan=clear_scan)
        elif name in ('gas-icon','gas-indicator'):
            if value not in (0,1):
                raise ValueError('gas-icon: 0 lub 1')
            self.set_gas_indicator(bool(value),clear_scan=clear_scan)
        else:
            raise ValueError('Nieznany parametr. Wpisz help.')

    def reset(self):
        with self._lock:
            self._payload[:] = self.original
            self.registers = dict(self.original_registers)
            self.register_overrides.clear()
            self.scan=None
        print('Przywrocono wszystkie oryginalne bajty 0B i rejestry 0D/0F.')
        self.show()

    def show(self):
        d=self.get()
        with self._lock:
            reg0d,reg0f=self.registers[0x0D],self.registers[0x0F]
        decoded=decode_0b(d,register_0d=reg0d,register_0f=reg0f)
        print('\n' + '='*68)
        print('ODPOWIEDZ 0B (50 bajtow, BEZ D0 i CRC)')
        print(' '.join(f'{x:02X}' for x in d))
        print(f'Ciśnienie gazu  [01] {d[1]:02X} -> {decoded["pressure_bar"]:.3f} bar')
        print(f'Tred            [02] {d[2]:02X} -> {decoded["tred_c"]:.2f}°C')
        print(f'Tgas            [03] {d[3]:02X} -> {decoded["tgas_c"]:.2f}°C')
        print(f'Napięcie?       [04] {d[4]:02X} -> {decoded["supply_voltage_candidate_v"]:.2f} V [etykieta do potwierdzenia]')
        print(f'Poziom LPG?     [05] {d[5]:02X} [przelicznik nieznany]')
        print(f'Analog/lambda?  [06] {d[6]:02X} [0.02 V/RAW tylko w trybach 1..6]')
        print(f'RPM          [07-08] {d[7:9].hex().upper()} -> {decoded["rpm"]} obr/min')
        print(f'Stan LPG [09,b7:b5] {decoded["lpg_state_code"]} -> {decoded["lpg_state_label"]}')
        print('Ikona G [11,b7] ' + ('CZERWONA' if decoded['gas_icon_red'] else 'WYSZARZONA'))
        for i in range(8):
            print(f'Wtrysk {i+1}: benzyna {decoded["petrol_words"][i]:04X} -> '
                  f'{decoded["petrol_time_ms"][i]:.3f} ms | gaz '
                  f'{decoded["gas_words"][i]:04X} -> {decoded["gas_time_ms"][i]:.3f} ms')
        print(f'MAP             [41] {d[41]:02X} -> {decoded["map_bar"]:.3f} bar '
              '(potwierdzone 15/15)')
        print(f'Bajt 42         [42] {d[42]:02X} [nieznane]')
        print(f'Bajty 43-44  [43-44] {d[43:45].hex(" ").upper()} [nieznane]')
        print(f'Flagi?          [45] {d[45]:02X} [znaczenie w GUI niepotwierdzone]')
        print(f'Boolean?        [46] {d[46]:02X} [znaczenie niepotwierdzone]')
        print(f'Temp3           [47] {d[47]:02X} -> {decoded["temperature3_c"]:.2f}°C [nazwa czujnika nieznana]')
        print(f'Bajty 48-49  [48-49] {d[48:50].hex(" ").upper()} [nieznane]')
        print(f'Rejestry specjalnego ciśnienia: 0D={reg0d:04X} 0F={reg0f:04X}')
        print('='*68+'\n')

    def start_scan(self, field: str, values: list[str]):
        if not values:
            raise ValueError('Brak wartości do skanowania')
        # Walidujemy wszystkie wartości ZANIM zmienimy aktualny stan.
        if field.startswith(('raw8:','raw16:')):
            off = int(field.split(':',1)[1]); width=1 if field.startswith('raw8:') else 2
            if not 1 <= off <= PAYLOAD_SIZE-width:
                raise ValueError('Zly offset skanu')
            for value in values:
                v=parse_hex(value)
                if not 0 <= v < 1 << (8*width):
                    raise ValueError('RAW poza zakresem')
        else:
            # Walidacja przez probe na kopii stanu, bez efektow ubocznych
            temp = FrameState(self.get(),log_path=self.log_path,original_registers=self.registers)
            with redirect_stdout(io.StringIO()):
                for v in values:
                    temp.set_physical(field,parse_float(v))
        with self._lock:
            self.scan=(field,values)
            self.scan_pos=0
        self.apply_scan()

    def apply_scan(self):
        if self.scan is None:
            print('Nie ma aktywnego skanu.')
            return
        field,values = self.scan
        value=values[self.scan_pos]
        print(f'>>> WARIANT {self.scan_pos+1}/{len(values)}: {field} = {value}')
        if field.startswith(('raw8:','raw16:')):
            width=1 if field.startswith('raw8:') else 2
            self.set_raw(int(field.split(':')[1]),parse_hex(value),width,clear_scan=False)
        else:
            self.set_physical(field,parse_float(value),clear_scan=False)
        self.show()

    def advance(self, step: int):
        if self.scan is None:
            print('Nie ma aktywnego skanu. Wpisz sweep/scan/scan16.')
            return
        self.scan_pos=(self.scan_pos+step)%len(self.scan[1])
        self.apply_scan()

    def record(self, displayed: str, notes: str=''):
        d=self.get()
        row = {
            'timestamp':datetime.now().isoformat(timespec='seconds'),
            'field': self.scan[0] if self.scan else 'manual',
            'variant': str(self.scan_pos+1) if self.scan else '',
            'requested_value': self.scan[1][self.scan_pos] if self.scan else '',
            'displayed_in_DiegoG3':displayed,
            'notes':notes,
            'pressure_raw':f'{d[1]:02X}',
            'reg_0d':f'{self.registers[0x0D]:04X}',
            'reg_0f':f'{self.registers[0x0F]:04X}',
            'tred_raw':f'{d[2]:02X}',
            'tgas_raw':f'{d[3]:02X}',
            'rpm_raw':d[7:9].hex().upper(),
            'payload_hex':d.hex(' ').upper(),
            'map_raw':f'{d[41]:02X}',
            'state_code':str(d[9]>>5),
            'gas_icon_flag':str(int(bool(d[11]&0x80))),
            'temp3_raw':f'{d[47]:02X}',
        }
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        existed = self.log_path.is_file() and self.log_path.stat().st_size > 0
        if existed:
            with self.log_path.open('r',encoding='utf-8-sig',newline='') as file:
                headers=csv.DictReader(file).fieldnames
            if headers != list(row):
                raise ValueError('CSV ma inny schemat kolumn. Uzyj --log NOWY_PLIK.csv '
                                 '(nie nadpisuje ani nie uszkadza starych pomiarow).')
        with self.log_path.open('a',newline='',encoding='utf-8' if existed else 'utf-8-sig') as file:
            writer=csv.DictWriter(file,fieldnames=list(row))
            if not existed:
                writer.writeheader()
            writer.writerow(row)
        print(f'Zapisano pomiar do {self.log_path}')


HELP_TEXT='''
DiegoG3 - emulator po aktualizacji 14.10.2026

Wartosci fizyczne (liczby dziesietne):
  tred 35.0                    temperatura reduktora [C]
  tgas 24.9                    temperatura gazu [C]
  temp3 22.0                   dodatkowa temperatura, bajt 47 [C]
  pressure 1.17                cisnienie gazu, bajt 1 [bar], tryb podstawowy
  pressure-special 1.91        cisnienie przez rejestry 0D/0F, RAW FA [bar]
  map 0.41                     cisnienie kolektora MAP, bajt 41 [bar]
  voltage 14.1                 napiecie (hipoteza nazwy pola), bajt 4 [V]
  rpm 860                      obroty w obr/min
  inj b 1 2.5                  benzyna kanal 1..8 [ms]; zachowanie flag
  inj g 8 3.5                  gaz kanal 1..8 [ms]

Statusy LPG (bez naruszania czasow):
  state 5                      0=OFF,1/2=zaplon,3=benzyna cisn. czerwone,
                               4=czekam,5=gaz,6/7=benzyna
  gas-icon 1                   1=ikona G czerwona, 0=wyszarzona
  flags b 1 A0                 historyczny alias do 'state 5'
  flags b 2 80                 historyczny alias do 'gas-icon 1'

Surowe bajty, ustawienia pozostalych nieznanych pol:
  set 45 00                    zmiana RAW jednego bajtu w ramce
  set16 11 8176                zmiana RAW dwoch bajtow (BE)
  reg 0D 008E                  rejestry specjalnego cisnienia 0D/0F

Sekwencje testowe i logowanie:
  sweep map 0.41 0.55 0.81 1.07
  sweep rpm 660 860 1260
  sweep petrol1 1.5 2 2.5 3
  sweep state 0 1 2 3 4 5 6 7
  scan 45 00 01 02 04 08 FF
  scan16 11 0176 8176
  next / prev                  kolejny / poprzedni wariant
  record 860 [uwagi]           zapisz wskazanie i cala ramke do CSV
  stop                         zakoncz skan bez przywracania wartosci
  trace on / off / all         trace on=ramka 0B, all=WSZYSTKIE zapytania
  show                         pokaz wszystkie wartosci z dekodera
  reset                        przywroc oryginalna ramke + rejestry
  help / quit

Wazne: bajt 45 i 42-46/48-49 maja nieustalone znaczenie w GUI.
Inne zapytania nadal obsluguje stary process_data() z emulator.py.
''' 


def command_loop(state: FrameState, stop: threading.Event):
    print(HELP_TEXT)
    state.show()
    while not stop.is_set():
        try:
            line=input('lpg> ').strip()
        except (EOFError,KeyboardInterrupt):
            break
        if not line:
            continue
        parts=line.split()
        cmd=parts[0].lower()
        try:
            if cmd=='help' and len(parts)==1:
                print(HELP_TEXT)
            elif cmd=='show' and len(parts)==1:
                state.show()
            elif cmd in ('tred','tgas','temp3','temperature3','pressure','press',
                         'pressure-special','press-special','map','manifold',
                         'voltage','vbat','rpm','state','gas-icon','gas-indicator') and len(parts)==2:
                state.set_physical(cmd,parse_float(parts[1])); state.show()
            elif cmd=='inj' and len(parts)==4:
                fuel={'b':'petrol','g':'gas'}.get(parts[1].lower())
                if fuel is None:
                    raise ValueError('Uzyj inj b/g NUMER MS')
                state.set_physical(fuel+parts[2],parse_float(parts[3])); state.show()
            elif cmd=='set' and len(parts)==3:
                state.set_raw(int(parts[1]),parse_hex(parts[2]),1); state.show()
            elif cmd=='set16' and len(parts)==3:
                state.set_raw(int(parts[1]),parse_hex(parts[2]),2); state.show()
            elif cmd=='reg' and len(parts)==3:
                state.set_register(parse_hex(parts[1]),parse_hex(parts[2])); state.show()
            elif cmd=='flags' and len(parts)==4:
                state.set_injection_flags_legacy(parts[1],int(parts[2]),parse_hex(parts[3]));state.show()
            elif cmd=='sweep' and len(parts)>=3:
                state.start_scan(parts[1].lower(),parts[2:])
            elif cmd=='scan' and len(parts)>=3:
                state.start_scan(f'raw8:{int(parts[1])}',parts[2:])
            elif cmd=='scan16' and len(parts)>=3:
                state.start_scan(f'raw16:{int(parts[1])}',parts[2:])
            elif cmd=='next' and len(parts)==1:
                state.advance(1)
            elif cmd=='prev' and len(parts)==1:
                state.advance(-1)
            elif cmd=='stop' and len(parts)==1:
                state.scan=None
                print('Skan zakonczony; pozostawiono ostatni wariant.')
            elif cmd=='record' and len(parts)>=2:
                state.record(parts[1],' '.join(parts[2:]))
            elif cmd=='trace' and len(parts)==2 and parts[1] in ('on','off','all'):
                state.trace=parts[1]
                print(f'Trace: {state.trace}')
            elif cmd=='reset' and len(parts)==1:
                state.reset()
            elif cmd in ('quit','exit') and len(parts)==1:
                break
            else:
                print('Nieznana komenda. Wpisz help.')
        except (ValueError,OverflowError) as exc:
            print(f'Blad: {exc}')
    stop.set()


def serial_loop(port, original, state: FrameState, stop: threading.Event):
    """Buforuj strumien (fragmenty pakietu moga pojawiac sie osobno)."""
    incoming=bytearray()
    try:
        while not stop.is_set():
            chunk=port.read(port.in_waiting or 1)
            if not chunk:
                continue
            incoming.extend(chunk)
            while incoming:
                if incoming[0] not in (0xC0,0xC2):
                    del incoming[0]
                    continue
                length=5 if incoming[0]==0xC2 else 4
                if len(incoming)<length:
                    break
                frame=bytes(incoming[:length])
                if frame==SPECIAL_NO_REPLY:
                    if state.trace=='all':
                        print(f'RX {frame.hex(" ").upper()} (brak odpowiedzi)')
                    del incoming[:length]
                    continue
                if frame[0]==0xC0 and (sum(frame[:3])&0xFF)!=frame[3]:
                    if state.trace!='off':
                        print(f'[SYNC] Niepoprawna suma zapytania: {frame.hex(" ").upper()}')
                    del incoming[0]
                    continue
                if state.trace=='all':
                    print(f'RX {frame.hex(" ").upper()}')
                if frame==REQUEST_0B:
                    response=state.get()
                else:
                    response=state.get_register_override(frame)
                    if response is None:
                        response=original.process_data(frame)
                del incoming[:length]
                if not response:
                    if state.trace!='off':
                        print(f'Brak odpowiedzi dla: {frame.hex(" ").upper()}')
                    continue
                crc=original.calculate_outgoing_crc(response)
                # Zachowujemy oryginalne dwa wywolania write: marker D0, payload+CRC.
                port.write(b'\xD0')
                port.write(response+bytes([crc]))
                if state.trace=='all' or (state.trace=='on' and frame==REQUEST_0B):
                    print(f'TX D0 {response.hex(" ").upper()} {crc:02X}')
    except SERIAL_ERRORS as exc:
        print(f'Blad komunikacji szeregowej: {exc}')
        stop.set()


def main():
    parser=argparse.ArgumentParser(description='DiegoG3 emulator — aktualizacja 2026-10-14')
    parser.add_argument('--source',default='emulator.py',help='Stary program ze slownikiem process_data')
    parser.add_argument('--port',default='COM18')
    parser.add_argument('--baudrate',type=int,default=9600)
    parser.add_argument('--log',default='DiegoG3_measurements_v2.csv')
    parser.add_argument('--dry-run',action='store_true',help='Wersja bez otwierania portu COM')
    args=parser.parse_args()
    original=load_original_program(args.source)
    registers={}
    for request, reg in REG_REQUESTS.items():
        answer=original.process_data(request)
        if len(answer)!=3 or answer[0]!=0x40:
            raise ValueError(f'Nieprawidlowa odpowiedz oryginalnego emulatora na rejestr {reg:02X}')
        registers[reg]=int.from_bytes(answer[1:],'big')
    state=FrameState(original.process_data(REQUEST_0B),log_path=Path(args.log),
                     original_registers=registers)
    stop=threading.Event()
    if args.dry_run:
        print('Tryb offline --dry-run: port COM nie zostal otwarty.')
        command_loop(state,stop)
        return
    if serial is None:
        raise SystemExit('Brak pyserial. Zainstaluj: python -m pip install pyserial')
    with serial.Serial(args.port,args.baudrate,timeout=0.1,write_timeout=1.0) as port:
        thread=threading.Thread(target=serial_loop,args=(port,original,state,stop),daemon=True)
        thread.start()
        try:
            command_loop(state,stop)
        finally:
            stop.set()
            thread.join(timeout=1.)


if __name__=='__main__':
    main()
