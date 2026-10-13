"""Interaktywny tester odpowiedzi 0B sterownika LPG.

Zachowaj dotychczasowy program jako emulator.py (lub podaj --source PLIK).
Uruchom: python lpg_emulator_tester.py --source emulator.py --port COM18

Import oryginalnego pliku nie uruchamia jego pętli, ponieważ jest ona
chroniona przez if __name__ == '__main__'.
"""

import argparse
import importlib.util
from pathlib import Path
import threading

import serial

LIVE_REQUEST = bytes.fromhex('C0 0B 0B D6')
SPECIAL_REQUEST = bytes.fromhex('C2 01 01 6B 87')
BASELINES = (0xA000, 0x8000, 0, 0)


def load_original(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Nie znaleziono oryginalnego programu: {path}')
    spec = importlib.util.spec_from_file_location('lpg_original_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'process_data', None)):
        raise ValueError('Brak funkcji process_data(data) w oryginalnym pliku')
    if not callable(getattr(module, 'calculate_outgoing_crc', None)):
        raise ValueError('Brak funkcji calculate_outgoing_crc(data) w oryginalnym pliku')
    return module


class LivePayload:
    def __init__(self, original):
        self.original = bytes(original)
        if len(self.original) != 50 or self.original[0] != 0x0B:
            raise ValueError(
                f'Odpowiedz 0B ma {len(self.original)} bajtow; oczekiwano 50 i ID 0B'
            )
        self.data = bytearray(self.original)
        self.lock = threading.Lock()
        self.scan = None
        self.scan_index = 0
        self.trace = 'off'

    def snapshot(self):
        with self.lock:
            return bytes(self.data)

    def reset(self):
        with self.lock:
            self.data[:] = self.original
            self.scan = None
        self.show()

    def _write(self, offset, width, value, cancel_scan=True):
        if not (1 <= offset and offset + width <= len(self.data)):
            raise ValueError(f'Zakres bajtow: 1..{len(self.data) - width}')
        if not (0 <= value < (1 << (width * 8))):
            raise ValueError(f'Wartosc poza zakresem {width * 8} bitow')
        with self.lock:
            self.data[offset:offset + width] = value.to_bytes(width, 'big')
            if cancel_scan:
                self.scan = None

    def set_byte(self, offset, value):
        self._write(offset, 1, value)
        self.show()

    def set_word(self, offset, value):
        self._write(offset, 2, value)
        self.show()

    def set_injection(self, fuel, channel, ms):
        fuel = fuel.lower()
        if fuel not in ('b', 'g') or not (1 <= channel <= 4):
            raise ValueError('Podaj rodzaj b/g i kanal 1..4')
        if ms < 0:
            raise ValueError('Czas nie moze byc ujemny')
        offset = (9 if fuel == 'b' else 17) + (channel - 1) * 2
        baseline = BASELINES[channel - 1] if fuel == 'b' else 0
        raw = baseline + round(ms * 187)
        self._write(offset, 2, raw)
        print(f'Zakodowano {ms:.4f} ms -> {raw:04X} (model /187)')
        self.show()

    def start_scan(self, offset, width, values):
        if not values:
            raise ValueError('Podaj przynajmniej jedna wartosc HEX')
        # Walidacja wszystkich wartości przed rozpoczęciem skanowania.
        if not (1 <= offset and offset + width <= len(self.data)):
            raise ValueError('Nieprawidlowy offset')
        if any(not (0 <= v < (1 << (width * 8))) for v in values):
            raise ValueError('Jedna z wartosci nie miesci sie w polu')
        self.scan = (offset, width, values)
        self.scan_index = 0
        self.apply_scan()

    def apply_scan(self):
        if self.scan is None:
            print('Brak aktywnego skanowania. Uzyj scan lub scan16.')
            return
        offset, width, values = self.scan
        value = values[self.scan_index]
        self._write(offset, width, value, cancel_scan=False)
        print(f'>>> WARIANT {self.scan_index + 1}/{len(values)}: '
              f'offset={offset}, RAW={value:0{width * 2}X}')
        self.show()

    def advance(self, step):
        if self.scan is None:
            print('Najpierw uruchom scan lub scan16.')
            return
        length = len(self.scan[2])
        self.scan_index = (self.scan_index + step) % length
        self.apply_scan()

    def show(self):
        d = self.snapshot()
        pressure = (max(0, 0.01058 * d[1] - 0.107)
                    if d[1] < 0xFA else None)
        rpm_raw = int.from_bytes(d[7:9], 'big')
        rpm_guess = (22_500_000 / rpm_raw) if rpm_raw else None

        print('\nRAMKA 0B (payload, bez D0 i CRC):')
        print(' '.join(f'{byte:02X}' for byte in d))
        print(f'  Cisnienie [1]: {d[1]:02X} -> '
              + (f'~{pressure:.2f} bar (hipoteza)' if pressure is not None
                 else 'wartosc specjalna / nieznana'))
        print(f'  Tred [2]: {d[2]:02X} | Tgas [3]: {d[3]:02X} '
              '(brak temperatury odniesienia)')
        print(f'  RPM [7:9]: {rpm_raw:04X} -> '
              + (f'~{rpm_guess:.0f} obr/min (hipoteza!)'
                 if rpm_guess is not None else 'nieznane'))
        for i in range(4):
            b = int.from_bytes(d[9 + 2*i:11 + 2*i], 'big')
            g = int.from_bytes(d[17 + 2*i:19 + 2*i], 'big')
            print(f'  Wtryskiwacz {i+1}: benzyna {b:04X} -> '
                  f'{(b-BASELINES[i])/187:.3f} ms | '
                  f'gaz {g:04X} -> {g/187:.3f} ms (model)')
        print()


HELP = '''Dostepne komendy (offsety dziesietne, wartosci RAW w HEX):
  show                         pokaz aktualna ramke 0B
  set 1 F9                     ustaw jeden bajt: cisnienie = F9
  set 2 69                     ustaw Tred (RAW)
  set 3 93                     ustaw Tgas (RAW)
  set16 7 65BA                 ustaw dwa bajty RPM (Big Endian)
  set16 9 A176                 ustaw dwa bajty benzyny 1
  inj b 1 2.50                 ustaw czas benzyny 1 = 2.50 ms (model /187)
  inj g 1 2.50                 ustaw czas gazu 1 = 2.50 ms (model /187)
  scan 1 0A 0B 68 D2 F9 FA FE FF   testuj bajt cisnienia
  scan 2 61 65 69 6D 71          testuj temperature reduktora
  scan 3 8B 8F 93 97            testuj temperature gazu
  scan 7 45 65 85               testuj starszy bajt RPM
  scan 8 9A BA DA               testuj mlodszy bajt RPM
  scan16 7 45BA 65BA 659A 65DA 85BA  testuj cale pole RPM
  scan16 9 A000 A032 A064 A100 A176 AA00  testuj benzyna 1
  next / prev                  nastepny / poprzedni wariant; potem porownaj GUI
  stop                         zakoncz sekwencje testowa, zostaw wariant
  reset                        przywroc oryginalny payload 0B
  trace on / off / all         loguj zapytania 0B / nic / wszystkie
  help                         pokaz pomoc
  quit                         zakoncz program

Po kazdej zmianie nastepna odpowiedz na C0 0B 0B D6 zawiera nowe wartosci.
Program nigdy nie modyfikuje pozostalych odpowiedzi ze slownika.
'''


def parse_hex(token):
    return int(token, 16)


def command_loop(state, stop):
    print(HELP)
    state.show()
    while not stop.is_set():
        try:
            line = input('lpg> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        fields = line.split()
        op = fields[0].lower()
        try:
            if op == 'help':
                print(HELP)
            elif op == 'show' and len(fields) == 1:
                state.show()
            elif op == 'set' and len(fields) == 3:
                state.set_byte(int(fields[1]), parse_hex(fields[2]))
            elif op == 'set16' and len(fields) == 3:
                state.set_word(int(fields[1]), parse_hex(fields[2]))
            elif op == 'inj' and len(fields) == 4:
                state.set_injection(fields[1], int(fields[2]), float(fields[3]))
            elif op in ('scan', 'scan16') and len(fields) >= 3:
                state.start_scan(int(fields[1]), 1 if op == 'scan' else 2,
                                 [parse_hex(s) for s in fields[2:]])
            elif op == 'next' and len(fields) == 1:
                state.advance(1)
            elif op == 'prev' and len(fields) == 1:
                state.advance(-1)
            elif op == 'stop' and len(fields) == 1:
                state.scan = None
                print('Skanowanie zakonczone, wartosc pozostala bez zmian.')
            elif op == 'reset' and len(fields) == 1:
                state.reset()
            elif op == 'trace' and len(fields) == 2 and fields[1] in ('on', 'off', 'all'):
                state.trace = fields[1]
                print(f'Trace: {state.trace}')
            elif op in ('quit', 'exit') and len(fields) == 1:
                break
            else:
                print('Nieznana skladnia. Wpisz help.')
        except (ValueError, IndexError) as exc:
            print(f'Blad polecenia: {exc}')
    stop.set()


def serial_loop(ser, module, state, stop):
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
                frame_len = 5 if buffer[0] == 0xC2 else 4
                if len(buffer) < frame_len:
                    break
                frame = bytes(buffer[:frame_len])
                if frame == SPECIAL_REQUEST:
                    del buffer[:frame_len]
                    continue
                if frame == LIVE_REQUEST:
                    response = state.snapshot()
                else:
                    response = module.process_data(frame)
                if response:
                    del buffer[:frame_len]
                    packet = b'\xD0' + response + bytes([
                        module.calculate_outgoing_crc(response)
                    ])
                    # Zachowujemy dwa osobne wywolania write() jak w oryginale.
                    ser.write(b'\xD0')
                    ser.write(packet[1:])
                    if state.trace == 'all' or (
                            state.trace == 'on' and frame == LIVE_REQUEST):
                        print(f'RX: {frame.hex(" ").upper()} | '
                              f'TX: {packet.hex(" ").upper()}')
                else:
                    print(f'Unknown data: {frame.hex().upper()}')
                    del buffer[0]
    except serial.SerialException as exc:
        print(f'Blad portu szeregowego: {exc}')
        stop.set()


def main():
    parser = argparse.ArgumentParser(description='Interaktywny tester ramek 0B LPG')
    parser.add_argument('--source', default='emulator.py',
                        help='Plik z oryginalnym process_data (domyslnie emulator.py)')
    parser.add_argument('--port', default='COM18', help='Port COM (domyslnie COM18)')
    parser.add_argument('--baudrate', type=int, default=9600)
    args = parser.parse_args()

    module = load_original(args.source)
    state = LivePayload(module.process_data(LIVE_REQUEST))
    stop = threading.Event()

    with serial.Serial(port=args.port, baudrate=args.baudrate, timeout=0.1) as ser:
        thread = threading.Thread(target=serial_loop,
                                  args=(ser, module, state, stop), daemon=True)
        thread.start()
        try:
            command_loop(state, stop)
        finally:
            stop.set()
            thread.join(timeout=0.5)


if __name__ == '__main__':
    main()
