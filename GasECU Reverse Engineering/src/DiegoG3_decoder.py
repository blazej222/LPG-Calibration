"""Statycznie odtworzone funkcje diagnostyczne DiegoG3 3.0.8.2.

Wymaga: DiegoG3_temperature_table.csv w tym samym katalogu.
Brak zaleznosci zewnetrznych i brak komunikacji ze sterownikiem.

Potwierdzone z oryginalnego kodu EXE:
- temperatura: double LUT, indeks 255 - raw; funkcja 0x00563D58
- cisnienie: wspolczynniki i switch 0xFA; funkcja 0x00563950
- rpm: przelicznik okresu, zaokraglenie do 10 obr/min; 0x00563EA0
- nieoflagowany czas wtrysku: raw/187; funkcja 0x00563DEC

Niepotwierdzone: rozmieszczenie/status bitow 0xA000/0x8000 w
surowych dwoch pierwszych kanalach wtrysku.
"""

import argparse
import csv
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _temperature_table():
    csv_path = Path(__file__).with_name('DiegoG3_temperature_table.csv')
    with csv_path.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 256:
        raise ValueError('Tablica musi miec dokladnie 256 pozycji')
    table = [None] * 256
    for row in rows:
        raw = int(row['raw_dec'])
        table[raw] = float(row['temperature_c_exact'])
    if any(x is None for x in table):
        raise ValueError('Brakuje kodow RAW')
    return tuple(table)


def decode_temperature(raw: int) -> float:
    """Tred i Tgas: dokladna wartosc double ze zmierzonego EXE."""
    if not 0 <= raw <= 255:
        raise ValueError('RAW temperatury musi byc bajtem 0..255')
    return _temperature_table()[raw]


def decode_pressure(raw: int, *, sensor_mode: int = 0,
                    register_0d: int | None = None,
                    register_0f: int | None = None) -> float:
    """Pierwszy kanal cisnienia: algorytm 0x00563950.

    sensor_mode==1: normalny wzor niezaleznie od 0xFA.
    sensor_mode!=1 i raw>=0xFA: tryb wyliczenia z dodatkowych rejestrow
    0x0D i 0x0F. Jeśli ich nie znamy, nie mozna odtworzyc wyniku.
    """
    if not 0 <= raw <= 255:
        raise ValueError('RAW cisnienia musi byc bajtem 0..255')
    if sensor_mode == 1 or raw < 0xFA:
        pressure = (raw / 256.0 - 0.04) / (0.00369 * 100.0)
    else:
        if register_0d is None or register_0f is None:
            raise ValueError('Dla RAW>=0xFA wymagane sa rejestry 0x0D i 0x0F')
        if not 0 <= register_0d <= 65535 or not 0 <= register_0f <= 65535:
            raise ValueError('Rejestry 0x0D/0x0F musza byc typu uint16')
        if register_0f > register_0d:
            return 0.0
        difference = register_0d - register_0f
        pressure = (difference / 256.0 + 0.00842) / (0.00242 * 100.0)
    return max(0.0, pressure)


def decode_rpm(raw_word: int) -> int:
    """Kod 0x00563EA0: `10 * Round((120/(N/187/1000))/10)`."""
    if not 0 <= raw_word <= 65535:
        raise ValueError('RAW RPM musi byc uint16')
    if raw_word == 0:
        return 0
    return round(2_244_000.0 / raw_word) * 10


def decode_injection_time_unflagged(raw_word: int) -> float:
    """Kod 0x00563DEC - otrzymuje juz przygotowana wartosc N."""
    if not 0 <= raw_word <= 65535:
        raise ValueError('RAW czasu musi byc uint16')
    return raw_word / 187.0


def experimental_injection_time_from_frame(raw_word: int) -> float:
    """HIPOTEZA: gorne 3 bity slowa stanowia znaczniki statusu.

    A176 -> 0176 -> 2.0 ms; 8176 -> 0176 -> 2.0 ms.
    Usuwanie znacznikow wymaga jeszcze potwierdzenia w asemblerze.
    """
    return decode_injection_time_unflagged(raw_word & 0x1FFF)


def read_hexword(two_bytes: bytes) -> int:
    if len(two_bytes) != 2:
        raise ValueError('Wymagane 2 bajty')
    return int.from_bytes(two_bytes, byteorder='big')


def quick_decode_0b(payload: bytes) -> dict:
    """Mapa pol na podstawie dotychczasowych eksperymentow z emulatorem.

    Uwaga: funkcje konwersji sa odtworzone z EXE, ale polozenie flag
    statusowych w czasach wtrysku nie jest jeszcze potwierdzone.
    """
    if len(payload) != 50 or payload[0] != 0x0B:
        raise ValueError('Oczekiwany payload 50 bajtow, rozpoczynajacy sie 0x0B')
    result = {
        'pressure_bar': decode_pressure(payload[1]) if payload[1] < 0xFA else None,
        'tred_c': decode_temperature(payload[2]),
        'tgas_c': decode_temperature(payload[3]),
        'rpm_candidate': decode_rpm(read_hexword(payload[7:9])),
        'petrol_ms_experimental': [],
        'gas_ms_experimental': [],
    }
    for i in range(4):
        result['petrol_ms_experimental'].append(
            experimental_injection_time_from_frame(read_hexword(payload[9+i*2:11+i*2])))
        result['gas_ms_experimental'].append(
            experimental_injection_time_from_frame(read_hexword(payload[17+i*2:19+i*2])))
    return result


def main():
    p = argparse.ArgumentParser(description='Odtworzone przeliczniki DiegoG3 3.0.8.2')
    sub = p.add_subparsers(dest='cmd', required=True)
    p_temp=sub.add_parser('temp', help='temp 69')
    p_temp.add_argument('hexbyte')
    p_pressure=sub.add_parser('pressure',help='pressure 79')
    p_pressure.add_argument('hexbyte')
    p_pressure.add_argument('--mode',type=int,default=0)
    p_pressure.add_argument('--r0d',type=lambda x:int(x,0))
    p_pressure.add_argument('--r0f',type=lambda x:int(x,0))
    p_rpm=sub.add_parser('rpm',help='rpm 65BA')
    p_rpm.add_argument('hexword')
    p_inj=sub.add_parser('injection', help='injection 0176')
    p_inj.add_argument('hexword')
    p_inj.add_argument('--experimental-flags',action='store_true')
    args = p.parse_args()
    if args.cmd=='temp':
        raw=int(args.hexbyte,16)
        print(f'0x{raw:02X} -> {decode_temperature(raw):.1f}°C')
    elif args.cmd=='pressure':
        raw=int(args.hexbyte,16)
        print(f'0x{raw:02X} -> {decode_pressure(raw,sensor_mode=args.mode, register_0d=args.r0d, register_0f=args.r0f):.2f} bar')
    elif args.cmd=='rpm':
        raw=int(args.hexword,16)
        print(f'0x{raw:04X} -> {decode_rpm(raw)} RPM')
    elif args.cmd=='injection':
        raw=int(args.hexword,16)
        value=experimental_injection_time_from_frame(raw) if args.experimental_flags else decode_injection_time_unflagged(raw)
        print(f'0x{raw:04X} -> {value:.4f} ms')


if __name__ == '__main__':
    main()
