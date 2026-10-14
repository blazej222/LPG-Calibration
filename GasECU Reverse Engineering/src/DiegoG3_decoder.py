"""DiegoG3 3.0.8.2: dekoder ramki 0B i funkcje kodowania.

Wersja po analizach 14.10.2026. Wszystkie offsety liczone od 0B,
BEZ znacznika D0 i BEZ końcowego bajtu kontrolnego.

Pewne z pomiarów: RPM (5 punktów), MAP w bieżącej konfiguracji (15),
status LPG (0..7), flaga ikony G. Potwierdzone analizą kodu:
maski dwóch pierwszych słów benzyny, dzielnik czasu 187 i LUT temperatur.
Nadal nieznane: semantyka bajtów 42..46/48..49, poziom gazu,
konfiguracje alternatywnych kanałów czujników.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

PAYLOAD_SIZE = 50
INJECTION_TICKS_PER_MS = 187
PETROL_OFFSETS = (9, 11, 13, 15, 25, 27, 29, 31)
GAS_OFFSETS = (17, 19, 21, 23, 33, 35, 37, 39)
PETROL_MASKS = (0x1FFF, 0x7FFF) + (0xFFFF,) * 6
GAS_MASKS = (0xFFFF,) * 8
LPG_STATES = {
    0: 'OFF',
    1: 'ZAPLON_1',
    2: 'ZAPLON_2',
    3: 'BENZYNA_CISNIENIE_CZERWONE',
    4: 'CZEKAM',
    5: 'GAZ',
    6: 'BENZYNA_6',
    7: 'BENZYNA_7',
}


@lru_cache(maxsize=1)
def _temperature_table() -> tuple[float, ...]:
    path = Path(__file__).with_name('DiegoG3_temperature_table.csv')
    with path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 256:
        raise ValueError(f'Tablica temperatur zawiera {len(rows)} zamiast 256 wierszy')
    values = [None] * 256
    for row in rows:
        idx = int(row['raw_dec'])
        if not 0 <= idx <= 255 or values[idx] is not None:
            raise ValueError('Duplikat lub zly indeks w tablicy temperatur')
        values[idx] = float(row['temperature_c_exact'])
    if any(v is None for v in values):
        raise ValueError('Niekompletna tablica temperatur')
    return tuple(values)


def _check(raw: int, bits: int):
    if not isinstance(raw, int) or not 0 <= raw < (1 << bits):
        raise ValueError(f'Wartosc RAW musi miescic sie w uint{bits}')


def decode_temperature(raw: int) -> float:
    _check(raw, 8)
    return _temperature_table()[raw]


def encode_temperature(celsius: float, previous: int | None = None) -> int:
    """Najbliższa osiągalna wartość; w przypadku remisu zachowaj poprzednią."""
    values = _temperature_table()
    if not min(values) <= celsius <= max(values):
        raise ValueError(f'Temperatura spoza zakresu LUT: {min(values):.2f}..{max(values):.2f} C')
    return min(range(256), key=lambda x: (abs(values[x] - celsius),
                                          abs(x-previous) if previous is not None else 0))


def decode_pressure(raw: int, *, sensor_mode: int = 0,
                    register_0d: int | None = None,
                    register_0f: int | None = None) -> float:
    """Ciśnienie gazu (byte 1); gałąź 0xFA..FF zależy od konfiguracji."""
    _check(raw, 8)
    if sensor_mode == 1 or raw < 0xFA:
        return max(0., (raw / 256.0 - 0.04) / 0.369)
    if register_0d is None or register_0f is None:
        raise ValueError('RAW>=FA wymaga rejestrow 0D i 0F')
    _check(register_0d, 16)
    _check(register_0f, 16)
    if register_0d < register_0f:
        return 0.0
    return ((register_0d - register_0f)/256 + 0.00842)/0.242


def encode_pressure(bar: float, previous: int | None = None) -> int:
    """Dobierz bajt z zakresu 00..F9, bez specjalnego kodu >=FA."""
    if not 0 <= bar <= decode_pressure(0xF9) + 0.005:
        raise ValueError('Standardowy zakres cisnienia: 0..~2.53 bar (RAW 00..F9)')
    return min(range(0xFA), key=lambda x: (abs(decode_pressure(x)-bar),
                                            abs(x-previous) if previous is not None else 0))


def decode_map(raw: int, *, mode: int = 0) -> float:
    """Ciśnienie kolektora (MAP, bajt 41).

    Tryb 0 odpowiada 15/15 pomiarom z DiegoG3.
    Inny wariant (mode=1) pozostaje zależny od konfiguracji czujnika.
    """
    _check(raw, 8)
    if mode == 1:
        return max(0., (raw/256 - 0.04)/0.369)
    return max(0., (raw/256 + 0.00842)/0.242)


def encode_map(bar: float, previous: int | None = None, *, mode: int = 0) -> int:
    achievable = [decode_map(i, mode=mode) for i in range(256)]
    low, high = min(achievable), max(achievable)
    if not low - 0.005 <= bar <= high + 0.005:
        raise ValueError(f'MAP poza dostepnym zakresem kodowania: {low:.3f}..{high:.3f} bar')
    return min(range(256), key=lambda x: (abs(achievable[x]-bar),
                                          abs(x-previous) if previous is not None else 0))


def decode_voltage(raw: int) -> float:
    """Prawdopodobnie napięcie zasilania (bajt 4)."""
    _check(raw, 8)
    return (5.0*raw/256)/0.215


def encode_voltage(volts: float, previous: int | None = None) -> int:
    max_v = decode_voltage(255)
    if not 0 <= volts <= max_v + 0.01:
        raise ValueError(f'Napiece poza zakresem: 0..{max_v:.2f} V')
    return min(range(256), key=lambda x: (abs(decode_voltage(x)-volts),
                                          abs(x-previous) if previous is not None else 0))


def decode_analog(raw: int, *, input_mode: int | None = None) -> float | None:
    """Potencjalne wejście lambda: 0.02 V/RAW wyłącznie w trybach 1..6."""
    _check(raw, 8)
    return raw*0.02 if input_mode in (1,2,3,4,5,6) else None


def decode_rpm(raw_word: int) -> int:
    _check(raw_word, 16)
    return 0 if raw_word == 0 else 10*round(2_244_000/raw_word)


def encode_rpm(rpm: float, previous: int | None = None) -> int:
    if rpm == 0:
        return 0
    if not decode_rpm(0xFFFF) <= rpm <= decode_rpm(1):
        raise ValueError(f'RPM poza zakresem: 0 lub {decode_rpm(0xFFFF)}..{decode_rpm(1)}')
    return min(range(1,65536), key=lambda x: (abs(decode_rpm(x)-rpm),
                                              abs(x-previous) if previous is not None else 0))


def decode_injection_time_unflagged(raw_word: int) -> float:
    _check(raw_word, 16)
    return raw_word / INJECTION_TICKS_PER_MS


def experimental_injection_time_from_frame(raw_word: int) -> float:
    """Zgodność ze starszym API: działa WYŁĄCZNIE dla benzyny 1."""
    return decode_injection_time_unflagged(raw_word & PETROL_MASKS[0])


def _offset(fuel: str, channel: int) -> tuple[int, int]:
    if not 1 <= channel <= 8:
        raise ValueError('Kanal musi byc w zakresie 1..8')
    if fuel.lower() in ('b','petrol','benzyna'):
        return PETROL_OFFSETS[channel-1], PETROL_MASKS[channel-1]
    if fuel.lower() in ('g','gas','gaz'):
        return GAS_OFFSETS[channel-1], GAS_MASKS[channel-1]
    raise ValueError('Rodzaj paliwa: b/g')


def encode_injection_word(ms: float, *, fuel: str, channel: int, previous: int) -> int:
    """Zachowaj TYLKO potwierdzone bity flag, nie zmieniaj innych kanałów."""
    _check(previous, 16)
    _, mask = _offset(fuel, channel)
    if ms < 0:
        raise ValueError('Czas wtrysku nie moze byc ujemny')
    ticks = round(ms*INJECTION_TICKS_PER_MS)
    if not 0 <= ticks <= mask:
        raise ValueError(f'Czas nie miesci sie w masce {mask:04X}, max={mask/187:.3f} ms')
    return (previous & ~mask) | ticks


def decode_injection_word(word: int, *, fuel: str, channel: int) -> float:
    _check(word, 16)
    _, mask = _offset(fuel, channel)
    return (word & mask) / INJECTION_TICKS_PER_MS


def read_hexword(two_bytes: bytes) -> int:
    if len(two_bytes) != 2:
        raise ValueError('Wymagane dokladnie dwa bajty')
    return int.from_bytes(two_bytes, 'big')


def decode_0b(payload: bytes, *, pressure_sensor_mode: int = 0,
              register_0d: int | None = None, register_0f: int | None = None,
              map_mode: int = 0, analog_input_mode: int | None = None) -> dict:
    if len(payload) != PAYLOAD_SIZE or payload[0] != 0x0B:
        raise ValueError('Oczekiwany payload 50 bajtow z ID 0B')
    p_words = [read_hexword(payload[i:i+2]) for i in PETROL_OFFSETS]
    g_words = [read_hexword(payload[i:i+2]) for i in GAS_OFFSETS]
    mode_code = payload[9] >> 5
    gas_icon_flag = bool(payload[11] & 0x80)
    if payload[1] >= 0xFA and pressure_sensor_mode != 1 and (register_0d is None or register_0f is None):
        p1 = None
    else:
        p1 = decode_pressure(payload[1], sensor_mode=pressure_sensor_mode,
                             register_0d=register_0d, register_0f=register_0f)
    return {
        'message_id': 0x0B,
        'pressure_raw': payload[1],
        'pressure_bar': p1,
        'tred_raw': payload[2], 'tred_c': decode_temperature(payload[2]),
        'tgas_raw': payload[3], 'tgas_c': decode_temperature(payload[3]),
        'voltage_raw': payload[4], 'supply_voltage_candidate_v': decode_voltage(payload[4]),
        'gas_level_raw_unknown_scale': payload[5],
        'analog_raw': payload[6], 'analog_volts_if_mode_known': decode_analog(payload[6],input_mode=analog_input_mode),
        'rpm_word': read_hexword(payload[7:9]), 'rpm': decode_rpm(read_hexword(payload[7:9])),
        'lpg_state_code': mode_code, 'lpg_state_label': LPG_STATES[mode_code],
        'gas_icon_red': gas_icon_flag,
        'petrol_words': p_words, 'gas_words': g_words,
        'petrol_time_ms': [(w & mask)/187. for w,mask in zip(p_words,PETROL_MASKS)],
        'gas_time_ms': [w/187. for w in g_words],
        'map_raw': payload[41], 'map_bar': decode_map(payload[41],mode=map_mode),
        'byte42_unknown': payload[42], 'byte43_unknown': payload[43],
        'byte44_unknown': payload[44], 'byte45_flags_unresolved': payload[45],
        'byte46_boolean': bool(payload[46]),
        'temperature3_raw': payload[47], 'temperature3_c': decode_temperature(payload[47]),
        'byte48_unknown': payload[48], 'byte49_unknown': payload[49],
    }


def quick_decode_0b(payload: bytes) -> dict:
    """Stary punkt wejścia; teraz zwraca pełną, poprawioną strukturę."""
    return decode_0b(payload)


def main():
    parser = argparse.ArgumentParser(description='Dekoder DiegoG3 3.0.8.2')
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('temp', 'pressure', 'map', 'rpm', 'injection'):
        cmd = sub.add_parser(name)
        cmd.add_argument('raw', help='Wartosc w HEX, np. 69 / 65BA')
    p = sub.add_parser('payload',help='50 bajtow HEX oddzielonych spacjami')
    p.add_argument('hex_data')
    args = parser.parse_args()
    if args.cmd == 'payload':
        print(json.dumps(decode_0b(bytes.fromhex(args.hex_data)), indent=2, ensure_ascii=False))
        return
    raw = int(args.raw,16)
    fn = {'temp':decode_temperature,'pressure':decode_pressure,
          'map':decode_map,'rpm':decode_rpm,'injection':decode_injection_time_unflagged}[args.cmd]
    print(fn(raw))


if __name__ == '__main__':
    main()
