"""DiegoG3 3.0.8.2 — parser payloadu 0B oparty na analizie EXE.

Uzycie:
    python DiegoG3_0B_verified.py
    python DiegoG3_0B_verified.py --csv DiegoG3_measurements.csv
    python DiegoG3_0B_verified.py --payload "0B 79 69 ..."

Wymagane pliki obok tego modułu:
    DiegoG3_decoder.py
    DiegoG3_temperature_table.csv

Analiza statyczna oryginalnego EXE: parser ramki 0B pod VA 0x00564E90.
Nie steruje zadnym urzadzeniem, nie uzywa pyserial.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from DiegoG3_decoder import decode_temperature, decode_pressure, decode_rpm

DEFAULT_PAYLOAD = bytes.fromhex(
    '0B 79 69 93 9B 6B 17 65 BA A1 76 81 76 01 76 01 76 '
    '01 76 01 76 01 76 01 76 '
    + '00 ' * 16
    + '17 72 00 00 FF 01 88 00 02'
)


def u16be(d: bytes, pos: int) -> int:
    return int.from_bytes(d[pos:pos + 2], 'big')


def decode_0b(payload: bytes, *, second_pressure_mode: int | None = None,
              lambda_input_mode: int | None = None) -> dict:
    """Dekoduje 50-bajtowy payload zaczynajacy sie od 0B (bez D0 i CRC).

    Nazwy fuel/petrol okreslono przez slady uzycia, a nie przez debug symbols.
    W konfiguracjach innych niz przykladowa niektore wskazania moga miec
    odmienny sens lub nie byc aktywne.
    """
    if len(payload) != 50 or payload[0] != 0x0B:
        raise ValueError(f'Nieprawidlowa odpowiedz 0B: {len(payload)} bajtow')

    rp = u16be(payload, 7)
    petrol_raw = [u16be(payload, x) for x in (9, 11, 13, 15, 25, 27, 29, 31)]
    gas_raw = [u16be(payload, x) for x in (17, 19, 21, 23, 33, 35, 37, 39)]

    # 0x00564F67..0x00564F99 — jedyne maski czasu w parserze 0B.
    mode_3bit = (petrol_raw[0] & 0xE000) >> 13
    second_flag = bool(petrol_raw[1] & 0x8000)
    petrol_times_raw = [petrol_raw[0] & 0x1FFF,
                        petrol_raw[1] & 0x7FFF, *petrol_raw[2:]]
    gas_times_raw = gas_raw

    voltage = 5.0 * payload[4] / (256.0 * 0.215)
    # 0x0057BEB0: tryb 1..6 -> raw * 0.02, poza zakresem 0.
    lambda_volts = (payload[6] * 0.02
                    if lambda_input_mode in range(1, 7) else None)

    if second_pressure_mode is None:
        second_pressure = None
    elif second_pressure_mode == 1:
        second_pressure = max(0.0,
            (payload[41] / 256.0 - 0.04) / (0.00369 * 100.0))
    else:
        second_pressure = max(0.0,
            (payload[41] / 256.0 + 0.00842) / (0.00242 * 100.0))

    return {
        'message_id': 0x0B,
        'pressure1_raw': payload[1],
        'pressure1_bar_standard_mode': decode_pressure(payload[1]) if payload[1] < 0xFA else None,
        'tred_c': decode_temperature(payload[2]),
        'tgas_c': decode_temperature(payload[3]),
        'supply_voltage_candidate_v': voltage,
        'level_sensor_raw_candidate': payload[5],
        'analog_lambda_raw': payload[6],
        'analog_lambda_volts_if_enabled': lambda_volts,
        'rpm_raw_word': rp,
        'rpm': decode_rpm(rp),
        'operating_state_3bit': mode_3bit,
        'petrol2_status_bit': second_flag,
        'petrol_words_original': petrol_raw,
        'petrol_words_time_only': petrol_times_raw,
        'petrol_ms': [round(v / 187., 6) for v in petrol_times_raw],
        'gas_words_original': gas_raw,
        'gas_ms': [round(v / 187., 6) for v in gas_times_raw],
        'pressure2_raw': payload[41],
        'pressure2_bar_if_mode_known': second_pressure,
        'byte42_raw': payload[42],
        'status_flags_43': payload[43],
        'status_flags_44': payload[44],
        'eight_bit_status_mask_45': payload[45],
        'boolean_46': bool(payload[46]),
        'temperature3_raw_47': payload[47],
        'temperature3_c_47': decode_temperature(payload[47]),
        'status_byte_48': payload[48],
        'parameter_49': payload[49],
    }


def validate_csv(filename: Path):
    count = 0
    with filename.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            hex_payload = row['payload_hex']
            decoded = decode_0b(bytes.fromhex(hex_payload))
            expected = int(row['displayed_in_DiegoG3'])
            actual = decoded['rpm']
            outcome = 'OK' if expected == actual else 'BLAD'
            print(f"{row['rpm_raw']} -> DiegoG3={expected:4d}, "
                  f"wzor={actual:4d} RPM [{outcome}]")
            count += 1
            assert expected == actual, f'RPM nie zgadza sie w wierszu {count}'
    print(f'Weryfikacja zakonczona: {count}/{count} zgodnych pomiarow RPM')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, help='Walidacja w CSV z emulatora')
    ap.add_argument('--payload', help='50 bajtow HEX, bez D0 i CRC')
    ap.add_argument('--second-pressure-mode', type=int)
    ap.add_argument('--lambda-input-mode', type=int)
    args = ap.parse_args()
    if args.csv:
        validate_csv(args.csv)
    else:
        p = bytes.fromhex(args.payload) if args.payload else DEFAULT_PAYLOAD
        print(json.dumps(decode_0b(p,
            second_pressure_mode=args.second_pressure_mode,
            lambda_input_mode=args.lambda_input_mode),
            ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
