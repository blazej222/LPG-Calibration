from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
import diego_protocol as p
import extract_temperature_lut as luttool


class FrameTest(unittest.TestCase):
    def test_query_checksum(self):
        for command,argument,check in [(0x0B,0x0B,0xD6),(0x53,0x00,0x13),(0x40,0x0D,0x0D),(0x40,0x0F,0x0F),(0xC0,0x00,0x80)]:
            self.assertEqual(p.build_c0(command,argument),bytes([0xC0,command,argument,check]))
            self.assertTrue(p.is_valid_c0(p.build_c0(command,argument)))
        self.assertFalse(p.is_valid_c0(bytes.fromhex('C0 53 00 14')))
        self.assertFalse(p.is_valid_c0(b'\xc0\x53\x00'))

    def test_response_crc(self):
        self.assertEqual(p.frame_response(bytes.fromhex('53 4A')),bytes.fromhex('D0 53 4A 6D'))
        self.assertEqual(p.frame_response(bytes.fromhex('40 00 8E')),bytes.fromhex('D0 40 00 8E 9E'))
        self.assertEqual(p.frame_response(bytes.fromhex('40 00 1A')),bytes.fromhex('D0 40 00 1A 2A'))
        self.assertTrue(p.validate_response(bytes.fromhex('D0 53 4A 6D'),2))
        self.assertFalse(p.validate_response(bytes.fromhex('D0 53 4A 6E'),2))

    def test_catalog_all_entries_lengths_and_crc(self):
        catalog=p.command_catalog()
        self.assertEqual(len(catalog),83)
        self.assertEqual(len([e for e in catalog.values() if e['provenance']=='special_no_reply']),1)
        self.assertEqual(len([e for e in catalog.values() if e['provenance']=='experimental_unmeasured']),1)
        for request,entry in catalog.items():
            self.assertEqual(len(request),entry['request_len'])
            self.assertEqual(entry['request_hex'],request.hex(' ').upper())
            if request[0]==0xc0:
                self.assertTrue(p.is_valid_c0(request))
            else:
                self.assertEqual(request,bytes.fromhex('C2 01 01 6B 87'))
            payload=bytes.fromhex(entry['payload_hex'])
            self.assertEqual(len(payload),entry['payload_len'])
            if entry['provenance']=='special_no_reply':
                self.assertIsNone(entry['wire_reply_hex'])
                self.assertEqual(entry['wire_reply_len'],0)
            else:
                expected=p.frame_response(payload)
                self.assertEqual(entry['wire_reply_len'],len(expected))
                self.assertEqual(entry['wire_reply_hex'],expected.hex(' ').upper())
                self.assertEqual(entry['payload_checksum_hex'],f'{expected[-1]:02X}')
                self.assertTrue(p.validate_response(expected,entry['payload_len']))
        self.assertEqual(p.command_catalog()[bytes.fromhex('C0 0B 0B D6')]['payload_len'],50)
        self.assertEqual(p.command_catalog()[bytes.fromhex('C0 0B 0B D6')]['wire_reply_len'],52)
        self.assertEqual(p.command_catalog()[bytes.fromhex('C0 0B 0B D6')]['payload_checksum_hex'],'77')

    def test_incremental_chunking_and_marker_in_payload(self):
        request=bytes.fromhex('C0 0B 0B D6')
        raw=bytes.fromhex(p.command_catalog()[request]['wire_reply_hex'])
        inst=p.ReplyAccumulator(request)
        frames=[]
        for byte in raw:
            frames.extend(inst.feed(bytes([byte])))
        self.assertEqual(frames,[raw])
        inst=p.ReplyAccumulator(request)
        self.assertEqual(inst.feed(b'\x00\x01'+raw[:8]),[])
        self.assertEqual(inst.feed(raw[8:]),[raw])
        req2=bytes.fromhex('C0 D0 00 90')
        raw2=bytes.fromhex(p.command_catalog()[req2]['wire_reply_hex'])
        self.assertEqual(p.ReplyAccumulator(req2).feed(raw2),[raw2])
        with self.assertRaises(ValueError):p.ReplyAccumulator(bytes.fromhex('C2 01 01 6B 87'))


class DecoderTest(unittest.TestCase):
    def setUp(self):
        self.record=p.command_catalog()[bytes.fromhex('C0 0B 0B D6')]
        self.payload=bytes.fromhex(self.record['payload_hex'])

    def test_baseline_field_offsets(self):
        d=p.decode_0b(self.payload, r0d=0x008E,r0f=0x001A,analog_input_mode=1)
        self.assertAlmostEqual(d['pressure_gas_bar'],1.17251016,places=5)
        self.assertAlmostEqual(d['tred_c'],33.53631112,places=7)
        self.assertAlmostEqual(d['tgas_c'],18.23821472,places=6)
        self.assertAlmostEqual(d['byte4_supply_voltage_candidate_v'],14.0806686,places=5)
        self.assertEqual(d['byte5_gas_level_raw_undecoded'],0x6b)
        self.assertAlmostEqual(d['byte6_analog_v_if_configured'],0.46,places=6)
        self.assertEqual(d['rpm_raw'],0x65BA)
        self.assertEqual(d['rpm'],860)
        self.assertEqual(d['lpg_state_code'],5)
        self.assertTrue(d['gas_indicator_G_red'])
        self.assertEqual(d['petrol_ms'],[2.0]*4+[0.0]*4)
        self.assertEqual(d['gas_ms'],[2.0]*4+[0.0]*4)
        self.assertAlmostEqual(d['map_bar'],0.4060558,delta=.00001)
        self.assertEqual(d['byte42_raw_unknown'],0x72)
        self.assertEqual(d['byte45_bitfield_unknown_semantics'],0xFF)
        self.assertTrue(d['byte46_boolean'])
        self.assertAlmostEqual(d['byte47_temperature_c_if_capability_enabled'],21.99276177,places=5)
        self.assertEqual(d['byte49_raw_unknown'],2)
        self.assertIsNone(p.decode_0b(self.payload,third_temp_from_payload=False)['byte47_temperature_c_if_capability_enabled'])
        with self.assertRaises(ValueError):p.decode_0b(self.payload[:-1])

    def test_masks_are_channel_specific(self):
        self.assertEqual(p.injection_ms(0xa176,'petrol',1),2.0)
        self.assertEqual(p.injection_ms(0x8176,'petrol',2),2.0)
        self.assertEqual(p.injection_ms(0xa176,'petrol',3),0xa176/187.0)
        self.assertEqual(p.injection_ms(0x8176,'gas',8),0x8176/187.0)
        self.assertEqual(p.encode_injection_ms(3.,0xa176,'petrol',1),0xa231)
        self.assertEqual(p.encode_injection_ms(3.,0x8176,'petrol',2),0x8231)
        self.assertEqual(p.encode_injection_ms(3.,0xffff,'petrol',3),0x0231)
        self.assertEqual(p.encode_injection_ms(3.,0xffff,'gas',8),0x0231)
        with self.assertRaises(ValueError):p.encode_injection_ms(60.,0xa176,'petrol',1)

    def test_rpm_measurements(self):
        with (ROOT/'reference'/'DiegoG3_measurements.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),5)
        for row in rows:
            sample=bytes.fromhex(row['payload_hex'])
            self.assertEqual(sample[7:9].hex().upper(),row['rpm_raw'])
            self.assertEqual(p.decode_0b(sample)['rpm'],int(row['displayed_in_DiegoG3']))
        self.assertEqual(p.rpm(0),0)

    def test_map_measurements(self):
        with (ROOT/'reference'/'DiegoG3_measurements(2).csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),15)
        for row in rows:
            sample=bytes.fromhex(row['payload_hex'])
            self.assertEqual(sample[41],int(row['requested_value'],16))
            self.assertEqual(f"{p.decode_0b(sample)['map_bar']:.2f}", f"{float(row['displayed_in_DiegoG3']):.2f}")
        for raw in range(256):
            self.assertEqual(p.encode_map(p.map_bar(raw)),raw)

    def test_status_and_icon_measurements(self):
        with (ROOT/'reference'/'DiegoG3_measurements(1).csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),10)
        expected=['off','zaplon','zaplon','benzyna_cisnienie_na_czerwono','czekam','gaz','benzyna_cisnienie_nie_na_czerwono','benzyna_cisnienie_nie_na_czerwono']
        for i,row in enumerate(rows[:8]):
            data=p.decode_0b(bytes.fromhex(row['payload_hex']))
            self.assertEqual(data['lpg_state_code'],i)
            self.assertEqual(data['lpg_state_display_observed'],expected[i])
            self.assertEqual(data['lpg_state_display_observed'], row['displayed_in_DiegoG3'] + ('_' + row['notes'] if row['notes'] else ''))
        self.assertFalse(p.decode_0b(bytes.fromhex(rows[8]['payload_hex']))['gas_indicator_G_red'])
        self.assertTrue(p.decode_0b(bytes.fromhex(rows[9]['payload_hex']))['gas_indicator_G_red'])

    def test_gas_pressure_special_and_boundaries(self):
        self.assertEqual(p.gas_pressure_bar(0xfa),None)
        self.assertAlmostEqual(p.gas_pressure_bar(0xfa,r0d=0x8e,r0f=0x1a),1.9072107438,places=7)
        for value in [0xfa,0xfe,0xff]:
            self.assertAlmostEqual(p.gas_pressure_bar(value,r0d=0x8e,r0f=0x1a),1.9072107438,places=7)
        self.assertEqual(p.gas_pressure_bar(0xfa,r0d=1,r0f=2),0.0)
        self.assertAlmostEqual(p.gas_pressure_bar(0x79),1.17251016,places=5)
        self.assertEqual(p.encode_gas_pressure_standard(1.17),0x79)
        self.assertEqual(p.encode_gas_pressure_standard(2.53),0xf9)
        with self.assertRaises(ValueError):p.encode_gas_pressure_standard(3.)

    def test_temperature_table_full_and_inverse(self):
        lut=p.temperature_lut()
        self.assertEqual(len(lut),256)
        self.assertEqual(lut[0x69],33.53631112)
        self.assertEqual(lut[0x80],24.92070832)
        self.assertEqual(lut[0x88],21.99276177)
        for raw in range(256):
            self.assertEqual(p.encode_temperature(p.temperature_c(raw)),raw)
        with self.assertRaises(ValueError):p.temperature_c(-1)

    def test_tred_55_conversation_measurements(self):
        with (ROOT/'reference'/'tred_measurements_from_conversation.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),55)
        for row in rows:
            raw=int(row['raw_dec'])
            self.assertEqual(f'{p.temperature_c(raw):.1f}',row['gui_tred_c'])

    def test_regenerate_lut_from_exe_if_available(self):
        exe=ROOT/'vendor'/'DiegoG3-3.0.8.2.exe'
        if not exe.is_file():
            self.skipTest('No local EXE: vendor/ is intentionally excluded from ZIP')
        extracted=luttool.extract(exe)
        self.assertEqual(extracted,list(p.temperature_lut()))


if __name__=='__main__':unittest.main()
