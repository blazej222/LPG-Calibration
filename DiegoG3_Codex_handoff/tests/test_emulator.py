from __future__ import annotations
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from emulator_cli import State
from diego_protocol import decode_0b, validate_response, map_bar, frame_response

class EmulatorOfflineTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state=State(Path(self.temp.name)/'log.csv')

    def test_default_packet_matches_catalog(self):
        a=self.state.response(bytes.fromhex('C0 0B 0B D6'))
        self.assertEqual(len(a),52)
        self.assertTrue(validate_response(a,50))
        self.assertEqual(a[-1],0x77)
        self.assertEqual(a[1:-1],self.state.snapshot())
        self.assertEqual(self.state.response(bytes.fromhex('C0 53 00 13')),bytes.fromhex('D0 53 4A 6D'))
        self.assertIsNone(self.state.response(bytes.fromhex('C2 01 01 6B 87')))
        self.assertIsNone(self.state.response(bytes.fromhex('C0 3A 00 FA')))

    def test_edit_map_state_icon_and_injection_flags(self):
        s=self.state
        s.edit('map','1.07');self.assertEqual(s.snapshot()[41],0x40)
        s.edit('state','3');self.assertEqual(s.snapshot()[9]&0xe0,0x60)
        s.edit('gas-icon','0');self.assertEqual(s.snapshot()[11]&0x80,0)
        d=decode_0b(s.snapshot())
        self.assertEqual(d['lpg_state_code'],3)
        self.assertFalse(d['gas_indicator_G_red'])
        s.edit('petrol1','3.0');s.edit('petrol2','3.0')
        d=decode_0b(s.snapshot())
        self.assertEqual(d['lpg_state_code'],3)
        self.assertFalse(d['gas_indicator_G_red'])
        self.assertEqual(d['petrol_ms'][:2],[3.0,3.0])
        s.edit('gas8','3.0');self.assertEqual(decode_0b(s.snapshot())['gas_ms'][7],3.0)
        self.assertTrue(validate_response(s.response(bytes.fromhex('C0 0B 0B D6')),50))

    def test_pressure_special_registers(self):
        s=self.state;s.edit('pressure-special','1.91')
        self.assertEqual(s.snapshot()[1],0xFA)
        self.assertEqual(s.response(bytes.fromhex('C0 40 0D 0D')),bytes.fromhex('D0 40 00 8E 9E'))
        self.assertEqual(s.response(bytes.fromhex('C0 40 0F 0F')),bytes.fromhex('D0 40 00 1A 2A'))
        self.assertAlmostEqual(decode_0b(s.snapshot(),r0d=s.registers[13],r0f=s.registers[15])['pressure_gas_bar'],1.91,delta=.01)

    def test_status_noninterference_raw49(self):
        before=self.state.snapshot()
        self.state.edit('raw8:49','AA')
        after=self.state.snapshot()
        self.assertEqual(after[:49],before[:49]);self.assertEqual(after[49],0xAA)

    def test_record_new_file(self):
        self.state.record('860','reference')
        self.assertIn('860',self.state.log.read_text(encoding='utf-8-sig'))
        self.assertEqual(self.state.log.read_text(encoding='utf-8-sig').count('\n'),2)

    def test_invalid_input(self):
        for field,val in [('map','5'),('state','8'),('g-icon','-1'),('petrol1','60'),('raw8:50','FF')]:
            with self.subTest(field=field,val=val),self.assertRaises(ValueError):
                self.state.edit(field,val)

if __name__=='__main__':unittest.main()
