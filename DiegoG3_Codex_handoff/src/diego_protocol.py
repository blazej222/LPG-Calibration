"""Read-only reference implementation for DiegoG3 3.0.8.2 protocol.

NO real ECU writes. Only implements framing, validation, emulator packet lookup,
parameter encode/decode, and an incremental RX parser for the known static replies.
The command catalog is an emulator snapshot, NOT evidence of live ECU response semantics.
"""
from __future__ import annotations
import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'reference'
LPG_STATES = {
    0: ('OFF','off'),
    1: ('IGNITION_1','zaplon'),
    2: ('IGNITION_2','zaplon'),
    3: ('PETROL_PRESSURE_RED','benzyna_cisnienie_na_czerwono'),
    4: ('WAITING','czekam'),
    5: ('GAS','gaz'),
    6: ('PETROL_6','benzyna_cisnienie_nie_na_czerwono'),
    7: ('PETROL_7','benzyna_cisnienie_nie_na_czerwono'),
}
PETROL_OFFSETS = (9, 11, 13, 15, 25, 27, 29, 31)
GAS_OFFSETS = (17, 19, 21, 23, 33, 35, 37, 39)
PETROL_TIME_MASKS = (0x1FFF,0x7FFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF)
INJECTION_DIVISOR=187

@lru_cache(maxsize=1)
def temperature_lut()->tuple[float,...]:
    values=[None]*256
    with (REFERENCE/'temperature_lut.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n=int(row['raw_dec']);assert 0<=n<=255 and values[n] is None
            values[n]=float(row['temperature_c_exact'])
    assert all(v is not None for v in values)
    return tuple(values)

def temperature_c(raw:int)->float:
    check_u8(raw)
    return temperature_lut()[raw]

def encode_temperature(celsius:float)->int:
    temperatures=temperature_lut()
    if not min(temperatures)<=celsius<=max(temperatures):
        raise ValueError('Requested temperature is outside LUT values')
    return min(range(256),key=lambda x:abs(temperatures[x]-celsius))

def check_u8(n:int):
    if not isinstance(n,int) or not 0<=n<=255:raise ValueError('Expected unsigned byte 0..255')

def check_u16(n:int):
    if not isinstance(n,int) or not 0<=n<=65535:raise ValueError('Expected uint16 0..65535')

def u16be(b:bytes,offset:int)->int:
    if len(b)<offset+2:raise ValueError('Not enough data for uint16')
    return (b[offset]<<8) | b[offset+1]

def request_checksum(prefix3:bytes)->int:
    if len(prefix3)!=3:raise ValueError('3 bytes required')
    return sum(prefix3)&255

def build_c0(cmd:int,arg:int)->bytes:
    check_u8(cmd); check_u8(arg)
    first=bytes((0xC0,cmd,arg))
    return first+bytes([request_checksum(first)])

def is_valid_c0(packet:bytes)->bool:
    return len(packet)==4 and packet[0]==0xC0 and packet[3]==(sum(packet[:3])&255)

def response_checksum(payload:bytes)->int:
    return (sum(payload)-0x30)&255

def frame_response(payload:bytes)->bytes:
    return b'\xD0'+payload+bytes((response_checksum(payload),))

def validate_response(packet:bytes,expected_payload_size:int|None=None)->bool:
    if len(packet)<3 or packet[0]!=0xD0:return False
    if expected_payload_size is not None and len(packet)!=expected_payload_size+2:return False
    return response_checksum(packet[1:-1])==packet[-1]

@lru_cache(maxsize=1)
def command_catalog()->dict[bytes,dict[str,Any]]:
    records=json.loads((REFERENCE/'known_commands.json').read_text(encoding='utf-8'))
    return {bytes.fromhex(r['request_hex']):r for r in records}


def map_bar(raw:int, mode:int=0)->float:
    """Mode 0 experimentally validated 15/15; alternate mode 1 from EXE, untested."""
    check_u8(raw)
    if mode==0:return max(0.,(raw/256.+0.00842)/0.242)
    if mode==1:return max(0.,(raw/256.-0.04)/0.369)
    raise ValueError('Mode must be 0 (observed) or 1 (unverified configuration)')

def encode_map(bar:float,mode:int=0)->int:
    vals=[map_bar(i,mode=mode) for i in range(256)]
    if not min(vals)-0.005<=bar<=max(vals)+0.005:
        raise ValueError(f'MAP outside encodable range: {min(vals):.3f}..{max(vals):.3f} bar')
    return min(range(256),key=lambda x:abs(vals[x]-bar))

def gas_pressure_bar(raw:int,*,sensor_mode:int=0,r0d:int|None=None,r0f:int|None=None)->float|None:
    check_u8(raw)
    if raw<0xFA or sensor_mode==1:
        return max(0.,(raw/256.-0.04)/0.369)
    if r0d is None or r0f is None:return None
    check_u16(r0d);check_u16(r0f)
    if r0f>r0d:return 0.
    return max(0.,((r0d-r0f)/256.+0.00842)/0.242)

def encode_gas_pressure_standard(bar:float)->int:
    vals=[gas_pressure_bar(i) for i in range(250)]
    if not 0<=bar<=max(vals)+0.005:raise ValueError('Above standard branch; requires FA and 0D/0F')
    return min(range(250),key=lambda x:abs(vals[x]-bar))

def rpm(word:int)->int:
    check_u16(word)
    return 0 if word==0 else 10*round(2_244_000/word)

def injection_ms(word:int,fuel:str,channel:int)->float:
    check_u16(word)
    if not 1<=channel<=8:raise ValueError('Channel 1..8')
    if fuel.lower() in ('petrol','benzyna','b'):
        word &= PETROL_TIME_MASKS[channel-1]
    elif fuel.lower() not in ('gas','gaz','g'):
        raise ValueError('Fuel petrol or gas')
    return word/INJECTION_DIVISOR

def encode_injection_ms(ms:float,old_word:int,fuel:str,channel:int)->int:
    check_u16(old_word)
    if not 1<=channel<=8:raise ValueError('Channel 1..8')
    if not math.isfinite(ms) or ms<0:raise ValueError('ms must be finite, nonnegative')
    if fuel.lower() in ('petrol','benzyna','b'):
        mask=PETROL_TIME_MASKS[channel-1]
    elif fuel.lower() in ('gas','gaz','g'):
        mask=0xffff
    else:raise ValueError('Fuel petrol or gas')
    ticks=round(ms*INJECTION_DIVISOR)
    if ticks>mask:raise ValueError('Time does not fit channel bitmask')
    return (old_word&~mask)|ticks

def decode_0b(payload:bytes,*,gas_pressure_mode:int=0,map_mode:int=0,
              r0d:int|None=None,r0f:int|None=None,analog_input_mode:int|None=None,
              third_temp_from_payload:bool=True)->dict[str,Any]:
    """Return values and explicitly uncertainty-labelled fields.

    The third temperature byte47 is only selected by some software configurations;
    if the EXE's capability gate is not satisfied, it reads a separate register.
    """
    if len(payload)!=50 or payload[0]!=0x0B:
        raise ValueError('0B payload must be exactly 50 bytes (without D0 and CRC)')
    petrol=[u16be(payload,i) for i in PETROL_OFFSETS]
    gas=[u16be(payload,i) for i in GAS_OFFSETS]
    state=(petrol[0]>>13)&7
    return {
        'id':payload[0], 'payload_hex':payload.hex(' ').upper(),
        'pressure_gas_bar':gas_pressure_bar(payload[1],sensor_mode=gas_pressure_mode,r0d=r0d,r0f=r0f),
        'tred_c':temperature_c(payload[2]),'tgas_c':temperature_c(payload[3]),
        'byte4_supply_voltage_candidate_v':(payload[4]*5/256)/0.215,
        'byte5_gas_level_raw_undecoded':payload[5],
        'byte6_analog_raw':payload[6],
        'byte6_analog_v_if_configured':payload[6]*0.02 if analog_input_mode in (1,2,3,4,5,6) else None,
        'rpm_raw':u16be(payload,7),'rpm':rpm(u16be(payload,7)),
        'lpg_state_code':state,'lpg_state_name':LPG_STATES[state][0],
        'lpg_state_display_observed':LPG_STATES[state][1],
        'gas_indicator_G_red':bool(petrol[1]&0x8000),
        'petrol_raw_words':petrol,'gas_raw_words':gas,
        'petrol_ms':[injection_ms(w,'petrol',c) for c,w in enumerate(petrol,1)],
        'gas_ms':[injection_ms(w,'gas',c) for c,w in enumerate(gas,1)],
        'map_raw':payload[41],'map_bar':map_bar(payload[41],mode=map_mode),
        'byte42_raw_unknown':payload[42],
        'byte43_raw_unknown':payload[43], 'byte44_raw_unknown':payload[44],
        'byte45_bitfield_unknown_semantics':payload[45],
        'byte46_boolean':bool(payload[46]),
        'byte47_temperature_c_if_capability_enabled':temperature_c(payload[47]) if third_temp_from_payload else None,
        'byte48_raw_unknown':payload[48], 'byte49_raw_unknown':payload[49],
    }

class ReplyAccumulator:
    """Fixed-length reply frame assembler keyed by the outstanding request.

    Never assume pyserial read boundaries equal frame boundaries. A single known
    reply is D0 + known-payload-len + CRC. Caller manages request timeouts and
    unsolicited frames; this parser does not infer length from D0 alone.
    """
    def __init__(self,request:bytes):
        entry=command_catalog().get(request)
        if entry is None or entry['provenance']=='special_no_reply':
            raise ValueError('Unknown request or request without response')
        self.expected_size=entry['payload_len']+2
        self.buf=bytearray()

    def feed(self,chunk:bytes)->list[bytes]:
        self.buf.extend(chunk)
        complete=[]
        while True:
            start=self.buf.find(0xd0)
            if start<0:
                self.buf.clear();break
            del self.buf[:start]
            if len(self.buf)<self.expected_size:break
            packet=bytes(self.buf[:self.expected_size])
            if validate_response(packet,self.expected_size-2):
                complete.append(packet)
                del self.buf[:self.expected_size]
            else:
                del self.buf[0]
        return complete

if __name__=='__main__':
    print(json.dumps(decode_0b(bytes.fromhex(command_catalog()[bytes.fromhex('C0 0B 0B D6')]['payload_hex']),r0d=0x008e,r0f=0x001a),indent=2,ensure_ascii=False))
