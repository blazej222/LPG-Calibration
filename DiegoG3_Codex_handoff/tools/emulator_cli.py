"""Opcjonalny SAMODZIELNY laboratoryjny emulator Diego G3.

Działa bez historycznego pliku emulator.py: statyczne odpowiedzi czyta z
reference/known_commands.json i pozwala edytować RAMKĘ 0B w ruchu.
Nie łączy się z prawdziwym ECU — podłącz tylko do testowego programu GUI.

  python tools/emulator_cli.py --dry-run
  python -m pip install pyserial
  python tools/emulator_cli.py --port COM18
"""
from __future__ import annotations
import argparse
from datetime import datetime
import csv
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diego_protocol import (
    LPG_STATES, PETROL_OFFSETS, GAS_OFFSETS,
    command_catalog, encode_temperature, encode_map,
    encode_gas_pressure_standard, encode_injection_ms,
    rpm, decode_0b, frame_response, is_valid_c0,
)

Q0B=bytes.fromhex('C0 0B 0B D6')

class State:
    def __init__(self,log:Path):
        self.lock=threading.RLock()
        self.catalog=command_catalog()
        self.payload=bytearray.fromhex(self.catalog[Q0B]['payload_hex'])
        self.original=bytes(self.payload)
        self.log=log
        self.trace=False
        self.scan:tuple[str,list[str]]|None=None
        self.index=0
        self.registers={13:0x008E,15:0x001A}
        self.overridden=set()

    def _set(self,off:int,data:bytes):
        if not 1<=off<=50-len(data):raise ValueError('Zły offset: 1..49 w payloadzie')
        with self.lock:
            self.payload[off:off+len(data)]=data

    def snapshot(self)->bytes:
        with self.lock:return bytes(self.payload)

    def response(self,req:bytes)->bytes|None:
        with self.lock:
            if req==Q0B:return frame_response(bytes(self.payload))
            if req==bytes.fromhex('C0 40 0D 0D') and 13 in self.overridden:
                return frame_response(b'\x40'+self.registers[13].to_bytes(2,'big'))
            if req==bytes.fromhex('C0 40 0F 0F') and 15 in self.overridden:
                return frame_response(b'\x40'+self.registers[15].to_bytes(2,'big'))
        e=self.catalog.get(req)
        if not e or e['provenance']=='special_no_reply':return None
        return bytes.fromhex(e['wire_reply_hex'])

    def edit(self,field:str,value:str):
        field=field.lower().strip()
        if field.startswith('raw8:'):
            n=int(field.split(':')[1]);v=int(value,16)
            if not 0<=v<=255:raise ValueError('uint8')
            self._set(n,bytes([v]));return
        if field.startswith('raw16:'):
            n=int(field.split(':')[1]);v=int(value,16)
            if not 0<=v<=65535:raise ValueError('uint16')
            self._set(n,v.to_bytes(2,'big'));return
        v=float(value.replace(',','.'))
        p=self.snapshot()
        if field in ('tred','tgas','temp3'):
            pos={'tred':2,'tgas':3,'temp3':47}[field]
            self._set(pos,bytes([encode_temperature(v)]))
        elif field in ('map','pressure-map'):
            self._set(41,bytes([encode_map(v)]))
        elif field=='pressure':
            self._set(1,bytes([encode_gas_pressure_standard(v)]))
        elif field=='rpm':
            if v==0:word=0
            elif v>0:
                # inverse of the discrete 10-RPM model; preserve previous word if valid.
                old=int.from_bytes(p[7:9],'big')
                if rpm(old)==v:word=old
                else:
                    # Candidate around 2_244_000/round(v/10); avoid scanning all 65535.
                    target=round(v/10)
                    if target<=0:raise ValueError('RPM must be >= 10 or 0')
                    central=max(1,min(65535,round(2_244_000/target)))
                    options=range(max(1,central-500),min(65535,central+500)+1)
                    word=min(options,key=lambda n:(abs(rpm(n)-v),abs(n-old)))
                self._set(7,word.to_bytes(2,'big'))
            else:raise ValueError('RPM >= 0')
        elif field in ('state','lpg-state'):
            n=int(v)
            if n!=v or not 0<=n<=7:raise ValueError('State 0..7')
            self._set(9,bytes([(p[9]&0x1f)|(n<<5)]))
        elif field in ('g-icon','gas-icon'):
            if v not in (0,1):raise ValueError('0 or 1')
            self._set(11,bytes([(p[11]&0x7f)|(0x80 if v else 0)]))
        elif field.startswith('petrol') or field.startswith('gas'):
            gas=field.startswith('gas')
            channel_text=field[3:] if gas else field[6:]
            ch=int(channel_text)
            if not 1<=ch<=8:raise ValueError('Channel 1..8')
            off=(GAS_OFFSETS if gas else PETROL_OFFSETS)[ch-1]
            prev=int.from_bytes(p[off:off+2],'big')
            word=encode_injection_ms(v,prev,'gas' if gas else 'petrol',ch)
            self._set(off,word.to_bytes(2,'big'))
        elif field=='pressure-special':
            if v<0:raise ValueError('Pressure nonnegative')
            if v==0:r0d,r0f=0,1
            else:
                diff=round((v*0.242-0.00842)*256)
                r0f=self.registers[15]
                r0d=r0f+diff
                if diff<0 or r0d<0:raise ValueError('Pressure unrepresentable')
                if r0d>65535:raise ValueError('uint16 overflow')
            with self.lock:
                self.payload[1]=0xfa
                self.registers.update({13:r0d,15:r0f})
                self.overridden.update({13,15})
        else:raise ValueError('Nieznane pole')

    def show(self):
        d=decode_0b(self.snapshot(),r0d=self.registers[13],r0f=self.registers[15])
        print('\n', self.snapshot().hex(' ').upper())
        print(f"Stan {d['lpg_state_code']} {d['lpg_state_name']}, G_red={d['gas_indicator_G_red']}; "
              f"RPM={d['rpm']}; MAP={d['map_bar']:.3f} bar; "
              f"Tred={d['tred_c']:.2f} C; Tgas={d['tgas_c']:.2f} C; "
              f"Pgas={d['pressure_gas_bar']} bar")
        print('Benzyna [ms]', ', '.join(f'{x:.3f}' for x in d['petrol_ms']))
        print('Gaz     [ms]', ', '.join(f'{x:.3f}' for x in d['gas_ms']))

    def record(self,gui:str,note:str=''):
        p=self.snapshot()
        row={'timestamp':datetime.now().isoformat(), 'field':self.scan[0] if self.scan else 'manual',
             'requested':self.scan[1][self.index] if self.scan else '',
             'raw41':f'{p[41]:02X}', 'raw9':f'{p[9]:02X}', 'GUI':gui,
             'notes':note,'payload_hex':p.hex(' ').upper()}
        self.log.parent.mkdir(parents=True,exist_ok=True)
        has=self.log.exists() and self.log.stat().st_size>0
        if has:
            with self.log.open(encoding='utf-8-sig',newline='') as f:
                if csv.DictReader(f).fieldnames!=list(row):
                    raise ValueError('Schema of existing log differs; use a new --log path')
        with self.log.open('a',encoding='utf-8' if has else 'utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if not has:w.writeheader()
            w.writerow(row)
        print('Zapisano:',self.log)


def serial_worker(state:State,port,stopping:threading.Event):
    incoming=bytearray()
    while not stopping.is_set():
        chunk=port.read(port.in_waiting or 1)
        if not chunk:continue
        incoming.extend(chunk)
        while incoming:
            if incoming[0] not in (0xc0,0xc2):del incoming[0];continue
            size=5 if incoming[0]==0xc2 else 4
            if len(incoming)<size:break
            packet=bytes(incoming[:size])
            if packet[0]==0xc0 and not is_valid_c0(packet):
                if state.trace:print('CRC fail',packet.hex(' ').upper())
                del incoming[0];continue
            del incoming[:size]
            answer=state.response(packet)
            if state.trace:print('RX',packet.hex(' ').upper(), 'TX',answer.hex(' ').upper() if answer else '<no reply>')
            if answer:
                # Two separate writes reproduce historical emulator behavior.
                port.write(answer[:1]);port.write(answer[1:])

HELP='''
Komendy: show | set OFFSET HEX | set16 OFFSET HEX | tred 33.5 | tgas 24.9 |
 temp3 22 | map 0.41 | pressure 1.17 | pressure-special 1.91 |
 rpm 860 | state 5 | gas-icon 1 | inj b 1 2.0 | inj g 8 2.0 |
 scan 41 00 17 40 FF | scan16 9 A176 0176 E176 |
 sweep map 0.41 1.07 2.10 | sweep state 0 1 2 3 4 5 6 7 |
 next | prev | record 0.41 notes | trace on/off | reset | help | quit
'''

def repl(state:State,stop:threading.Event):
    print(HELP);state.show()
    while not stop.is_set():
        try:line=input('lpg> ').strip()
        except (EOFError,KeyboardInterrupt):break
        if not line:continue
        fields=line.split();cmd=fields[0].lower()
        try:
            if cmd=='help':print(HELP)
            elif cmd=='show':state.show()
            elif cmd in ('set','set16') and len(fields)==3:
                state.edit(('raw8:' if cmd=='set' else 'raw16:')+fields[1],fields[2]);state.scan=None;state.show()
            elif cmd=='inj' and len(fields)==4:
                fuel={'b':'petrol','g':'gas'}[fields[1].lower()]
                state.edit(fuel+fields[2],fields[3]);state.scan=None;state.show()
            elif cmd in ('scan','scan16','sweep') and len(fields)>=3:
                name=(('raw8:' if cmd=='scan' else 'raw16:')+fields[1]) if cmd!='sweep' else fields[1]
                state.scan=(name,fields[2:]);state.index=0
                state.edit(name,state.scan[1][0]);state.show()
            elif cmd in ('next','prev') and state.scan:
                name,vals=state.scan;state.index=(state.index+(1 if cmd=='next' else -1))%len(vals)
                state.edit(name,vals[state.index]);state.show()
            elif cmd=='record' and len(fields)>=2:state.record(fields[1],' '.join(fields[2:]))
            elif cmd=='reset':
                with state.lock:
                    state.payload[:]=state.original
                    state.registers={13:0x8e,15:0x1a};state.overridden.clear();state.scan=None
                state.show()
            elif cmd=='trace' and len(fields)==2:state.trace=(fields[1].lower()=='on')
            elif cmd in ('quit','exit'):break
            elif len(fields)==2:
                state.edit(cmd,fields[1]);state.scan=None;state.show()
            else:print(HELP)
        except (ValueError,KeyError) as ex:print('Błąd:',ex)
    stop.set()

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--port',default='COM18');arg.add_argument('--baudrate',type=int,default=9600)
    arg.add_argument('--log',type=Path,default=Path('DiegoG3_measurements_handoff.csv'))
    arg.add_argument('--dry-run',action='store_true')
    args=arg.parse_args()
    s=State(args.log);ev=threading.Event()
    if args.dry_run:
        print('DRY RUN: no serial port opened');repl(s,ev);return
    try:import serial
    except ImportError:raise SystemExit('Install pyserial: python -m pip install pyserial')
    print('Emulator on',args.port,'-- DO NOT CONNECT TO LIVE LPG ECU')
    with serial.Serial(args.port,args.baudrate,timeout=0.1) as port:
        th=threading.Thread(target=serial_worker,args=(s,port,ev),daemon=True);th.start()
        try:repl(s,ev)
        finally:ev.set();th.join(timeout=.5)
if __name__=='__main__':main()
