"""Build a JSON catalog from the 2026-10-14 historical Python response map.
Static replies are emulator samples, not a full specification of ECU behavior.
"""
from __future__ import annotations
import json
from pathlib import Path

# Each pair: request hex / reply payload (without D0 and checksum); NO RESPONSE is null.
# Values transcribed from historical process_data() supplied by user.
rows = [
 ('C0 01 01 C2','01 30'),
 ('C0 02 02 C4','02 31 36 3A 32 36 3A 31 31 4D 61 72 20 31 32 20 32 30 31 32'),
 ('C0 53 00 13','53 4A'),
 ('C0 53 01 14','53 04'),
 ('C0 53 02 15','53 00'),
 ('C0 53 03 16','53 04'),
 ('C0 53 04 17','53 02'),
 ('C0 53 05 18','53 00'),
 ('C0 53 06 19','53 FF'),
 ('C0 40 1A 1A','40 00 5B'),
 ('C0 3F 00 FF','3F F0'),
 ('C0 07 22 E9','07 22 9C'),
 ('C0 07 21 E8','07 21 03'),
 ('C0 40 1C 1C','40 00 01'),
 ('C0 D8 D8 70','D8 03 0F 80 0A 72 DB 24 1E 07 CD 00 82 01 F9 01 00 1C 01 F4 01 F4 04 39 96 03'),
 ('C0 35 00 F5','35 C8 8F 5C 28'),
 ('C0 36 00 F6','36 40 4E 5B 68'),
 ('C0 07 23 EA','07 23 C3'),
 ('C0 51 0B 1C','51 0B 01 90'),
 ('C0 51 0C 1D','51 0C 1F 40'),
 ('C0 07 1F E6','07 1F 20'),
 ('C0 07 1D E4','07 1D 5A'),
 ('C0 51 0E 1F','51 0E 00 07'),
 ('C0 07 20 E7','07 20 01'),
 ('C0 07 25 EC','07 25 41'),
 ('C0 12 4A 1C','12 4A 00 03'),
 ('C0 07 2A F1','07 2A 9C'),
 ('C0 07 28 EF','07 28 01'),
 ('C0 51 10 21','51 10 02 31'),
 ('C0 07 26 ED','07 26 4D'),
 ('C0 07 27 EE','07 27 32'),
 ('C0 07 13 DA','07 13 03'),
 ('C0 07 2B F2','07 2B 32'),
 ('C0 07 2C F3','07 2C E2'),
 ('C0 93 00 53','93 00 00 00 00 00 00 00 00'),
 ('C0 07 2D F4','07 2D 02'),
 ('C0 07 2E F5','07 2E 02'),
 ('C0 51 12 23','51 12 09 C4'),
 ('C0 51 13 24','51 13 02 EC'),
 ('C0 07 31 F8','07 31 00'),
 ('C0 07 32 F9','07 32 00'),
 ('C0 51 16 27','51 16 12 43'),
 ('C0 56 00 16','56 3C 27 10'),
 ('C0 54 00 14','54 3B C5 15 07 7E 91 92 D4 3B C5 15 07 7E 8B A4 34 3B C5 15 07 7E 8C A3 89 3B C5 15 07 7E 8C DA C7 3B C5 15 07 7E 8C ED 33 00'),
 ('C0 07 34 FB','07 34 00'),
 ('C0 07 2F F6','07 2F 00'),
 ('C0 51 11 22','51 11 00 00'),
 ('C0 51 18 29','51 18 01 F4'),
 ('C0 51 19 2A','51 19 FF FF'),
 ('C0 51 14 25','51 14 01 76'),
 ('C0 51 15 26','51 15 0E 9C'),
 ('C0 40 28 28','40 00 00'),
 ('C0 07 18 DF','07 18 00'),  # synthetic, explicitly unmeasured
 ('C0 0B 0B D6',None),  # filled with sample from measured CSV
 ('C0 40 29 29','40 00 05'),
 ('C0 64 A5 C9','64 03 6A 0D 60 7A 6B 18 B8 00 00 00'),
 ('C0 40 0D 0D','40 00 8E'),
 ('C0 40 0F 0F','40 00 1A'),
 ('C0 07 35 FC','07 35 3C'),
 ('C0 10 10 E0','10 04 19 00 F3 03 A7 07 4E 0A F5 0E 9C 10 00 10 00 10 00 10 00'),
 ('C0 13 13 E6','13 ' + '00 '*16),
 ('C0 95 00 55','95 00 00 00 F3 01 B6 03 1A 01 F9 03 1B 02 D9 04 8B 04 4F 06 13 05 8D 07 1F 06 A6 08 06 08 09 09 12 0A 98 0B 76 12 43 11 6C'),
 ('C0 95 01 56','95 ' + 'FF '*39),
 ('C0 07 1E E5','07 1E 68'),
 ('C0 D4 00 94','D4 00 00 00 F3 01 B6 03 1A 01 F9 03 1B 02 D9 04 8B 04 4F 06 13 05 8D 07 1F 06 A6 08 06 08 09 09 12 0A 98 0B 76 12 43 11 6C'),
 ('C0 D4 01 95','D4 ' + 'FF '*39),
 ('C0 12 01 D3','12 01 04 19'),
 ('C0 12 02 D4','12 02 00 F3'),
 ('C0 12 0B DD','12 0B 04 00'),
 ('C0 12 0C DE','12 0C 00 00'),
 ('C0 12 0D DF','12 0D 00 00'),
 ('C0 C0 00 80','C0 01 F9 02 EC 02 FF 04 E5'),
 ('C0 D0 00 90','D0 66 50 3C 2F 21 17 0F 09 00 FA F6 F4 F0 EE EA E8 E6 E4 E1 E0'),
 ('C0 D2 00 92','D2 F2 F4 F6 F7 F8 FB FC FE FF 00 01 01 01 02 02 02 03 03 04 05'),
 ('C0 DB 00 9B','DB 00 00 00 00 00 00 01 02 03 04 05 08 0A 0B 0C 0D 0E 0F 10 11'),
 ('C0 97 00 57','97 ' + '00 '*20),
 ('C0 60 A5 C5','60 0A 14 1E 28 32 3C 46 50 5A 64 01 F4 03 E8 05 DC 07 D0 09 C4 0B B8 0D AC 0F A0 11 94 13 88'),
 ('C0 61 10 31','61 10 '+'00 '*26+'FB 05 0F '+'00 '*19),
 ('C0 61 20 41','61 20 '+'00 '*48),
 ('C0 61 11 32','61 11 '+'00 '*48),
 ('C0 61 21 42','61 21 '+'00 '*48),
 ('C0 12 68 3A','12 68 00 00'),
 ('C2 01 01 6B 87',''),
]

# The 0B payload is taken verbatim from captured emulator test CSV (not reconstructed by eyeballing).
import csv
with (Path(__file__).resolve().parents[1] / 'reference' / 'DiegoG3_measurements.csv').open(encoding='utf-8-sig',newline='') as f:
    sample=bytes.fromhex(next(csv.DictReader(f))['payload_hex'])
assert len(sample)==50 and sample[0]==0x0B
rows = [(request, sample.hex(' ').upper() if request=='C0 0B 0B D6' else response) for request,response in rows]

catalog=[]
seen=set()
for cmd,payload in rows:
    request=bytes.fromhex(cmd)
    if len(request)!= (5 if request[0]==0xC2 else 4):
        raise ValueError(('bad request length',cmd))
    if request[0]==0xC0 and sum(request[:3])%256!=request[3]:
        raise ValueError(('bad request checksum',cmd))
    if cmd in seen:raise ValueError(('duplicate request',cmd))
    seen.add(cmd)
    if payload is None:
        raise ValueError(('missing payload',cmd))
    p=bytes.fromhex(payload)
    special=(request[0]==0xC2)
    tag=('experimental_unmeasured' if cmd=='C0 07 18 DF' else
         'special_no_reply' if special else 'emulator_snapshot')
    catalog.append({
      'request_hex':request.hex(' ').upper(),
      'request_len':len(request),
      'payload_hex':p.hex(' ').upper(),
      'payload_len':len(p),
      'wire_reply_hex': None if special else (bytes([0xD0])+p+bytes([(sum(p)-0x30)&255])).hex(' ').upper(),
      'wire_reply_len':0 if special else len(p)+2,
      'payload_checksum_hex':None if special else f'{(sum(p)-0x30)&255:02X}',
      'provenance':tag,
    })
output=Path(__file__).resolve().parents[1]/'reference'/'known_commands.json'
output.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('commands',len(catalog),'catalog:',output,'long payloads:',[(e['request_hex'],e['payload_len']) for e in catalog if e['payload_len']>30]);print('wire known',sum(e['wire_reply_len']>0 for e in catalog))
