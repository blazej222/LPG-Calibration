"""Transcription of temperature tests described in user conversation; no raw CSV was uploaded for these.
Report 'historical_conversation_transcription' to avoid confusing with 3 machine-generated CSVs.
"""
from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1]
raw_points={
 0x10:99.7,0x20:76.1,0x30:61.9,0x40:51.9,0x50:43.9,0x60:36.9,
 0x61:36.7,0x65:35.0,0x69:33.5,0x6D:31.9,0x71:30.5,
 0x80:24.9,0x90:19.3,0xA0:13.7,0xB0:7.9,0xC0:1.4,0xD0:-5.5,0xE0:-14.1,0xF0:-27.9,
 0x18:86.1,0x38:56.7,0x58:40.3,0x88:22.0,0xB8:4.8,0xD8:-9.6,0xE8:-20.1,
 0x11:97.7,0x12:95.8,0x13:93.8,0x14:91.9,0x15:90.9,0x16:89.0,0x17:87.1,
 0x19:84.3,0x1A:83.3,0x1B:81.5,0x1C:80.6,0x1D:79.7,0x1E:77.9,0x1F:77.0,
 0x78:27.9,0x79:27.4,0x7A:27.0,0x7B:26.7,0x7C:26.3,0x7D:26.1,
 0x7E:25.6,0x7F:25.4,0x81:24.5,0x82:24.2,0x83:23.8,
 0x84:23.6,0x85:23.1,0x86:22.9,0x87:22.4,
}
out=root/'reference'/'tred_measurements_from_conversation.csv'
with out.open('w',newline='',encoding='utf-8') as f:
 wr=csv.writer(f);wr.writerow(['raw_hex','raw_dec','gui_tred_c','source'])
 for raw,temp in sorted(raw_points.items()):wr.writerow([f'{raw:02X}',raw,f'{temp:.1f}','historical_conversation_transcription'])
print('saved',out,'unique points',len(raw_points))
