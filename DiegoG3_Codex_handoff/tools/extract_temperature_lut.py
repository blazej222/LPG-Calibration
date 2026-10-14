"""Extract byte->temperature LUT from EXACT DiegoG3-3.0.8.2.exe version.
Position is EXE file offset 0x001B7744 (virtual 0x005B8B44), 256 little-endian double values indexed by (255-raw).
No execution of the EXE; the SHA256 is checked before reading.
"""
from __future__ import annotations
import argparse,csv,hashlib,struct
from pathlib import Path
EXPECTED_SHA256='f182bbf3476fdd68678670de87235329e36f4be8beda2b4c6f88cb147794bd4a'
TABLE_FILE_OFFSET=0x001B7744
COUNT=256

def extract(exe:Path)->list[float]:
    blob=exe.read_bytes()
    sha=hashlib.sha256(blob).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError(f'Niepoprawna wersja EXE lub zmodyfikowany plik: {sha}; oczekiwano {EXPECTED_SHA256}')
    if len(blob)<TABLE_FILE_OFFSET+COUNT*8:raise ValueError('Plik za krótki')
    raw_table=struct.unpack_from('<256d',blob,TABLE_FILE_OFFSET)
    return [raw_table[255-code] for code in range(256)]

def write_csv(data:list[float],output:Path):
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(['raw_hex','raw_dec','temperature_c_exact','temperature_c_rounded_1dp','table_index'])
        for i,temp in enumerate(data):
            wr.writerow([f'0x{i:02X}',i,repr(temp),f'{temp:.1f}',255-i])

def main():
    cli=argparse.ArgumentParser();cli.add_argument('exe',type=Path);cli.add_argument('--out',type=Path,default=Path('reference/temperature_lut.csv'))
    args=cli.parse_args();arr=extract(args.exe);write_csv(arr,args.out)
    print('Wrote',args.out,'entries',len(arr),'0x69',arr[0x69],'0x80',arr[0x80],'0x88',arr[0x88])
if __name__=='__main__':main()
