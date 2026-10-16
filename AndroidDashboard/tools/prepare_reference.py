"""Generate Android references from immutable source data; never changes handoff fixtures."""
import ast
import csv
import hashlib
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "AndroidDashboard/app/src"
OUT = APP / "main/java/com/blazej222/lpgdashboard/protocol"
REFERENCE = ROOT / "DiegoG3_Codex_handoff/reference"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (REFERENCE / "temperature_lut.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    values = {int(row["raw_dec"]): row["temperature_c_exact"] for row in rows}
    assert set(values) == set(range(256))
    lines = [", ".join(values[i] for i in range(start, start + 8)) for start in range(0, 256, 8)]
    (OUT / "TemperatureTable.kt").write_text(
        'package com.blazej222.lpgdashboard.protocol\n\n'
        '// Generated from reference/temperature_lut.csv; EXE-derived doubles, RAW order.\n'
        'object TemperatureTable {\n    val values = doubleArrayOf(\n        '
        + ',\n        '.join(lines) + '\n    )\n}\n', encoding="utf-8")

    log = ROOT / "GasECU Reverse Engineering/data/PORTMAN.LOG"
    rows = log.read_text().splitlines()
    startup = []
    for index, row in enumerate(rows):
        if "IRP_MJ_WRITE" not in row:
            continue
        match = re.search(r"Length (\d+):\s*((?:[0-9A-F]{2}\s*)+)", row)
        if not match:
            continue
        wire = bytes.fromhex(match[2])
        assert len(wire) == int(match[1])
        if wire == bytes.fromhex("C0 0B 0B D6"):
            break
        length = 0
        for reply in rows[index + 1:]:
            if "IRP_MJ_WRITE" in reply:
                break
            if "IRP_MJ_READ" in reply and "SUCCESS" in reply:
                count = re.search(r"Length (\d+):", reply)
                if count:
                    length += int(count[1])
        startup.append((wire, max(0, length - 2), index + 1))
    assert len(startup) == 77
    module = ast.parse((ROOT / "GasECU Reverse Engineering/src/emulator.py").read_text())
    dictionary = next(node.value for node in ast.walk(module) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "responses" for t in node.targets))
    emulator = ast.literal_eval(dictionary)
    for wire, length, _ in startup:
        assert len(emulator[wire]) == length, wire.hex()
    commands = []
    for wire, size, line in startup:
        if wire[0] == 0xC0:
            commands.append(f"        StartupCommand(DiegoProtocol.request(0x{wire[1]:02X}, 0x{wire[2]:02X}), {size}), // line {line}")
        else:
            commands.append(f'        StartupCommand(byteArrayOf(0xC2.toByte(), 0x01, 0x01, 0x6B, 0x87.toByte()), 0, 450), // line {line}; observed timeout')
    (OUT / "StartupSequence.kt").write_text(
        'package com.blazej222.lpgdashboard.protocol\n\n'
        f'// Generated from PORTMAN.LOG (SHA256 {hashlib.sha256(log.read_bytes()).hexdigest()}).\n'
        '// Captured requests before first 0B. Semantics of C2 remain unknown.\n'
        'object StartupSequence {\n    val commands = listOf(\n'
        + '\n'.join(commands) + '\n    )\n}\n', encoding="utf-8")
    fixtures = APP / "test/resources"
    fixtures.mkdir(parents=True, exist_ok=True)
    for file in REFERENCE.glob("*.csv"):
        shutil.copyfile(file, fixtures / file.name)
    (fixtures / "startup.tsv").write_text('\n'.join(
        f'{line}\t{wire.hex(" ").upper()}\t{size}' for wire, size, line in startup) + '\n', encoding="utf-8")
    (fixtures / "startup_replies.tsv").write_text('\n'.join(
        f'{wire.hex(" ").upper()}\t{emulator[wire].hex(" ").upper()}' for wire, size, _ in startup if size) + '\n', encoding="utf-8")
    (fixtures / "baseline.hex").write_text(emulator[bytes.fromhex("C0 0B 0B D6")].hex(" ").upper(), encoding="utf-8")
    print(f"Generated 256 LUT entries and {len(startup)} startup commands; emulator lengths match capture.")


if __name__ == "__main__":
    main()
