"""Relay a Windows test COM port to QEMU's virtual USB serial device.

Keeps the existing ECU emulator on the other end of a com0com pair unchanged.
No protocol decoding, generated ECU replies, or physical USB driver replacement.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import socket
import threading
import sys

try:
    import serial
except ImportError:
    serial = None


def relay(client, port, *, trace=False, ready=None):
    """Forward arbitrary chunks in both directions until either endpoint closes."""
    stopped = threading.Event()
    errors = []
    client.settimeout(0.1)

    def copy_to_com():
        try:
            while not stopped.is_set():
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                sent = 0
                while sent < len(data):
                    written = port.write(data[sent:])
                    if not written:
                        raise OSError("COM write made no progress")
                    sent += written
                if trace:
                    print("Android -> COM", data.hex(" ").upper(), flush=True)
        except OSError as error:
            if not stopped.is_set():
                errors.append(error)
        finally:
            stopped.set()

    def copy_from_com():
        try:
            while not stopped.is_set():
                data = port.read(port.in_waiting or 1)
                if data:
                    client.sendall(data)
                    if trace:
                        print("COM -> Android", data.hex(" ").upper(), flush=True)
        except OSError as error:
            if not stopped.is_set():
                errors.append(error)
        finally:
            stopped.set()

    readers = [threading.Thread(target=copy_to_com, name="socket-to-com"),
               threading.Thread(target=copy_from_com, name="com-to-socket")]
    try:
        for thread in readers:
            thread.start()
        if ready is not None:
            ready.set()
        while not stopped.wait(0.1):
            pass
    finally:
        stopped.set()
        try:
            client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        client.close()
        # Serial reads and writes have finite timeouts supplied by main().
        for thread in readers:
            thread.join()
    if errors:
        raise errors[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="COM19 -> QEMU USB-UART bridge for the existing Diego G3 emulator")
    parser.add_argument("--com", default="COM19", help="The client end of the virtual pair; ECU emulator stays on COM18")
    parser.add_argument("--port", type=int, default=34567, help="Local socket used by QEMU, default 34567")
    parser.add_argument("--trace", action="store_true", help="Print raw forwarded chunks")
    args = parser.parse_args(argv)
    if serial is None:
        parser.error("pyserial is missing; use the Python environment in 'GasECU Reverse Engineering/.venv'")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    try:
        with serial.Serial(args.com, 9600, timeout=0.1, write_timeout=1,
                           bytesize=8, parity="N", stopbits=1,
                           xonxoff=False, rtscts=False, dsrdtr=False) as port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind(("127.0.0.1", args.port))
                server.listen(1)
                server.settimeout(0.5)
                print(f"READY {args.com} 9600 8N1 <-> 127.0.0.1:{args.port}", flush=True)
                print("Keep your existing ECU emulator running on the other COM port. Ctrl+C stops the bridge.", flush=True)
                while True:
                    try:
                        client, address = server.accept()
                    except socket.timeout:
                        continue
                    port.reset_input_buffer()
                    print(f"{datetime.now().isoformat(timespec='seconds')} QEMU connected: {address}", flush=True)
                    try:
                        relay(client, port, trace=args.trace)
                    except OSError as error:
                        print(f"Connection ended: {error}", file=sys.stderr, flush=True)
                    print("QEMU disconnected; waiting for another connection", flush=True)
    except KeyboardInterrupt:
        print("Bridge stopped", flush=True)
        return 0
    except (OSError, serial.SerialException) as error:
        print(f"Cannot start bridge: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
