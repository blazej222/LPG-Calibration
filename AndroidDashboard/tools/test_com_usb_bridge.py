"""Stream relay regression, uses no Windows COM ports or ECU hardware."""
import socket
import threading
import unittest
from queue import Queue, Empty

from com_usb_bridge import relay


class FakeSerial:
    def __init__(self):
        self.incoming = Queue()
        self.outgoing = Queue()

    @property
    def in_waiting(self):
        return 0

    def read(self, count):
        try:
            return self.incoming.get(timeout=0.1)
        except Empty:
            return b""

    def write(self, data):
        # Verify that partial serial writes don't drop bytes.
        chunk = bytes(data[:2])
        self.outgoing.put(chunk)
        return len(chunk)


class BridgeTest(unittest.TestCase):
    def test_duplex_binary_stream_and_clean_disconnect(self):
        remote, local = socket.socketpair()
        remote.settimeout(2)
        port = FakeSerial()
        ready = threading.Event()
        errors = []

        def worker():
            try:
                relay(local, port, ready=ready)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(ready.wait(2))
        request = bytes.fromhex("C0 0B 0B D6 C2 01 01 6B 87")
        remote.sendall(request[:3])
        remote.sendall(request[3:])
        received = b""
        while len(received) < len(request):
            received += port.outgoing.get(timeout=2)
        self.assertEqual(received, request)
        reply = bytes.fromhex("D0 0B C0 D0 00 FF 77")
        port.incoming.put(reply[:1])
        port.incoming.put(reply[1:])
        received = b""
        while len(received) < len(reply):
            received += remote.recv(4096)
        self.assertEqual(received, reply)
        remote.close()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
