import serial


def process_data(data):
    responses = {
        b"\xC0\x01\x01\xC2": b"\x01\x30",
        b"\xC0\x02\x02\xC4": b"\x02\x31\x36\x3A\x32\x36\x3A\x31\x31\x4D\x61\x72\x20\x31\x32\x20\x32\x30\x31\x32",

        b"\xC0\x53\x00\x13": b"\x53\x4A",
        b"\xC0\x53\x01\x14": b"\x53\x04",
        b"\xC0\x53\x02\x15": b"\x53\x00",
        b"\xC0\x53\x03\x16": b"\x53\x04",
        b"\xC0\x53\x04\x17": b"\x53\x02",
        b"\xC0\x53\x05\x18": b"\x53\x00",
        b"\xC0\x53\x06\x19": b"\x53\xFF",

        b"\xC0\x40\x1A\x1A": b"\x40\x00\x5B",
        b"\xC0\x3F\x00\xFF": b"\x3F\xF0",

        b"\xC0\x07\x22\xE9": b"\x07\x22\x9C",
        b"\xC0\x07\x21\xE8": b"\x07\x21\x03",

        b"\xC0\x40\x1C\x1C": b"\x40\x00\x01",

        b"\xC0\xD8\xD8\x70": b"\xD8\x03\x0F\x80\x0A\x72\xDB\x24\x1E\x07\xCD\x00\x82\x01\xF9\x01\x00\x1C\x01\xF4\x01"
                             b"\xF4\x04\x39\x96\x03",

        b"\xC0\x35\x00\xF5": b"\x35\xC8\x8F\x5C\x28",

        b"\xC0\x36\x00\xF6": b"\x36\x40\x4E\x5B\x68",

        b"\xC0\x07\x23\xEA": b"\x07\x23\xC3",

        b"\xC0\x51\x0B\x1C": b"\x51\x0B\x01\x90",
        b"\xC0\x51\x0C\x1D": b"\x51\x0C\x1F\x40",

        b"\xC0\x07\x1F\xE6": b"\x07\x1F\x20",
        b"\xC0\x07\x1D\xE4": b"\x07\x1D\x5A",

        b"\xC0\x51\x0E\x1F": b"\x51\x0E\x00\x07",

        b"\xC0\x07\x20\xE7": b"\x07\x20\x01",
        b"\xC0\x07\x25\xEC": b"\x07\x25\x41",

        b"\xC0\x12\x4A\x1C": b"\x12\x4A\x00\x03",

        b"\xC0\x07\x2A\xF1": b"\x07\x2A\x9C",
        b"\xC0\x07\x28\xEF": b"\x07\x28\x01",

        b"\xC0\x51\x10\x21": b"\x51\x10\x02\x31",

        b"\xC0\x07\x26\xED": b"\x07\x26\x4D",
        b"\xC0\x07\x27\xEE": b"\x07\x27\x32",
        b"\xC0\x07\x13\xDA": b"\x07\x13\x03",
        b"\xC0\x07\x2B\xF2": b"\x07\x2B\x32",
        b"\xC0\x07\x2C\xF3": b"\x07\x2C\xE2",

        b"\xC0\x93\x00\x53": b"\x93\x00\x00\x00\x00\x00\x00\x00\x00",

        b"\xC0\x07\x2D\xF4": b"\x07\x2D\x02",
        b"\xC0\x07\x2E\xF5": b"\x07\x2E\x02",

        b"\xC0\x51\x12\x23": b"\x51\x12\x09\xC4",
        b"\xC0\x51\x13\x24": b"\x51\x13\x02\xEC",

        b"\xC0\x07\x31\xF8": b"\x07\x31\x00",
        b"\xC0\x07\x32\xF9": b"\x07\x32\x00",

        b"\xC0\x51\x16\x27": b"\x51\x16\x12\x43",
        b"\xC0\x56\x00\x16": b"\x56\x3C\x27\x10",
        b"\xC0\x54\x00\x14": b"\x54\x3B\xC5\x15\x07\x7E\x91\x92\xD4\x3B\xC5\x15\x07\x7E\x8B\xA4\x34\x3B\xC5\x15"
                             b"\x07\x7E\x8C\xA3\x89\x3B\xC5\x15\x07\x7E\x8C\xDA\xC7\x3B\xC5\x15\x07\x7E\x8C\xED\x33"
                             b"\x00",

        b"\xC0\x07\x34\xFB": b"\x07\x34\x00",
        b"\xC0\x07\x2F\xF6": b"\x07\x2F\x00",

        b"\xC0\x51\x11\x22": b"\x51\x11\x00\x00",
        b"\xC0\x51\x18\x29": b"\x51\x18\x01\xF4",
        b"\xC0\x51\x19\x2A": b"\x51\x19\xFF\xFF",
        b"\xC0\x51\x14\x25": b"\x51\x14\x01\x76",
        b"\xC0\x51\x15\x26": b"\x51\x15\x0E\x9C",
        b"\xC0\x40\x28\x28": b"\x40\x00\x00",
        # b"\xBC\xE1\xE1\x7E": b"\xBC\xE1\xE1\x7E"
        b"\xC0\x0B\x0B\xD6": b"\x0B"
                             b"\x79"  # Air pressure
                             b"\x69"  # Tred
                             b"\x93"  # Tgas
                             b"\x9B\x6B\x17"  # Unknown
                             b"\x65\xBA"  # RPM
                             b"\xC8\x00\xA8\x00"  # Benzin injection time injector 1/2
                             b"\x28\x00\x28\x00" # Benzin injection time injector 3/4
                             b"\x00\x00\x00\x00" # Gas injection time injector 1/2
                             b"\x00\x00\x00\x00" # Gas injection time injector 3/4
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x17\x72\x00\x00\xFF\x01\x88\x00\x02",


        b"\xC0\x40\x29\x29": b"\x40\x00\x05",

        b"\xC0\x64\xA5\xC9": b"\x64\x03\x6A\x0D\x60\x7A\x6B\x18\xB8\x00\x00\x00",

        b"\xC0\x40\x0D\x0D": b"\x40\x00\x8E",
        b"\xC0\x40\x0F\x0F": b"\x40\x00\x1A",

        b"\xC0\x07\x35\xFC": b"\x07\x35\x3C",
        b"\xC0\x10\x10\xE0": b"\x10\x04\x19\x00\xF3\x03\xA7\x07\x4E\x0A\xF5\x0E\x9C\x10\x00\x10\x00\x10\x00\x10\x00",
        b"\xC0\x13\x13\xE6": b"\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",

        b"\xC0\x95\x00\x55": b"\x95\x00\x00\x00\xF3\x01\xB6\x03\x1A\x01\xF9\x03\x1B\x02\xD9\x04\x8B\x04\x4F\x06"
                             b"\x13\x05\x8D\x07\x1F\x06\xA6\x08\x06\x08\x09\x09\x12\x0A\x98\x0B\x76\x12\x43\x11\x6C",

        b"\xC0\x95\x01\x56": b"\x95\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF"
                             b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF",

        b"\xC0\x07\x1E\xE5": b"\x07\x1E\x68",

        b"\xC0\xD4\x00\x94": b"\xD4\x00\x00\x00\xF3\x01\xB6\x03\x1A\x01\xF9\x03\x1B\x02\xD9\x04\x8B\x04\x4F\x06"
                             b"\x13\x05\x8D\x07\x1F\x06\xA6\x08\x06\x08\x09\x09\x12\x0A\x98\x0B\x76\x12\x43\x11\x6C",

        b"\xC0\xD4\x01\x95": b"\xD4\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF"
                             b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF",


        b"\xC0\x12\x01\xD3": b"\x12\x01\x04\x19",
        b"\xC0\x12\x02\xD4": b"\x12\x02\x00\xF3",
        b"\xC0\x12\x0B\xDD": b"\x12\x0B\x04\x00",
        b"\xC0\x12\x0C\xDE": b"\x12\x0C\x00\x00",
        b"\xC0\x12\x0D\xDF": b"\x12\x0D\x00\x00",

        b"\xC0\xC0\x00\x80": b"\xC0\x01\xF9\x02\xEC\x02\xFF\x04\xE5",
        b"\xC0\xD0\x00\x90": b"\xD0\x66\x50\x3C\x2F\x21\x17\x0F\x09\x00\xFA\xF6\xF4\xF0\xEE\xEA\xE8\xE6\xE4\xE1\xE0",
        b"\xC0\xD2\x00\x92": b"\xD2\xF2\xF4\xF6\xF7\xF8\xFB\xFC\xFE\xFF\x00\x01\x01\x01\x02\x02\x02\x03\x03\x04\x05",

        b"\xC0\xDB\x00\x9B": b"\xDB\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x08\x0A\x0B\x0C\x0D\x0E\x0F\x10\x11",
        b"\xC0\x97\x00\x57": b"\x97\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",

        b"\xC0\x60\xA5\xC5": b"\x60\x0A\x14\x1E\x28\x32\x3C\x46\x50\x5A\x64\x01\xF4\x03\xE8\x05\xDC\x07\xD0\x09\xC4"
                             b"\x0B\xB8\x0D\xAC\x0F\xA0\x11\x94\x13\x88",

        b"\xC0\x61\x10\x31": b"\x61\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\xFB\x05\x0F\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",

        b"\xC0\x61\x20\x41": b"\x61\x20\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",

        b"\xC0\x61\x11\x32": b"\x61\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",

        b"\xC0\x61\x21\x42": b"\x61\x21\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                             b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",

        b"\xC0\x12\x68\x3A": b"\x12\x68\x00\x00",
        # b"\xC0\x40\x0D\x0D": b"\x40\x00\x64\x74",
        # b"\xC0\x40\x0F\x0F": b"\x40\x00\x17\x27"
        }
    return responses.get(data, b'')


def calculate_outgoing_crc(bytearr):
    crc = 0
    for byte in bytearr:
        crc = (crc + byte) & 0xFF
    return (crc - 0x30) & 0xFF


if __name__ == "__main__":
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.timeout = 0
    ser.port = "COM11"
    ser.open()

    to_send = []

    # input_buffer = []

    while True:
        data = ser.read(ser.in_waiting)
        length = len(data)
        # if length > 0:
        #     print(length)

        if length == 0:
            continue

        # print("Processing data")
        response = process_data(data)
        if response:
            # print("We got this")
            ser.write(bytes([0xD0]))
            crc = calculate_outgoing_crc(data)
            # print(crc)
            response = response + calculate_outgoing_crc(response).to_bytes()
            # print(response)
            ser.write(response)
        else:
            print(f"Unknown data: {data.hex().upper()}")
