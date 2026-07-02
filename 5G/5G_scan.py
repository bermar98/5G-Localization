# wifi/5g_scan.py
import serial
import time
import re

class FiveGScanner:

    def __init__(self, port="/dev/ttyUSB2", baudrate=115200):
        # ttyUSB2 ist der AT-Port laut Dokumentation
        self.port     = port
        self.baudrate = baudrate

    def _send_at(self, command, timeout=3):
        with serial.Serial(self.port, self.baudrate, timeout=timeout) as ser:
            ser.write((command + "\r\n").encode())
            time.sleep(0.5)
            response = ser.read(ser.in_waiting).decode(errors="ignore")
            return response

    def scan_cell(self):
        response = self._send_at('AT+QENG="servingcell"')
        return self._parse_servingcell(response)

    def _parse_servingcell(self, response):
        # Beispiel-Antwort NR5G-SA:
        # +QENG: "servingcell","NOCONN","NR5G-SA","FDD",262,02,
        #         1234567,123,3450,78,27,27,-85,-12,-60,18

        result = {
            "rat":  None,   # Netztyp: NR5G-SA, NR5G-NSA, LTE
            "rsrp": None,   # Reference Signal Received Power (dBm)
            "rsrq": None,   # Reference Signal Received Quality (dB)
            "sinr": None,   # Signal to Interference Noise Ratio
            "pci":  None,   # Physical Cell ID
        }

        # NR5G (5G)
        match = re.search(
            r'"NR5G-\w+","\w+",(\d+),(\d+),\w+,\w+,(\d+),\w+,(\d+),(-?\d+),(-?\d+),(-?\d+)',
            response
        )
        if match:
            result["rat"]  = "NR5G"
            result["pci"]  = match.group(3)
            result["rsrp"] = int(match.group(5))
            result["rsrq"] = int(match.group(6))
            result["sinr"] = int(match.group(7))
            return result

        # LTE Fallback
        match = re.search(
            r'"LTE","FDD",\d+,\d+,\w+,(\d+),\d+,\d+,\d+,\d+,\d+,(-?\d+),(-?\d+),(-?\d+)',
            response
        )
        if match:
            result["rat"]  = "LTE"
            result["pci"]  = match.group(1)
            result["rsrp"] = int(match.group(2))
            result["rsrq"] = int(match.group(3))
            result["sinr"] = int(match.group(4))

        return result