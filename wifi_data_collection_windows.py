import subprocess
import time
import csv
import math
import threading

import re
import os

class Collection:

    def __init__(self):
        self.collecting = False

        self.timestamps = []
        self.wifi_data = []

        self.mac_list = []
        self.mac_list_len = 0

    def keyboard_listener(self):
        while True:
            cmd = input("ENTER = Start/Stop | q = Beenden: ").lower()

            if cmd == "":
                self.collecting = not self.collecting

                if self.collecting:
                    print("Messung gestartet...")
                else:
                    print("Messung gestoppt.")

            elif cmd == "q":
                self.collecting = False
                break

    def scan_wifi(self):
        result = subprocess.check_output(
            "netsh wlan show networks mode=bssid",
            shell=True,
            encoding="utf-8",
            errors="ignore"
        )
        return result

    def extract_data(self, raw_text):

        addresses = re.findall(r"([0-9A-Fa-f:]{17})", raw_text)

        signals_raw = re.findall(r"Signal\s*:\s*(\d+)%", raw_text)

        signals = []

        for s in signals_raw:
            percent = int(s)

            # Umrechnung grob in dBm
            dbm = (percent / 2) - 100
            signals.append(int(dbm))

        return addresses, signals

    def collect(self):

        listener = threading.Thread(target=self.keyboard_listener)
        listener.daemon = True
        listener.start()

        print("WLAN Logger bereit.")

        while listener.is_alive():

            if self.collecting:

                raw = self.scan_wifi()

                addresses, signals = self.extract_data(raw)

                timestamp = time.time()

                self.timestamps.append(timestamp)
                self.wifi_data.append([addresses, signals])

                for mac in addresses:
                    if mac not in self.mac_list:
                        self.mac_list.append(mac)
                        self.mac_list_len += 1

                print("Messung", len(self.timestamps), "gespeichert")

                time.sleep(2)

            else:
                time.sleep(0.2)

    def make_csv(self, file_path):

        file_name = os.path.join(file_path, "wifi_data.csv")

        with open(file_name, "w", newline="") as f:

            writer = csv.writer(f)

            header = ["timestamp"] + self.mac_list
            writer.writerow(header)

            for idx, entry in enumerate(self.wifi_data):

                line = [math.nan] * (self.mac_list_len + 1)
                line[0] = self.timestamps[idx]

                addresses = entry[0]
                signals = entry[1]

                for i in range(self.mac_list_len):

                    mac = self.mac_list[i]

                    if mac in addresses:
                        pos = addresses.index(mac)
                        line[i + 1] = signals[pos]

                writer.writerow(line)

        print("CSV gespeichert:")
        print(file_name)


if __name__ == "__main__":

    file_path = r"C:\Dokumente\Studium\Master\Masterarbeit\Code"

    a = Collection()
    a.collect()
    a.make_csv(file_path)