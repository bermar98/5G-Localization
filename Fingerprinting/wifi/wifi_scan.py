from pywifi import PyWiFi
import time


class WiFiScanner:

    def __init__(self):

        wifi = PyWiFi()

        self.iface = wifi.interfaces()[0]

    def scan_networks(self):

        self.iface.scan()

        time.sleep(3)

        results = self.iface.scan_results()

        networks = []

        for r in results:

            if r.ssid.strip() == "":
                continue

            networks.append({
                "ssid": r.ssid, "bssid": r.bssid, "rssi": r.signal
            })

        return networks
        
