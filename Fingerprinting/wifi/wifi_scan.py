#wifi.wifi_scan
from pywifi import PyWiFi
import time


class WiFiScanner:

    def __init__(self):

        wifi = PyWiFi()

        self.iface = wifi.interfaces()[0]
    
    def _rssi_to_distance(self, rssi, tx_power=-40, path_loss_exp=2.7):
        distance = 10 ** ((tx_power - rssi) / (10 * path_loss_exp))
        return round(distance, 2)


    def scan_networks(self):

        self.iface.scan()

        time.sleep(3)

        results = self.iface.scan_results()

        networks = []

        for r in results:

            if r.ssid.strip() == "":
                continue
            
            bssid = r.bssid.upper()
            networks.append({
                "ssid": r.ssid, "bssid": bssid, "rssi": r.signal, "distance_m": self._rssi_to_distance(r.signal)
            })

        return networks
        
