from wifi_scanner import WiFiScanner

scanner = WiFiScanner()
scanner.scan_networks()

print("Available Networks:")
for ssid, signal in zip(scanner.ssids, scanner.signal_strengths):
    print(f"SSID: {ssid}, Signal Strength: {signal} dBm")