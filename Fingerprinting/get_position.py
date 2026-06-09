# get_position.py
from pathlib import Path
import json
import numpy as np
from tensorflow.keras import models
from ml.predict import Predictor
from wifi.wifi_scan import WiFiScanner
import joblib


BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.keras"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

def predict_from_scan():
    
    # 1. Modell laden
    model = models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. BSSID-Liste laden
    with open(SPLITS_DIR / "all_bssids.json") as f:
        all_bssids = json.load(f)

    # 3. WLAN-Messung
    scanner = WiFiScanner()
    raw = scanner.scan_networks()  # gibt Liste zurück

    # 4. Liste → Dict umwandeln (so wie in deiner fingerprints.json)
    fingerprint = {
        network["bssid"].upper().rstrip(":"): {"rssi": network["rssi"]}
        for network in raw
    }
     # Debug — nach Schritt 4 einfügen
    matches = sum(1 for bssid in all_bssids if bssid in fingerprint)
    print(f"Übereinstimmende BSSIDs: {matches} von {len(all_bssids)}")
    # Vektor direkt bauen — ohne Predictor
    vector = []
    for bssid in all_bssids:
        if bssid in fingerprint:
            vector.append(fingerprint[bssid]["rssi"])
        else:
            vector.append(-100)
    X = scaler.transform(np.array([vector]))  # ← normalisieren
    
    # 5. Vorhersage
    prediction = model.predict(X)
    x = float(prediction[0][0])
    y = float(prediction[0][1])

    print(f"Geschätzte Position: x={x:.2f}m, y={y:.2f}m")
    return {"x": x, "y": y}

if __name__ == "__main__":
    predict_from_scan()