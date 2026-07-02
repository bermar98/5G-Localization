# ml/predict.py
from pathlib import Path
import json
import numpy as np
import joblib
from tensorflow.keras import models
from wifi.wifi_scan import WiFiScanner
from ml.dataset_builder import DatasetBuilder

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "models" / "trained_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
SPLITS_DIR  = BASE_DIR / "data" / "splits"


class PositionEstimator:

    def __init__(self):
        self.model   = models.load_model(MODEL_PATH)
        self.scaler  = joblib.load(SCALER_PATH)
        self.scanner = WiFiScanner()

        with open(SPLITS_DIR / "all_bssids.json") as f:
            self.all_bssids = json.load(f)

    def estimate(self):

        raw = self.scanner.scan_networks()
        if not raw:
            print("[PositionEstimator] Warnung: Leerer WiFi-Scan, Vorhersage evtl. unzuverlässig")

        known_count = sum(1 for n in raw if n["bssid"].upper().rstrip(":") in self.all_bssids)
        if known_count == 0:
            print("[PositionEstimator] Warnung: Kein bekannter Access Point im Scan gefunden")

        fingerprint = {
            network["bssid"].upper().rstrip(":"): {"rssi": network["rssi"]}
            for network in raw
        }

        vector = DatasetBuilder.build_feature_vector(
            {"fingerprint": fingerprint},
            self.all_bssids
        )

        X = self.scaler.transform(np.array([vector]))
        prediction = self.model.predict(X, verbose=0)

        return {
            "x": float(prediction[0][0]),
            "y": float(prediction[0][1])
        }