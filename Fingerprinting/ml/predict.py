# ml/predict.py
from pathlib import Path
import json
import numpy as np
import joblib
from tensorflow.keras import models
from wifi.wifi_scan_linux import WiFiScanner
from ml.dataset_builder import DatasetBuilder

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "models" / "trained_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURE_CONFIG_PATH = BASE_DIR / "models" / "feature_config.json"
SPLITS_DIR  = BASE_DIR / "data" / "splits"


class PositionEstimator:

    def __init__(self):
        self.model   = models.load_model(MODEL_PATH)
        self.scaler  = joblib.load(SCALER_PATH)
        self.scanner = WiFiScanner()

        with open(SPLITS_DIR / "all_bssids.json") as f:
            self.all_bssids = json.load(f)
            
            # Feature-Konfiguration laden, damit der Vektor-Aufbau exakt dem
        # beim Training verwendeten entspricht (z.B. ob ein Presence-
        # Feature mit eingebaut wurde). Fällt auf "mit Presence-Feature"
        # zurück, falls die Datei aus einem älteren Trainingsstand fehlt.
        if FEATURE_CONFIG_PATH.exists():
            with open(FEATURE_CONFIG_PATH) as f:
                feature_config = json.load(f)
        else:
            print(f"[PositionEstimator] Warnung: {FEATURE_CONFIG_PATH} nicht "
                  f"gefunden, nehme Default-Konfiguration an.")
            feature_config = {"use_presence_feature": True}
 
        self.use_presence_feature = feature_config.get("use_presence_feature", True)

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
            self.all_bssids,
            use_presence_feature=self.use_presence_feature
        )

        X = self.scaler.transform(np.array([vector]))
        prediction = self.model.predict(X, verbose=0)

        return {
            "x": float(prediction[0][0]),
            "y": float(prediction[0][1])
        }