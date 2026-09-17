# ml/predict.py
from pathlib import Path
import json
import numpy as np
import joblib
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
from ml.dataset_builder_Zeitstempel import DatasetBuilder

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "models" / "modell_Zeitstempel.tflite"
SCALER_PATH = BASE_DIR / "models" / "scaler_Zeitstempel.pkl"
SPLITS_DIR  = BASE_DIR / "data" / "splits"


class PositionEstimator:

    def __init__(self):
        self.interpreter = Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.scaler  = joblib.load(SCALER_PATH)

        with open(SPLITS_DIR / "all_bssids.json") as f:
            self.all_bssids = json.load(f)

    def estimate(self, fingerprint, timestamp):
        """
        Schätzt die Position anhand eines vorgegebenen Fingerprints und
        Zeitstempels.

        fingerprint: dict im Format
            { "BSSID": {"rssi": <int>, ...}, ... }
            (z. B. ein Eintrag aus fingerprints_langsame_fahrt.json)
        timestamp: String im Format "YYYY-MM-DD HH:MM:SS", muss zum
            Trainings-Format in DatasetBuilder passen.
        """
        if not fingerprint:
            print("[PositionEstimator] Warnung: Leeres Fingerprint, Vorhersage evtl. unzuverlässig")

        # Keys auf das erwartete Format normalisieren (Großschreibung, ohne trailing ':')
        fingerprint = {
            bssid.upper().rstrip(":"): data
            for bssid, data in fingerprint.items()
        }

        known_count = sum(1 for bssid in fingerprint if bssid in self.all_bssids)
        if known_count == 0:
            print("[PositionEstimator] Warnung: Kein bekannter Access Point im Fingerprint gefunden")

        vector = DatasetBuilder.build_feature_vector(
            {"fingerprint": fingerprint, "timestamp": timestamp},
            self.all_bssids
        )

        X = self.scaler.transform(np.array([vector])).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

        return {
            "x": float(prediction[0][0]),
            "y": float(prediction[0][1])
        }
