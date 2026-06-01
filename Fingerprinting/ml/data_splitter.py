from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_builder import DatasetBuilder

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "fingerprints.json"
OUTPUT_DIR = BASE_DIR.parent / "data" / "splits"


def split_and_save(test_size=0.2, random_state=42):

    # Daten laden
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Gesamt-Samples: {len(data)}")

    # Feature-Vektoren bauen
    X, y, all_bssids = DatasetBuilder.build_dataset(data)

    # Aufteilen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Trainingsdaten:  {len(X_train)} Samples")
    print(f"Testdaten:       {len(X_test)} Samples")

    # Speichern
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_test.npy",  X_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy",  y_test)

    # BSSID-Liste speichern — wichtig für spätere Vorhersagen!
    with open(OUTPUT_DIR / "all_bssids.json", "w") as f:
        json.dump(all_bssids, f)

    print(f"Gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    split_and_save(test_size=0.2, random_state=42)