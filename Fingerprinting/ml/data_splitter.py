#ml.data_splitter
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_builder import DatasetBuilder

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "fingerprints.json"
SPLITS_DIR = BASE_DIR.parent / "data" / "splits"


def split_and_save(data, test_size=0.2, random_state=42):


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
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(SPLITS_DIR / "X_train.npy", X_train)
    np.save(SPLITS_DIR / "X_test.npy",  X_test)
    np.save(SPLITS_DIR / "y_train.npy", y_train)
    np.save(SPLITS_DIR / "y_test.npy",  y_test)

    # BSSID-Liste speichern — wichtig für spätere Vorhersagen!
    with open(SPLITS_DIR / "all_bssids.json", "w") as f:
        json.dump(all_bssids, f)

    print(f"Gespeichert in: {SPLITS_DIR}")

    return X_train, X_test, y_train, y_test, all_bssids

