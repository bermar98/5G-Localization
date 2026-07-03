#ml.data_splitter
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_builder import DatasetBuilder

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "fingerprints.json"
SPLITS_DIR = BASE_DIR.parent / "data" / "splits"


def split_and_save(data, test_size=0.2, random_state=42, use_presence_feature=True):


    # Feature-Vektoren bauen
    X, y, all_bssids = DatasetBuilder.build_dataset(data, use_presence_feature=use_presence_feature)

    # Aufteilen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Trainingsdaten:  {len(X_train)} Samples")
    print(f"Testdaten:       {len(X_test)} Samples")
    print(f"Feature-Dimension: {X_train.shape[1]} "
          f"({'mit' if use_presence_feature else 'ohne'} Presence-Feature, "
          f"{len(all_bssids)} BSSIDs)")
    
    # Speichern
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(SPLITS_DIR / "X_train.npy", X_train)
    np.save(SPLITS_DIR / "X_test.npy",  X_test)
    np.save(SPLITS_DIR / "y_train.npy", y_train)
    np.save(SPLITS_DIR / "y_test.npy",  y_test)

    # BSSID-Liste speichern — wichtig für spätere Vorhersagen!
    with open(SPLITS_DIR / "all_bssids.json", "w") as f:
        json.dump(all_bssids, f)
     
    # Feature-Konfiguration speichern — wichtig, damit predict.py exakt
    # denselben Feature-Vektor-Aufbau nutzt wie beim Training!    
    feature_config = {
        "use_presence_feature": use_presence_feature,
        "sentinel_rssi": DatasetBuilder.SENTINEL_RSSI,
        "bssid_count": len(all_bssids),
        "feature_dim": X_train.shape[1],
    }
    with open(SPLITS_DIR / "feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)

    print(f"Gespeichert in: {SPLITS_DIR}")

    return X_train, X_test, y_train, y_test, all_bssids

