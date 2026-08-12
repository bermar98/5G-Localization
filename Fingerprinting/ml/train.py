#ml.train.py
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset_builder import DatasetBuilder
from data_splitter import split_and_save
from model import IndoorLocalizationModel

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "fingerprints.json"
MODEL_PATH = BASE_DIR.parent / "models" / "trained_model.keras"
SCALER_PATH = BASE_DIR.parent / "models" / "scaler.pkl"
HISTORY_PATH = BASE_DIR.parent / "models" / "history.json"
FEATURE_CONFIG_PATH = BASE_DIR.parent / "models" / "feature_config.json"
SPLITS_DIR = BASE_DIR.parent / "data" / "splits"

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor='val_loss')
]
USE_PRESENCE_FEATURE = True

def main():

    # 1. Daten lesen
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 2. Split auslagern an data_splitter
    X_train, X_test, y_train, y_test, all_bssids = split_and_save(data, use_presence_feature=USE_PRESENCE_FEATURE)

    # Normalisierung
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Scaler speichern
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    
    
    with open(SPLITS_DIR / "feature_config.json", "r") as f:
        feature_config = json.load(f)
    feature_config["scaler_type"] = "MinMaxScaler"
    with open(FEATURE_CONFIG_PATH, "w") as f:
        json.dump(feature_config, f, indent=2)
    # 3. Modell trainieren
    model = IndoorLocalizationModel.create_model(
        X_train.shape[1]
    )

    model.summary()

    history = model.fit(X_train, y_train,
              epochs=100,
              batch_size=16,
              validation_data=(X_test, y_test),
              callbacks=callbacks)
 
    # History für Visualisierung speichern (Loss/MAE je Epoche)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history.history, f)

    model.save(MODEL_PATH)
    predictions = model.predict(X_test)
    distances = np.sqrt(
        (predictions[:, 0] - y_test[:, 0])**2 +
        (predictions[:, 1] - y_test[:, 1])**2
    )
    print(f"Mittlerer Fehler: {distances.mean():.2f}m")
    print(f"Max. Fehler:      {distances.max():.2f}m")
    print(f"Min. Fehler:      {distances.min():.2f}m")
    


if __name__ == "__main__":
    main()