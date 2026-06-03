import json
from pathlib import Path
from dataset_builder import DatasetBuilder
from data_splitter import split_and_save
from model import IndoorLocalizationModel

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "fingerprints.json"
MODEL_PATH = BASE_DIR / "models" / "trained_model.keras"

def main():

    # 1. Daten lesen
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 2. Split auslagern an data_splitter
    X_train, X_test, y_train, y_test, all_bssids = split_and_save(data)

    # 3. Modell trainieren
    model = IndoorLocalizationModel.create_model(
        X_train.shape[1]
    )

    model.summary()

    model.fit(X_train, y_train,
              epochs=50,
              batch_size=8,
              validation_data=(X_test, y_test))

    model.save(
        "trained_model.keras"
    )


if __name__ == "__main__":
    main()