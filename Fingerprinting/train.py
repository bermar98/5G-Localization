import json
from pathlib import Path


from dataset_builder import DatasetBuilder
from model import IndoorLocalizationModel

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "fingerprints.json"

def main():

    ''' with open(
        "fingerprints.json",
        "r",
        encoding="utf-8"
    ) as file:'''
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    X, y, all_bssids = DatasetBuilder.build_dataset(data)

    model = IndoorLocalizationModel.create_model(
        X.shape[1]
    )

    model.summary()

    model.fit(
        X,
        y,
        epochs=50,
        batch_size=8,
        validation_split=0.2
    )

    model.save(
        "trained_model.keras"
    )


if __name__ == "__main__":
    main()