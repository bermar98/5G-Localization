# ml/visualize.py
# Erzeugt alle Diagnose-Plots für das trainierte Indoor-Localization-Modell:
#   1. Architektur (Layer-Grafik)
#   2. Trainingsverlauf (Loss/MAE über Epochen)
#   3. RSSI-Heatmap über alle Trainings-Samples
#   4. PCA-Projektion des Feature-Raums, eingefärbt nach Position
#   5. Grundriss-Plot: echte vs. vorhergesagte Position (Testset)
#   6. Feature-Importance (grobe Heuristik über Gewichte des ersten Layers)
#
# Nutzung:
#   python visualize.py            -> erzeugt alle Plots aus den gespeicherten Artefakten
#
# Voraussetzung: train.py wurde bereits ausgeführt (Modell, Scaler, Splits, History vorhanden)

import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "models" / "trained_model.keras"
SCALER_PATH = BASE_DIR.parent / "models" / "scaler.pkl"
HISTORY_PATH = BASE_DIR.parent / "models" / "history.json"
SPLITS_DIR = BASE_DIR.parent / "data" / "splits"
PLOTS_DIR = BASE_DIR.parent / "models" / "plots"


def _load_artifacts():
    """Lädt Modell, Scaler, Splits und BSSID-Liste von der Festplatte."""
    model = models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_train = np.load(SPLITS_DIR / "X_train.npy")
    X_test = np.load(SPLITS_DIR / "X_test.npy")
    y_train = np.load(SPLITS_DIR / "y_train.npy")
    y_test = np.load(SPLITS_DIR / "y_test.npy")

    with open(SPLITS_DIR / "all_bssids.json", "r", encoding="utf-8") as f:
        all_bssids = json.load(f)

    history = None
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        print(f"[visualize] Hinweis: {HISTORY_PATH} nicht gefunden — Trainingsverlauf wird übersprungen.")
        print("[visualize] Ergänze in train.py: json.dump(history.history, ...) nach dem Training.")

    return model, scaler, X_train, X_test, y_train, y_test, all_bssids, history


def plot_architecture(model):
    """1. Architektur-Grafik des Modells (Layer, Shapes, Aktivierungen)."""
    out_path = PLOTS_DIR / "architecture.png"
    try:
        plot_model(
            model,
            to_file=str(out_path),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            rankdir="TB",
        )
        print(f"[visualize] Architektur gespeichert: {out_path}")
    except ImportError:
        print("[visualize] Übersprungen: 'pydot' und/oder 'graphviz' fehlen.")
        print("            Installieren mit: pip install pydot graphviz  (+ 'sudo apt install graphviz')")


def plot_training_history(history):
    """2. Loss- und MAE-Verlauf über die Trainingsepochen."""
    if history is None:
        return

    out_path = PLOTS_DIR / "training_history.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoche")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss-Verlauf")

    axes[1].plot(history["mae"], label="Train MAE")
    axes[1].plot(history["val_mae"], label="Val MAE")
    axes[1].set_xlabel("Epoche")
    axes[1].set_ylabel("MAE (Meter)")
    axes[1].legend()
    axes[1].set_title("MAE-Verlauf")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[visualize] Trainingsverlauf gespeichert: {out_path}")


def plot_rssi_heatmap(X_train):
    """3. Heatmap der RSSI-Werte über alle Trainingssamples und BSSIDs."""
    out_path = PLOTS_DIR / "rssi_heatmap.png"

    plt.figure(figsize=(14, 6))
    sns.heatmap(X_train, cmap="viridis", cbar_kws={"label": "RSSI (dBm) bzw. skaliert"})
    plt.xlabel("BSSID-Index")
    plt.ylabel("Sample-Index")
    plt.title("RSSI-Fingerprints über alle Trainingsdaten")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[visualize] RSSI-Heatmap gespeichert: {out_path}")


def plot_pca(X_train, y_train, scaler):
    """4. PCA-Projektion des Feature-Raums, eingefärbt nach x-Position.

    Nutzt die skalierten Daten (scaler.transform), da das der Feature-Raum
    ist, den das Modell tatsächlich sieht. Auf rohen RSSI-Werten würde der
    -100-Sentinelwert für "nicht gesehene BSSID" die Varianz dominieren.
    """
    out_path = PLOTS_DIR / "pca_features.png"

    X_train_scaled = scaler.transform(X_train)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train[:, 0], cmap="coolwarm")
    plt.colorbar(scatter, label="x-Position (m)")
    plt.xlabel("PCA Komponente 1")
    plt.ylabel("PCA Komponente 2")
    plt.title("Feature-Raum, eingefärbt nach x-Position")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[visualize] PCA-Plot gespeichert: {out_path}")


def plot_prediction_errors(model, scaler, X_test, y_test):
    """5. Grundriss-Plot: echte vs. vorhergesagte Positionen im Testset."""
    out_path = PLOTS_DIR / "prediction_errors.png"

    # WICHTIG: X_test aus den .npy-Splits ist unskaliert (roher RSSI-Wert).
    # Das Modell wurde aber auf skalierten Daten trainiert -> hier muss
    # exakt derselbe Scaler wie beim Training angewendet werden.
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled, verbose=0)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test[:, 0], y_test[:, 1], c="blue", label="Echte Position", alpha=0.6)
    plt.scatter(predictions[:, 0], predictions[:, 1], c="red", label="Vorhergesagte Position", alpha=0.6)

    for i in range(len(y_test)):
        plt.plot(
            [y_test[i, 0], predictions[i, 0]],
            [y_test[i, 1], predictions[i, 1]],
            "gray", alpha=0.3, linewidth=0.5
        )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.title("Positionsfehler: echte vs. vorhergesagte Position")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    distances = np.sqrt(
        (predictions[:, 0] - y_test[:, 0]) ** 2 +
        (predictions[:, 1] - y_test[:, 1]) ** 2
    )
    print(f"[visualize] Vorhersage-Plot gespeichert: {out_path}")
    print(f"[visualize] Mittlerer Fehler: {distances.mean():.2f}m | Max. Fehler: {distances.max():.2f}m")


def print_feature_importance(model, all_bssids, top_n=10):
    """6. Grobe Feature-Importance-Heuristik über die Gewichte des ersten Dense-Layers."""
    first_dense = None
    for layer in model.layers:
        weights = layer.get_weights()
        if weights and weights[0].ndim == 2:
            first_dense = weights[0]
            break

    if first_dense is None:
        print("[visualize] Kein Dense-Layer mit Gewichten gefunden, Feature-Importance übersprungen.")
        return

    importance = np.abs(first_dense).mean(axis=1)
    top_indices = np.argsort(importance)[::-1][:top_n]

    print(f"\n[visualize] Top {top_n} wichtigste BSSIDs (grobe Heuristik, kein SHAP):")
    for idx in top_indices:
        if idx < len(all_bssids):
            print(f"  {all_bssids[idx]}: Wichtigkeit {importance[idx]:.4f}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, scaler, X_train, X_test, y_train, y_test, all_bssids, history = _load_artifacts()

    plot_architecture(model)
    plot_training_history(history)
    plot_rssi_heatmap(X_train)
    plot_pca(X_train, y_train, scaler)
    plot_prediction_errors(model, scaler, X_test, y_test)
    print_feature_importance(model, all_bssids)

    print(f"\n[visualize] Alle Plots liegen in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()