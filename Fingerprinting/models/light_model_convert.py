import tensorflow as tf
import os

# Ordner des aktuellen Skripts ermitteln
script_dir = os.path.dirname(os.path.abspath(__file__))

# Modell laden (ebenfalls aus dem Skript-Ordner)
model_path = os.path.join(script_dir, "trained_model.keras")
model = tf.keras.models.load_model(model_path)

# Converter erstellen
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Speichern im gleichen Ordner wie das Skript
output_path = os.path.join(script_dir, "modell.tflite")
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"Gespeichert unter: {output_path}")