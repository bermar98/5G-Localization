import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

datei = r"C:\Dokumente\Studium\Master\Masterarbeit\Code\agv_tracking.json"

# AGV Maße (anpassen!)
LENGTH = 0.2   # Länge
WIDTH = 0.6    # Breite


x_werte = []
y_werte = []
theta_werte = []

''' with open(datei, "r", encoding="utf-8") as f:
    for zeile in f:
        daten = json.loads(zeile)
        x_werte.append(daten["x"])
        y_werte.append(daten["y"])

plt.figure(figsize=(10,8))
plt.plot(x_werte, y_werte, linewidth=2, label="Fahrlinie")
plt.scatter(x_werte[0], y_werte[0], s=100, label="Start")
plt.scatter(x_werte[-1], y_werte[-1], s=100, label="Ende")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("AGV Fahrtrajektorie")
plt.grid(True)
plt.axis("equal")
plt.legend()
 '''

plt.ion()  # interaktiver Modus
fig, ax = plt.subplots()
linie, = ax.plot([], [], linewidth=2)
# punkt, = ax.plot([], [], 'ro')
# AGV Rechteck (initial)
agv_patch = Polygon(np.zeros((4, 2)), closed=True, facecolor="red", edgecolor="red", alpha=0.6)
ax.add_patch(agv_patch)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("AGV Replay Simulation")
ax.grid(True)
ax.axis("equal")

def create_rectangle(x, y, theta):
    # Rechteck im lokalen Koordinatensystem
    rect = np.array([
        [-LENGTH/2, -WIDTH/2],
        [ LENGTH/2, -WIDTH/2],
        [ LENGTH/2,  WIDTH/2],
        [-LENGTH/2,  WIDTH/2]
    ])

    # Rotation
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    rect = rect @ rot.T
    rect[:, 0] += x
    rect[:, 1] += y

    return rect

with open(datei, "r", encoding="utf-8") as f:
    for zeile in f:
        daten = json.loads(zeile)

        x_werte.append(daten["x"])
        y_werte.append(daten["y"])
        theta_werte.append(daten["theta"]-0.3)
        

        # Update Plot
        x = float(x_werte[-1])
        y = float(y_werte[-1])
        theta = float(theta_werte[-1])
        linie.set_data(x_werte, y_werte)
        # punkt.set_data([x], [y])
        # AGV Rechteck aktualisieren
        rect = create_rectangle(x, y, theta)
        agv_patch.set_xy(rect)

        ax.relim()
        ax.autoscale_view()

        plt.pause(0.05)  # Geschwindigkeit der Simulation

plt.ioff()
plt.show()