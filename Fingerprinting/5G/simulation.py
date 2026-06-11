# 5g_simulation/simulation.py
import numpy as np
import matplotlib.pyplot as plt
from toolkit5G.ChannelModels import SimulationLayout, AntennaArrays
from config import HallenConfig as cfg, SimulationsConfig as scfg


class FabrikhallenSimulation:

    def __init__(self):
        self.config = cfg()
        self.layout = None
        self.bs_positionen = None
        self.agv_positionen = None

    def _erstelle_layout(self):
        """Simulationslayout basierend auf Konfiguration erstellen."""

        # Antennen-Arrays definieren
        bs_antenne = AntennaArrays(
            antennaType   = "3GPP_38_901",
            centerFreq    = cfg.TRAEGERFREQUENZ,
            arrayStructure = np.array([1, 1, 4, 1, 1])  # 4 Antennen
        )

        agv_antenne = AntennaArrays(
            antennaType   = "3GPP_38_901",
            centerFreq    = cfg.TRAEGERFREQUENZ,
            arrayStructure = np.array([1, 1, 1, 1, 1])  # 1 Antenne
        )

        # Layout erstellen
        self.layout = SimulationLayout(
            numOfBS          = cfg.ANZAHL_BS,
            numOfUE          = cfg.ANZAHL_AGV,
            layoutType       = "IndoorFactory",
            layoutLength     = cfg.LAENGE_M,
            layoutWidth      = cfg.BREITE_M,
            heightOfRoom     = cfg.HOEHE_M,
            heightOfBS       = cfg.BS_HOEHE_M,
            heightOfUE       = cfg.AGV_HOEHE_M,
            carrierFrequency = cfg.TRAEGERFREQUENZ,
            numOfSnapShots   = cfg.ANZAHL_SNAPSHOTS,
            bsAntennaArray   = bs_antenne,
            ueAntennaArray   = agv_antenne,
            BS_layout        = cfg.BS_LAYOUT,
        )

        self.bs_positionen  = self.layout.BSLocations
        self.agv_positionen = self.layout.UELocations

        print(f"Layout erstellt: {cfg.LAENGE_M}m x {cfg.BREITE_M}m x {cfg.HOEHE_M}m")
        print(f"Basisstationen: {cfg.ANZAHL_BS} @ {cfg.BS_HOEHE_M}m Höhe")
        print(f"BS Positionen:\n{self.bs_positionen}")

    def _berechne_toa(self):
        """Time of Arrival von AGV zu jeder Basisstation berechnen."""

        toa_messungen = []

        for bs_idx in range(cfg.ANZAHL_BS):
            bs_pos  = self.bs_positionen[bs_idx]
            agv_pos = self.agv_positionen[0]  # erstes AGV

            # Euklidische Distanz
            distanz = np.sqrt(
                (agv_pos[0] - bs_pos[0])**2 +
                (agv_pos[1] - bs_pos[1])**2 +
                (agv_pos[2] - bs_pos[2])**2
            )

            # ToA berechnen
            toa = distanz / scfg.LICHTGESCHWINDIGKEIT

            # Rauschen hinzufügen (realistisch)
            rauschen = np.random.normal(0, 1e-9)  # 1ns Standardabweichung
            toa_messungen.append(toa + rauschen)

            print(f"BS {bs_idx}: Distanz={distanz:.2f}m, ToA={toa*1e9:.2f}ns")

        return toa_messungen

    def _trilateration(self, toa_messungen):
        """Position aus ToA-Messungen schätzen."""
        from scipy.optimize import minimize

        distanzen = [t * scfg.LICHTGESCHWINDIGKEIT for t in toa_messungen]

        def residuals(pos):
            return sum(
                (np.sqrt(
                    (pos[0] - self.bs_positionen[i][0])**2 +
                    (pos[1] - self.bs_positionen[i][1])**2
                ) - distanzen[i])**2
                for i in range(cfg.ANZAHL_BS)
            )

        # Startpunkt: Hallenmitte
        start = [cfg.LAENGE_M / 2, cfg.BREITE_M / 2]
        result = minimize(residuals, start, method="Nelder-Mead")

        return result.x

    def visualisieren(self, wahre_pos, geschaetzte_pos):
        """Ergebnis visualisieren."""

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Halle zeichnen
        halle = plt.Rectangle(
            (0, 0), cfg.LAENGE_M, cfg.BREITE_M,
            fill=False, edgecolor="black", linewidth=2
        )
        ax.add_patch(halle)

        # Basisstationen
        for i, bs in enumerate(self.bs_positionen):
            ax.plot(bs[0], bs[1], "^", markersize=12,
                    color="blue", label=f"BS {i}" if i == 0 else "")
            ax.annotate(f"BS{i}", (bs[0], bs[1]),
                        textcoords="offset points", xytext=(5, 5))

        # Wahre Position
        ax.plot(wahre_pos[0], wahre_pos[1], "go",
                markersize=12, label="Wahre Position", zorder=5)

        # Geschätzte Position
        ax.plot(geschaetzte_pos[0], geschaetzte_pos[1], "rx",
                markersize=12, markeredgewidth=3,
                label="Geschätzte Position", zorder=5)

        # Fehler einzeichnen
        ax.plot(
            [wahre_pos[0], geschaetzte_pos[0]],
            [wahre_pos[1], geschaetzte_pos[1]],
            "r--", linewidth=1.5
        )

        fehler = np.sqrt(
            (wahre_pos[0] - geschaetzte_pos[0])**2 +
            (wahre_pos[1] - geschaetzte_pos[1])**2
        )

        ax.set_xlim(-2, cfg.LAENGE_M + 2)
        ax.set_ylim(-2, cfg.BREITE_M + 2)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(
            f"5G Indoor Positionierung (UL-ToA)\n"
            f"Halle: {cfg.LAENGE_M}m x {cfg.BREITE_M}m | "
            f"Fehler: {fehler:.2f}m"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("simulation_ergebnis.png", dpi=150)
        plt.show()

        return fehler

    def run(self):
        print("=== 5G Indoor Positionierung Simulation ===\n")

        # 1. Layout erstellen
        self._erstelle_layout()

        # 2. ToA berechnen
        print("\nBerechne ToA Messungen...")
        toa = self._berechne_toa()

        # 3. Position schätzen
        print("\nSchätze Position via Trilateration...")
        geschaetzte_pos = self._trilateration(toa)

        wahre_pos = self.agv_positionen[0]
        print(f"\nWahre Position:      x={wahre_pos[0]:.2f}m, y={wahre_pos[1]:.2f}m")
        print(f"Geschätzte Position: x={geschaetzte_pos[0]:.2f}m, y={geschaetzte_pos[1]:.2f}m")

        # 4. Visualisieren
        fehler = self.visualisieren(wahre_pos, geschaetzte_pos)
        print(f"\nPositionsfehler: {fehler:.2f}m")
        return fehler


if __name__ == "__main__":
    sim = FabrikhallenSimulation()
    sim.run()