# 5g_simulation/tdoa_simulation.py
import numpy as np
import matplotlib.pyplot as plt

from toolkit5G.ChannelModels    import AntennaArrays, SimulationLayout, ParameterGenerator, ChannelGenerator
from toolkit5G.ResourceMapping  import ResourceMapperSRS
from toolkit5G.ReceiverAlgorithms import ChannelEstimationSRS
from toolkit5G.Positioning      import ToAEstimation, LeastSquareTDoA
from toolkit5G.ChannelProcessing import AddNoise

from config import HallenConfig as cfg


class FabrikTDoASimulation:

    def __init__(self):
        self.layout      = None
        self.bs_pos      = None
        self.agv_pos     = None

    # ── 1. Layout ──────────────────────────────────────────────────────────
    def erstelle_layout(self):

        # AGV Antenne (OMNI, 1 Element)
        agv_antenne = AntennaArrays(
            antennaType      = "OMNI",
            centerFrequency  = cfg.TRAEGERFREQUENZ,
            arrayStructure   = np.array([1, 1, 1, 1, 1])
        )
        agv_antenne()

        # BS Antenne (3GPP_38.901, 4x4 MIMO)
        bs_antenne = AntennaArrays(
            antennaType      = "3GPP_38.901",
            centerFrequency  = cfg.TRAEGERFREQUENZ,
            arrayStructure   = np.array([1, 1, 4, 4, 1])
        )
        bs_antenne()

        # SimulationLayout — Indoor Factory
        self.layout = SimulationLayout(
            numOfBS          = cfg.ANZAHL_BS,
            numOfUE          = cfg.ANZAHL_AGV,
            layoutType       = "IndoorFactory",
            layoutLength     = cfg.LAENGE_M,
            layoutWidth      = cfg.BREITE_M,
            heightOfRoom     = cfg.HOEHE_M,
            heightOfBS       = cfg.BS_HOEHE_M,
            heightOfUE       = cfg.AGV_HOEHE_M,
            isd              = cfg.ISD,
            bsLayoutType     = "Rectangular",
            ueDropType       = "Rectangular",
            carrierFrequency = cfg.TRAEGERFREQUENZ,
            numOfSnapShots   = cfg.ANZAHL_SNAPSHOTS,
            bsAntennaArray   = bs_antenne,
            ueAntennaArray   = agv_antenne,
        )

        # Manuelle BS-Positionen überschreiben
        if cfg.BS_POSITIONEN_MANUELL is not None:
            self.bs_pos = cfg.BS_POSITIONEN_MANUELL
        else:
            self.bs_pos = self.layout.BSLocations

        self.agv_pos = self.layout.UELocations

        print("=== Layout ===")
        print(f"Halle: {cfg.LAENGE_M}m x {cfg.BREITE_M}m x {cfg.HOEHE_M}m")
        for i, bs in enumerate(self.bs_pos):
            print(f"  BS{i}: x={bs[0]:.1f}m, y={bs[1]:.1f}m, z={bs[2]:.1f}m")
        print(f"  AGV:  x={self.agv_pos[0,0]:.2f}m, y={self.agv_pos[0,1]:.2f}m")

    # ── 2. Kanal ───────────────────────────────────────────────────────────
    def generiere_kanal(self):

        print("\n=== Kanal ===")
        param_gen = ParameterGenerator(self.layout)
        params    = param_gen()

        kanal_gen = ChannelGenerator(self.layout, params)
        kanal     = kanal_gen()

        print("Kanal generiert.")
        return kanal

    # ── 3. SRS Mapping ─────────────────────────────────────────────────────
    def generiere_srs(self):

        srs_mapper = ResourceMapperSRS(
            numRBs    = cfg.NUM_RBS,
            numSlots  = 1,
            Nfft      = cfg.NFFT,
            scs       = cfg.SCS,
        )
        srs_signal = srs_mapper()
        return srs_mapper, srs_signal

    # ── 4. ToA Schätzung ───────────────────────────────────────────────────
    def schaetze_toa(self, kanal, srs_mapper, srs_signal):

        print("\n=== ToA Schätzung ===")

        # Rauschen hinzufügen
        add_noise  = AddNoise()
        rx_signal  = add_noise(srs_signal, snrdB=cfg.SNR_DB)

        # Kanalschätzung via SRS
        ch_estimator = ChannelEstimationSRS(srs_mapper)
        h_est        = ch_estimator(rx_signal, kanal)

        # ToA aus Kanalimpulsantwort
        toa_estimator = ToAEstimation(
            nfft = cfg.NFFT,
            scs  = cfg.SCS,
        )
        toa = toa_estimator(h_est)

        for i, t in enumerate(toa):
            print(f"  ToA BS{i}: {t*1e9:.2f} ns")

        return toa

    # ── 5. TDoA → Position ─────────────────────────────────────────────────
    def schaetze_position(self, toa):

        print("\n=== TDoA Positionsschätzung ===")

        # TDoA berechnen (Differenz zur Referenz-BS0)
        tdoa = np.array([toa[i] - toa[0] for i in range(1, len(toa))])
        for i, t in enumerate(tdoa):
            print(f"  TDoA BS{i+1}-BS0: {t*1e9:.2f} ns")

        # LeastSquareTDoA aus dem Toolkit
        tdoa_estimator = LeastSquareTDoA()

        position, fehler = tdoa_estimator(
            refPosition = self.bs_pos,   # (N_bs, 3)
            tdoa        = tdoa,          # (N_bs-1,)
        )

        return position, fehler

    # ── 6. Visualisierung ──────────────────────────────────────────────────
    def visualisieren(self, geschaetzte_pos):

        wahre_pos = self.agv_pos[0]
        fehler    = np.sqrt(
            (wahre_pos[0] - geschaetzte_pos[0])**2 +
            (wahre_pos[1] - geschaetzte_pos[1])**2
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        halle = plt.Rectangle(
            (0, 0), cfg.LAENGE_M, cfg.BREITE_M,
            fill=False, edgecolor="black", linewidth=2
        )
        ax.add_patch(halle)

        for i, bs in enumerate(self.bs_pos):
            ax.plot(bs[0], bs[1], "b^", markersize=12)
            ax.annotate(f"BS{i}", (bs[0], bs[1]),
                       textcoords="offset points", xytext=(5, 5))

        ax.plot(wahre_pos[0], wahre_pos[1], "go",
               markersize=14, label=f"Wahre Position ({wahre_pos[0]:.2f}, {wahre_pos[1]:.2f})")
        ax.plot(geschaetzte_pos[0], geschaetzte_pos[1], "rx",
               markersize=14, markeredgewidth=3,
               label=f"TDoA Schätzung ({geschaetzte_pos[0]:.2f}, {geschaetzte_pos[1]:.2f})")

        ax.annotate("", xy=(geschaetzte_pos[0], geschaetzte_pos[1]),
                   xytext=(wahre_pos[0], wahre_pos[1]),
                   arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

        ax.set_xlim(-3, cfg.LAENGE_M + 3)
        ax.set_ylim(-3, cfg.BREITE_M + 3)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(
            f"5G Indoor Lokalisierung — UL-TDoA (IndoorFactory)\n"
            f"Halle: {cfg.LAENGE_M}m × {cfg.BREITE_M}m  |  "
            f"Fehler: {fehler:.2f}m"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tdoa_ergebnis.png", dpi=150)
        plt.show()

        print(f"\nWahre Position:      x={wahre_pos[0]:.2f}m, y={wahre_pos[1]:.2f}m")
        print(f"Geschätzte Position: x={geschaetzte_pos[0]:.2f}m, y={geschaetzte_pos[1]:.2f}m")
        print(f"Positionsfehler:     {fehler:.2f}m")
        return fehler

    # ── Main ───────────────────────────────────────────────────────────────
    def run(self):
        print("=== 5G Indoor Lokalisierung — UL-TDoA ===\n")

        self.erstelle_layout()
        kanal                   = self.generiere_kanal()
        srs_mapper, srs_signal  = self.generiere_srs()
        toa                     = self.schaetze_toa(kanal, srs_mapper, srs_signal)
        position, unsicherheit  = self.schaetze_position(toa)
        fehler                  = self.visualisieren(position)

        return fehler


if __name__ == "__main__":
    sim = FabrikTDoASimulation()
    sim.run()