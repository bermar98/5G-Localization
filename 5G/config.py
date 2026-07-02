# 5g_simulation/config.py
import numpy as np

class Config:
    # ── Simulation ─────────────────────────────────
    N_FRAMES        = 2           # Anzahl 10ms Frames
    FC              = 3e9         # Trägerfrequenz Hz
    NUM_GNB         = 7           # Anzahl Basisstationen (3-19)
    INTERSITE_DIST  = 500         # Abstand zwischen BS in Metern
    NOISE_FIGURE_DB = 6           # Rauschzahl UE (dB)
    RX_ANT_TEMP     = 290         # Antennentemperatur (K)
    SEED            = 50          # Seed für Reproduzierbarkeit

    # ── PRS Konfiguration ──────────────────────────
    NUM_RB          = 52          # Ressource Blöcke
    COMB_SIZE       = 2           # Comb-Size
    NUM_PRS_SYMBOLS = 12          # PRS Symbole pro Slot
    SCS             = 15e3        # Subcarrier Spacing (Hz)
    NFFT            = 1024        # FFT Größe

    # ── Halle (später anpassen) ────────────────────
    LAENGE_M        = 50.0
    BREITE_M        = 30.0

    # ── Manuelle gNB Positionen [x, y, z] ─────────
    # None = automatisch generiert
    GNB_POSITIONEN_MANUELL = None
    # Beispiel:
    # GNB_POSITIONEN_MANUELL = np.array([
    #     [  0,    0,  25],
    #     [500,    0,  25],
    #     [250,  433,  25],
    # ])

    # ── Physikalische Konstanten ───────────────────
    C               = 3e8         # Lichtgeschwindigkeit m/s
    K_BOLTZ         = 1.380649e-23 # Boltzmann Konstante