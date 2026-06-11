# 5g_simulation/config.py
import numpy as np

class HallenConfig:

    # ── Raumgeometrie ──────────────────────────────
    LAENGE_M         = 50.0
    BREITE_M         = 30.0
    HOEHE_M          = 8.0
    BS_HOEHE_M       = 6.0
    AGV_HOEHE_M      = 1.5

    # ── Netzwerk ───────────────────────────────────
    ANZAHL_BS        = 4           # mind. 4 für TDoA
    ANZAHL_AGV       = 1
    TRAEGERFREQUENZ  = 3.7e9       # 3.7 GHz Campusnetz
    ISD              = 20          # Inter-Site-Distance

    # ── Manuelle BS-Positionen [x, y, z] ──────────
    # None = automatisch (Rectangular Layout)
    BS_POSITIONEN_MANUELL = None
    # Beispiel:
    # BS_POSITIONEN_MANUELL = np.array([
    #     [ 5.0,  5.0, 6.0],
    #     [45.0,  5.0, 6.0],
    #     [ 5.0, 25.0, 6.0],
    #     [45.0, 25.0, 6.0],
    # ])

    # ── Simulation ─────────────────────────────────
    ANZAHL_SNAPSHOTS = 50
    SNR_DB           = 20          # Signal/Rausch-Verhältnis
    NUM_RBS          = 52          # Ressource Blocks
    SCS              = 30e3        # Subcarrier Spacing 30kHz
    NFFT             = 512