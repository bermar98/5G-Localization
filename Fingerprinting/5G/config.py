# 5g_simulation/config.py

class HallenConfig:
    """
    Alle Parameter der Fabrikhalle hier anpassen.
    Später einfach mit echten Werten befüllen.
    """

    # ── Raumgeometrie ──────────────────────────────
    LAENGE_M        = 50.0    # Hallenlänge in Metern
    BREITE_M        = 30.0    # Hallenbreite in Metern
    HOEHE_M         = 8.0     # Deckenhöhe in Metern

    # ── Basisstationen ─────────────────────────────
    ANZAHL_BS       = 4       # Anzahl 5G Access Points
    BS_HOEHE_M      = 6.0     # Montagehöhe an der Decke
    BS_LAYOUT       = "rectangular"  # "rectangular" oder "hexagonal"

    # ── AGV ────────────────────────────────────────
    ANZAHL_AGV      = 1       # Anzahl zu lokalisierende AGVs
    AGV_HOEHE_M     = 0.5     # Höhe der AGV-Antenne

    # ── 5G Parameter ───────────────────────────────
    TRAEGERFREQUENZ = 3.7e9   # 3.7 GHz (typisch für Campusnetze)
    BANDBREITE      = 100e6   # 100 MHz

    # ── Simulation ─────────────────────────────────
    ANZAHL_SNAPSHOTS = 100    # Messpunkte pro Simulation
    RAUSCH_FIGUR_DB  = 7.0    # Rauschzahl des Empfängers


class SimulationsConfig:
    """
    Parameter für die ToA-Positionierung
    """
    LICHTGESCHWINDIGKEIT = 3e8   # m/s
    ZIEL_GENAUIGKEIT_M   = 1.0   # angestrebte Genauigkeit in Metern
    METHODE              = "UL-ToA"  # Uplink Time of Arrival