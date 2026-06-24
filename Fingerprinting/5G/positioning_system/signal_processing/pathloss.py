# =============================================================================
#  signal_processing/pathloss.py
#  Pathloss-Modell analog zu nrPathLoss(plCfg, fc, losFlag, gNBPos, UEPos)
#  3GPP TR 38.901 Tabelle 7.4.1-1
#
#  Matlab:
#    plCfg = nrPathLossConfig;
#    plCfg.Scenario = 'UMa';
#    PLdB = nrPathLoss(plCfg, fc, losFlag, gNBPos(:), UEPos(:));
#    PL   = 10^(PLdB/10);
#    rx   = txWaveform / sqrt(PL);
# =============================================================================

import numpy as np


def nrPathLoss(
    fc_hz:    float,
    gnb_pos:  np.ndarray,
    ue_pos:   np.ndarray,
    scenario: str  = "UMa",
    los:      bool = True,
) -> float:
    """
    Berechnet Pathloss in dB analog zu nrPathLoss().
    3GPP TR 38.901 Tabelle 7.4.1-1.

    Parameters
    ----------
    fc_hz    : Trägerfrequenz [Hz]
    gnb_pos  : gNB-Position [x, y, z] in Metern
    ue_pos   : UE-Position  [x, y, z] in Metern
    scenario : "UMa" oder "UMi" (wie plCfg.Scenario in Matlab)
    los      : True = LOS, False = NLOS (wie losFlag in Matlab)

    Returns
    -------
    pl_db : float – Pathloss in dB
    """
    fc_ghz = fc_hz / 1e9

    # 3D-Distanz (rangeangle() in Matlab = euklidischer Abstand)
    d3d = np.linalg.norm(np.array(gnb_pos) - np.array(ue_pos))
    d3d = max(d3d, 1.0)   # Mindestdistanz 1m

    # 2D-Distanz (horizontal)
    d2d = np.sqrt((gnb_pos[0]-ue_pos[0])**2 + (gnb_pos[1]-ue_pos[1])**2)
    d2d = max(d2d, 1.0)

    h_bs = gnb_pos[2]   # BS-Höhe
    h_ue = ue_pos[2]    # UE-Höhe

    if scenario == "UMa":
        if los:
            # TR 38.901 Tabelle 7.4.1-1 UMa LOS
            # PL1 = 28.0 + 22log10(d3d) + 20log10(fc)
            pl_db = 28.0 + 22*np.log10(d3d) + 20*np.log10(fc_ghz)
        else:
            # UMa NLOS
            # PL = 13.54 + 39.08log10(d3d) + 20log10(fc) - 0.6(h_ue-1.5)
            pl_db = (13.54
                     + 39.08 * np.log10(d3d)
                     + 20    * np.log10(fc_ghz)
                     - 0.6   * (h_ue - 1.5))

    elif scenario == "UMi":
        if los:
            # UMi LOS
            # PL = 32.4 + 21log10(d3d) + 20log10(fc)
            pl_db = 32.4 + 21*np.log10(d3d) + 20*np.log10(fc_ghz)
        else:
            # UMi NLOS
            pl_db = (35.3 * np.log10(d3d)
                     + 22.4
                     + 21.3 * np.log10(fc_ghz)
                     - 0.3  * (h_ue - 1.5))
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}. Wähle 'UMa' oder 'UMi'.")

    return float(pl_db)


def nrPathLoss_linear(fc_hz, gnb_pos, ue_pos, scenario="UMa", los=True) -> float:
    """Pathloss als linearer Faktor (nicht dB) – direkt für rx / sqrt(PL)."""
    pl_db = nrPathLoss(fc_hz, gnb_pos, ue_pos, scenario, los)
    return 10**(pl_db / 10)
