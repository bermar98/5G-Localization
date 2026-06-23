# =============================================================================
#  signal_processing/srs_generator.py
#  SRS-Generierung nach 3GPP TS 38.211 §6.4.1.4
# =============================================================================

import numpy as np
import config


# Bandbreiten-Konfigurationstabelle (TS 38.211 Tabelle 6.4.1.4.3-1)
_M_SRS_TABLE = {
    0:  [4,   4,  4, 4], 1:  [8,   4,  4, 4], 2:  [12,  4,  4, 4],
    3:  [16,  4,  4, 4], 4:  [16,  8,  4, 4], 5:  [20,  4,  4, 4],
    6:  [24,  4,  4, 4], 7:  [24, 12,  4, 4], 8:  [28,  4,  4, 4],
    9:  [32, 16,  8, 4], 10: [36, 12,  4, 4], 11: [40, 20,  4, 4],
    12: [48, 16,  8, 4], 13: [48, 24, 12, 4], 14: [52,  4,  4, 4],
    15: [56, 28,  4, 4],
}


def _srs_base_sequence(M: int, n_id: int, cyclic_shift: int = 0) -> np.ndarray:
    """
    CAZAC-Basissequenz nach 3GPP TS 38.211 §6.4.1.4.2.

    Parameters
    ----------
    M            : Anzahl SRS-Subcarrier
    n_id         : Sequenz-ID (entspricht UE-Index / sequenceId)
    cyclic_shift : Zyklische Verschiebung α

    Returns
    -------
    seq : np.ndarray, shape (M,), dtype complex64, normiert auf |seq|=1/√M
    """
    u     = n_id % 30
    alpha = 2 * np.pi * cyclic_shift / 8
    phi   = np.pi * u * np.arange(M) * (np.arange(M) + 1) / 31
    seq   = np.exp(1j * (phi + alpha * np.arange(M)))
    return (seq / np.sqrt(M)).astype(np.complex64)


def generate_srs_grid(
    Nsc:              int,
    nue_idx:          int,
    transmission_comb: int = config.SRS_TRANSMISSION_COMB,
    symbol_start:     int = config.SRS_SYMBOL_START,
    cSRS:             int = config.SRS_CSRS,
    bSRS:             int = config.SRS_BSRS,
    cyclic_shift:     int = config.SRS_CYCLIC_SHIFT,
):
    """
    Erzeugt SRS-Ressourcengitter (14 × Nsc) für ein UE.

    Implementiert die Comb-Struktur nach TS 38.211 §6.4.1.4.3:
    - SRS-Subcarrier liegen auf jedem KTC-ten Subcarrier
    - Comb-Offset = nue_idx % KTC (UE-Trennung im Frequenzbereich)

    Parameters
    ----------
    Nsc              : Gesamtanzahl Subcarrier im Grid
    nue_idx          : UE-Index (bestimmt Comb-Offset und Sequenz)
    transmission_comb: KTC (2 oder 4)
    symbol_start     : OFDM-Symbol für SRS im Slot
    cSRS, bSRS       : Bandbreiten-Konfigurationsindizes
    cyclic_shift     : Zyklische Verschiebung

    Returns
    -------
    grid     : np.ndarray, shape (14, Nsc), dtype complex64
    srs_idx  : np.ndarray, Subcarrier-Indizes der SRS-Symbole
    tx_sym   : np.ndarray, gesendete SRS-Symbole (für Kanalschätzung)
    """
    bw_cfg   = _M_SRS_TABLE.get(cSRS, [48, 24, 12, 4])
    M_sc_SRS = bw_cfg[bSRS] * 12 // transmission_comb

    # Startsubcarrier (zentriert im Grid)
    k0      = max(0, (Nsc - bw_cfg[bSRS] * 12) // 2)
    comb_off = nue_idx % transmission_comb

    # Subcarrier-Indizes mit Comb-Struktur
    srs_idx = k0 + comb_off + np.arange(M_sc_SRS) * transmission_comb
    srs_idx = srs_idx[srs_idx < Nsc]
    M_sc_SRS = len(srs_idx)

    # SRS-Sequenz generieren
    tx_sym = _srs_base_sequence(M_sc_SRS, nue_idx, cyclic_shift)

    # Ressourcengitter befüllen
    grid = np.zeros((14, Nsc), dtype=np.complex64)
    grid[symbol_start, srs_idx] = tx_sym

    return grid, srs_idx, tx_sym
