# =============================================================================
#  signal_processing/prs_generator.py
#  Analog zu: nrPRS(), nrPRSIndices(), nrPRSConfig
#  3GPP TS 38.211 §7.4.1.7
# =============================================================================

import numpy as np
import config

# Staggered k'-Offset Tabelle (TS 38.211 Tabelle 7.4.1.7.3-1)
# Analog zur Figure 8 in der Matlab-Dokumentation
_STAGGER_TABLE = {
    2:  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    4:  [0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3],
    6:  [0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5],
    12: [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11],
}


def _gold_sequence(length: int, c_init: int) -> np.ndarray:
    """
    Gold-Sequenz nach TS 38.211 §5.2.1.
    Analog zu der internen Sequenzgenerierung in nrPRS().

    Zwei m-Sequenzen (x1, x2) mit Vorlaufzeit Nc=1600,
    dann XOR → Gold-Sequenz c(i).
    """
    Nc = 1600
    N  = length + Nc

    x1 = np.zeros(N + 31, dtype=np.int8)
    x2 = np.zeros(N + 31, dtype=np.int8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1

    for n in range(N):
        x1[n+31] = (x1[n+3] + x1[n]) % 2
        x2[n+31] = (x2[n+3] + x2[n+2] + x2[n+1] + x2[n]) % 2

    return (x1[Nc:Nc+length] + x2[Nc:Nc+length]) % 2


def nrPRS_sequence(length: int, nprs_id: int,
                   slot: int, symbol: int) -> np.ndarray:
    """
    PRS-Sequenz r(m) nach TS 38.211 §7.4.1.7.2:

        r(m) = (1-2c(2m))/√2 + j(1-2c(2m+1))/√2

    c_init = 2^22·(14·n_slot + l + 1)·(2·N_ID_PRS + 1) + 2·N_ID_PRS

    Analog zu: nrPRS(carrier, prs)
    NPRSID entspricht N_ID_PRS ∈ {0,...,4095}
    """
    c_init = (
        (2**22) * (14*slot + symbol + 1) * (2*nprs_id + 1)
        + 2*nprs_id
    ) % (2**31)

    c = _gold_sequence(2*length, int(c_init))
    r = ((1 - 2*c[0::2]) + 1j*(1 - 2*c[1::2])) / np.sqrt(2)
    return r.astype(np.complex64)


def nrPRSIndices(Nsc: int, slot: int,
                 comb_size:   int = config.PRS_COMB_SIZE,
                 num_symbols: int = config.PRS_NUM_SYMBOLS,
                 symbol_start: int = config.PRS_SYMBOL_START,
                 num_rbs:     int = config.PRS_NUM_RBS,
                 rb_offset:   int = config.PRS_RB_OFFSET,
                 re_offset:   int = 0) -> dict:
    """
    PRS-Ressourcenindizes nach TS 38.211 §7.4.1.7.3:

        k = m·K_PRS + ((k_PRS_offset + k') mod K_PRS)

    Analog zu: nrPRSIndices(carrier, prs)

    Returns
    -------
    indices : dict {symbol_abs: np.ndarray of subcarrier indices}
    """
    stagger  = _STAGGER_TABLE.get(comb_size, [0]*12)
    k_start  = rb_offset * 12
    n_sc_prs = num_rbs * 12
    indices  = {}

    for sym_local in range(num_symbols):
        sym_abs = symbol_start + sym_local
        k_prime = stagger[sym_local % len(stagger)]
        eff_offset = (re_offset + k_prime) % comb_size
        n_sc_this  = n_sc_prs // comb_size
        k_idx = k_start + eff_offset + np.arange(n_sc_this) * comb_size
        k_idx = k_idx[k_idx < Nsc]
        indices[sym_abs] = k_idx

    return indices


def nrPRS(Nsc: int, nprs_id: int, slot: int,
          comb_size:   int = config.PRS_COMB_SIZE,
          num_symbols: int = config.PRS_NUM_SYMBOLS,
          symbol_start: int = config.PRS_SYMBOL_START,
          num_rbs:     int = config.PRS_NUM_RBS,
          rb_offset:   int = config.PRS_RB_OFFSET,
          re_offset:   int = 0) -> tuple:
    """
    Erzeugt PRS-Grid (14 × Nsc) und zugehörige Indizes/Symbole.

    Analog zu:
        prsSym = nrPRS(carrier, prs)
        prsInd = nrPRSIndices(carrier, prs)
        slotGrid(prsInd) = prsSym

    Parameters
    ----------
    Nsc      : Gesamtanzahl Subcarrier im Grid
    nprs_id  : NPRSID der gNB [0..4095]
    slot     : Slot-Nummer

    Returns
    -------
    prs_grid : np.ndarray, shape (14, Nsc)
    indices  : dict {symbol_abs: subcarrier_indices}
    symbols  : dict {symbol_abs: komplexe PRS-Symbole}
    """
    indices = nrPRSIndices(Nsc, slot, comb_size, num_symbols,
                           symbol_start, num_rbs, rb_offset, re_offset)

    prs_grid = np.zeros((14, Nsc), dtype=np.complex64)
    symbols  = {}

    for sym_abs, k_idx in indices.items():
        sym_local = sym_abs - symbol_start
        r = nrPRS_sequence(len(k_idx), nprs_id, slot, sym_abs)
        prs_grid[sym_abs, k_idx] = r
        symbols[sym_abs] = r

    return prs_grid, indices, symbols
