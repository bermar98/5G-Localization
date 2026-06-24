# =============================================================================
#  signal_processing/pdsch_generator.py
#  PDSCH-Generierung – nutzt py3gpp wenn verfügbar
#
#  Matlab-Äquivalente:
#    nrPDSCHConfig()        → py3gpp.nrPDSCHConfig()     ✅
#    nrPDSCH()              → py3gpp.nrPDSCH()           ✅
#    nrPDSCHIndices()       → py3gpp.nrPDSCHIndices()    ✅
#    nrPDSCHDMRS()          → py3gpp.nrPDSCHDMRS()       ✅
#    nrPDSCHDMRSIndices()   → py3gpp.nrPDSCHDMRSIndices()✅
#    nrResourceGrid()       → py3gpp.nrResourceGrid()    ✅
# =============================================================================

import numpy as np
import config

try:
    from py3gpp import *
    PY3GPP_PDSCH = True
except ImportError:
    PY3GPP_PDSCH = False


def _make_carrier(slot: int = 0):
    """
    Erstellt nrCarrierConfig analog zu Matlab:
        carrier = nrCarrierConfig;
        carrier.NSizeGrid = 52;
        carrier.SubcarrierSpacing = 15;
        carrier.NSlot = slotIdx;
    """
    carrier = nrCarrierConfig()
    carrier.NSizeGrid         = config.NUM_RBS
    carrier.SubcarrierSpacing = int(config.SCS_HZ / 1e3)
    carrier.NSlot             = slot
    return carrier


def _make_pdsch_config():
    """
    Erstellt nrPDSCHConfig analog zu Matlab:
        pdsch = nrPDSCHConfig;
        pdsch.PRBSet = 0:51;
        pdsch.SymbolAllocation = [0 14];
        pdsch.DMRS.NumCDMGroupsWithoutData = 1;

    py3gpp Hinweise:
    - PRBSet muss np.ndarray oder range sein, nicht list
    - SymbolAllocation: [StartSymbol, NumSymbols]
    """
    pdsch = nrPDSCHConfig()
    pdsch.PRBSet           = np.arange(config.NUM_RBS)  # kein list!
    pdsch.SymbolAllocation = [0, 14]
    return pdsch


def nrPDSCH_grid(slot: int = 0, rng_seed: int = 0) -> tuple:
    """
    Generiert PDSCH-Ressourcengitter (Nsc × 14) mit py3gpp oder Eigenimplementierung.

    Analog zu Matlab:
        [pdschInd, pdschInfo] = nrPDSCHIndices(carrier, pdsch);
        data     = randi([0 1], pdschInfo.G, 1);
        pdschSym = nrPDSCH(carrier, pdsch, data);
        dmrsInd  = nrPDSCHDMRSIndices(carrier, pdsch);
        dmrsSym  = nrPDSCHDMRS(carrier, pdsch);
        dataSlotGrid(pdschInd) = pdschSym;
        dataSlotGrid(dmrsInd)  = dmrsSym;

    Returns
    -------
    grid     : np.ndarray, shape (14, Nsc) – PDSCH + DM-RS
    pdsch_idx: Subcarrier-Indizes PDSCH
    dmrs_idx : Subcarrier-Indizes DM-RS
    """
    if PY3GPP_PDSCH:
        return _nrPDSCH_py3gpp(slot, rng_seed)
    else:
        return _nrPDSCH_eigen(slot, rng_seed)


def _nrPDSCH_py3gpp(slot: int, rng_seed: int) -> tuple:
    """PDSCH-Generierung via py3gpp – exakt analog zu Matlab."""
    carrier = _make_carrier(slot)
    pdsch   = _make_pdsch_config()
    Nsc     = config.NUM_RBS * 12

    grid = np.zeros((Nsc, 14), dtype=np.complex64)

    try:
        # nrPDSCHIndices gibt (N, 2) Array zurück: [subcarrier, symbol]
        # py3gpp Besonderheit: Index ist multidimensional (nicht 1D wie Matlab)
        pdsch_ind = nrPDSCHIndices(carrier, pdsch)
        pdsch_ind = np.array(pdsch_ind)

        # Datenbits generieren (2 Bits pro QPSK-Symbol)
        rng  = np.random.default_rng(rng_seed)
        n_sym = pdsch_ind.shape[0] if pdsch_ind.ndim > 1 else len(pdsch_ind)
        bits = rng.integers(0, 2, n_sym * 2).tolist()

        pdsch_sym = nrPDSCH(carrier, pdsch, bits)
        pdsch_sym = np.array(pdsch_sym).flatten()

        # Indizes in Grid eintragen
        if pdsch_ind.ndim == 2:
            # Format (N, 2): Spalte 0 = Subcarrier, Spalte 1 = Symbol
            sc_idx  = pdsch_ind[:, 0].astype(int)
            sym_idx = pdsch_ind[:, 1].astype(int)
            grid[sc_idx, sym_idx] = pdsch_sym[:len(sc_idx)]
        else:
            # Format (N,): linearer Index im Grid
            idx_flat = pdsch_ind.astype(int).flatten()
            for k, idx in enumerate(idx_flat):
                sc  = idx % Nsc
                sym = idx // Nsc
                if k < len(pdsch_sym):
                    grid[sc, sym] = pdsch_sym[k]

        # DM-RS
        dmrs_ind = nrPDSCHDMRSIndices(carrier, pdsch)
        dmrs_sym = nrPDSCHDMRS(carrier, pdsch)
        dmrs_ind = np.array(dmrs_ind)
        dmrs_sym = np.array(dmrs_sym).flatten()

        if dmrs_ind.ndim == 2:
            sc_idx  = dmrs_ind[:, 0].astype(int)
            sym_idx = dmrs_ind[:, 1].astype(int)
            grid[sc_idx, sym_idx] = dmrs_sym[:len(sc_idx)]
        else:
            idx_flat = dmrs_ind.astype(int).flatten()
            for k, idx in enumerate(idx_flat):
                sc  = idx % Nsc
                sym = idx // Nsc
                if k < len(dmrs_sym):
                    grid[sc, sym] = dmrs_sym[k]

    except Exception as e:
        print(f"[PDSCH] py3gpp Fehler: {e} → Fallback auf Eigenimplementierung")
        return _nrPDSCH_eigen(slot, rng_seed)

    # Transponieren auf (14, Nsc) für unser System
    return grid.T.astype(np.complex64), pdsch_ind, dmrs_ind


def _nrPDSCH_eigen(slot: int, rng_seed: int) -> tuple:
    """PDSCH-Eigenimplementierung als Fallback."""
    Nsc    = config.NUM_RBS * 12
    rng    = np.random.default_rng(rng_seed)
    grid   = np.zeros((14, Nsc), dtype=np.complex64)
    all_sc = np.arange(Nsc)

    # DM-RS auf Symbol 2 und 11 (Standard PDSCH Mapping Type A)
    dmrs_symbols = [2, 11]
    pdsch_idx_all = []
    dmrs_idx_all  = []

    for sym in range(14):
        bits = rng.integers(0, 2, 2 * Nsc)
        sym_data = ((1 - 2*bits[:Nsc]) + 1j*(1 - 2*bits[Nsc:])) / np.sqrt(2)
        grid[sym, all_sc] = sym_data.astype(np.complex64)
        if sym in dmrs_symbols:
            dmrs_idx_all.append(all_sc)
        else:
            pdsch_idx_all.append(all_sc)

    return grid, pdsch_idx_all, dmrs_idx_all


def should_transmit_pdsch(slot_idx: int, prs_slot_offsets: list) -> bool:
    """
    Bestimmt ob PDSCH in diesem Slot gesendet wird.
    Analog zu Matlab: if all(cellfun(@isempty, prsInd))
    """
    return slot_idx not in prs_slot_offsets
