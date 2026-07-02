# =============================================================================
#  signal_processing/ofdm.py
#  OFDM Mod/Demod – nutzt py3gpp wenn verfügbar
#
#  Matlab-Äquivalente:
#    nrOFDMModulate()   → py3gpp.nrOFDMModulate()   ✅
#    nrOFDMDemodulate() → py3gpp.nrOFDMDemodulate() ✅
#    nrOFDMInfo()       → py3gpp.nrOFDMInfo()       (falls vorhanden)
# =============================================================================

import numpy as np
import config

try:
    from py3gpp import *
    PY3GPP_OFDM = True
    print("[py3gpp] nrOFDMModulate/Demodulate verfügbar ✓")
except ImportError:
    PY3GPP_OFDM = False


def _make_carrier(slot: int = 0):
    carrier = nrCarrierConfig()
    carrier.NSizeGrid         = config.NUM_RBS
    carrier.SubcarrierSpacing = int(config.SCS_HZ / 1e3)
    carrier.NSlot             = slot
    return carrier


def ofdm_modulate(grid: np.ndarray, Nfft: int = config.NFFT,
                  slot: int = 0) -> np.ndarray:
    """
    OFDM-Modulation: Ressourcengitter (14 × Nsc) → Zeitbereich-Waveform.
    Nutzt py3gpp.nrOFDMModulate() wenn verfügbar.

    Analog zu Matlab:
        txWaveform = nrOFDMModulate(carrier, prsGrid + dataGrid)

    py3gpp Aufruf:
        nrOFDMModulate(carrier, grid)
        grid hat Format (Nsc, 14) in py3gpp – wir transponieren intern.
    """
    if PY3GPP_OFDM:
        carrier = _make_carrier(slot)
        grid_py3gpp = grid.T   # (14, Nsc) → (Nsc, 14)
        result = nrOFDMModulate(carrier, grid_py3gpp)
        # nrOFDMModulate gibt (waveform, info) zurück – analog zu Matlab
        waveform = result[0] if isinstance(result, (tuple, list)) else result
        return np.array(waveform, dtype=np.complex64).flatten()
    else:
        return _ofdm_modulate_eigen(grid, Nfft)


def ofdm_demodulate(waveform: np.ndarray, Nfft: int = config.NFFT,
                    Nsc: int = config.NSC, n_symbols: int = 14,
                    slot: int = 0) -> np.ndarray:
    """
    OFDM-Demodulation: Zeitbereich-Waveform → Ressourcengitter (14 × Nsc).
    Nutzt py3gpp.nrOFDMDemodulate() wenn verfügbar.
    """
    if PY3GPP_OFDM:
        carrier = _make_carrier(slot)
        result = nrOFDMDemodulate(carrier, waveform)
        grid_py3gpp = result[0] if isinstance(result, (tuple, list)) else result
        return np.array(grid_py3gpp, dtype=np.complex64).T
    else:
        return _ofdm_demodulate_eigen(waveform, Nfft, Nsc, n_symbols)


# =============================================================================
#  Eigenimplementierung (Fallback wenn py3gpp nicht verfügbar)
# =============================================================================

def _cp_lengths(Nfft: int, n_symbols: int = 14) -> list:
    """CP-Längen für normalen CP bei gegebener SCS."""
    cp_long  = int(round(Nfft * 144 / 2048)) + Nfft // 128
    cp_short = int(round(Nfft * 144 / 2048))
    return [cp_long] + [cp_short] * (n_symbols - 1)


def _ofdm_modulate_eigen(grid: np.ndarray, Nfft: int) -> np.ndarray:
    n_sym, Nsc = grid.shape
    cps  = _cp_lengths(Nfft, n_sym)
    half = Nsc // 2
    samples = []
    for l in range(n_sym):
        fd = np.zeros(Nfft, dtype=np.complex64)
        fd[1:half+1]       = grid[l, half:]
        fd[Nfft-half:Nfft] = grid[l, :half]
        td = np.fft.ifft(fd) * np.sqrt(Nfft)
        samples.append(np.concatenate([td[-cps[l]:], td]))
    return np.concatenate(samples).astype(np.complex64)


def _ofdm_demodulate_eigen(waveform, Nfft, Nsc, n_symbols):
    cps  = _cp_lengths(Nfft, n_symbols)
    grid = np.zeros((n_symbols, Nsc), dtype=np.complex64)
    half = Nsc // 2
    idx  = 0
    for l in range(n_symbols):
        idx += cps[l]
        fd   = np.fft.fft(waveform[idx:idx+Nfft]) / np.sqrt(Nfft)
        idx += Nfft
        grid[l, half:] = fd[1:half+1]
        grid[l, :half] = fd[Nfft-half:Nfft]
    return grid
