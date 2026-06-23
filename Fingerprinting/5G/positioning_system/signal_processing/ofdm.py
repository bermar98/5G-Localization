# =============================================================================
#  signal_processing/ofdm.py
#  OFDM Modulation / Demodulation nach 3GPP TS 38.211 §5.3
# =============================================================================

import numpy as np
import config


def _cyclic_prefix_lengths(Nfft: int, n_symbols: int = 14) -> list:
    """
    CP-Längen für normalen CP bei 30 kHz SCS nach TS 38.211 Tabelle 5.3.1-1.
    Erstes Symbol hat längeren CP (Slot-Ausrichtung).
    """
    cp_long  = int(round(Nfft * 144 / 2048)) + Nfft // 128
    cp_short = int(round(Nfft * 144 / 2048))
    return [cp_long] + [cp_short] * (n_symbols - 1)


def ofdm_modulate(grid: np.ndarray, Nfft: int = config.NFFT) -> np.ndarray:
    """
    OFDM-Modulation: Ressourcengitter → Zeitbereich-Waveform.

    Subcarrier-Mapping: zentriert um DC (DC selbst bleibt leer).

    Parameters
    ----------
    grid : np.ndarray, shape (n_symbols, Nsc), dtype complex64
    Nfft : FFT-Größe

    Returns
    -------
    waveform : np.ndarray, shape (N_samples,), dtype complex64
    """
    n_sym, Nsc = grid.shape
    assert Nsc <= Nfft, f"Nsc ({Nsc}) > Nfft ({Nfft})"
    cps  = _cyclic_prefix_lengths(Nfft, n_sym)
    half = Nsc // 2
    samples = []

    for l in range(n_sym):
        fd = np.zeros(Nfft, dtype=np.complex64)
        fd[1:half + 1]           = grid[l, half:]   # obere Hälfte
        fd[Nfft - half:Nfft]     = grid[l, :half]   # untere Hälfte
        td = np.fft.ifft(fd) * np.sqrt(Nfft)
        cp = td[-cps[l]:]                            # Cyclic Prefix
        samples.append(np.concatenate([cp, td]))

    return np.concatenate(samples).astype(np.complex64)


def ofdm_demodulate(
    waveform: np.ndarray,
    Nfft:     int = config.NFFT,
    Nsc:      int = config.NSC,
    n_symbols: int = 14
) -> np.ndarray:
    """
    OFDM-Demodulation: Zeitbereich-Waveform → Ressourcengitter.

    Parameters
    ----------
    waveform  : np.ndarray, shape (N_samples,), dtype complex64
    Nfft      : FFT-Größe
    Nsc       : Anzahl Subcarrier
    n_symbols : Anzahl OFDM-Symbole pro Slot

    Returns
    -------
    grid : np.ndarray, shape (n_symbols, Nsc), dtype complex64
    """
    cps  = _cyclic_prefix_lengths(Nfft, n_symbols)
    grid = np.zeros((n_symbols, Nsc), dtype=np.complex64)
    half = Nsc // 2
    idx  = 0

    for l in range(n_symbols):
        idx += cps[l]                                # CP überspringen
        sym  = waveform[idx:idx + Nfft]
        idx += Nfft
        fd   = np.fft.fft(sym) / np.sqrt(Nfft)
        grid[l, half:] = fd[1:half + 1]             # obere Hälfte
        grid[l, :half] = fd[Nfft - half:Nfft]       # untere Hälfte

    return grid
