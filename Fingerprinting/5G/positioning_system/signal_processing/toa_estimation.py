# =============================================================================
#  signal_processing/toa_estimation.py
#  ToA via Kreuzkorrelation – nutzt py3gpp.nrTimingEstimate() wenn verfügbar
#
#  Matlab:
#    [~,mag] = nrTimingEstimate(carrier, rxWaveform, prsGrid);
#    corr    = mag(1:Nfft*SCS/15);
#    delayEst = find(corr==max(corr),1) - 1;
# =============================================================================

import numpy as np
import config

try:
    from py3gpp import *
    PY3GPP_TIMING = True
    print("[py3gpp] nrTimingEstimate verfügbar ✓")
except ImportError:
    PY3GPP_TIMING = False


def _make_carrier(slot: int = 0):
    carrier = nrCarrierConfig()
    carrier.NSizeGrid         = config.NUM_RBS
    carrier.SubcarrierSpacing = int(config.SCS_HZ / 1e3)
    carrier.NSlot             = slot
    return carrier


def nrTimingEstimate(rx_waveform: np.ndarray,
                     ref_waveform_or_grid,
                     sample_rate: float,
                     max_delay_samples: int = None,
                     slot: int = 0) -> tuple:
    """
    ToA-Schätzung via Kreuzkorrelation.
    Nutzt py3gpp.nrTimingEstimate() wenn verfügbar.

    Analog zu Matlab:
        [~, mag] = nrTimingEstimate(carrier, rxWaveform, prsGrid)
        corr     = mag(1 : Nfft*SCS/15)
        delayEst = find(corr == max(corr), 1) - 1

    Parameters
    ----------
    rx_waveform          : Empfangenes IQ-Signal
    ref_waveform_or_grid : Referenz-Waveform (Zeitbereich) ODER PRS-Grid (14×Nsc)
                           py3gpp nimmt das Grid, Eigenimpl. nimmt die Waveform
    sample_rate          : Abtastrate [Hz]
    max_delay_samples    : Maximale Korrelationslänge (analog Matlab-Truncation)

    Returns
    -------
    delay_samples : int
    delay_s       : float [s]
    corr_mag      : np.ndarray – Korrelationsbetrag für Plot
    """
    if max_delay_samples is None:
        max_delay_samples = int(config.NFFT * config.SCS_HZ / 15e3)

    if PY3GPP_TIMING and isinstance(ref_waveform_or_grid, np.ndarray) and ref_waveform_or_grid.ndim == 2:
        return _timing_py3gpp(rx_waveform, ref_waveform_or_grid,
                               sample_rate, max_delay_samples, slot)
    else:
        return _timing_eigen(rx_waveform, ref_waveform_or_grid,
                             sample_rate, max_delay_samples)


def _timing_py3gpp(rx_waveform, prs_grid, sample_rate,
                   max_delay_samples, slot) -> tuple:
    """
    py3gpp.nrTimingEstimate(carrier, rxWaveform, refGrid)
    refGrid hat Format (Nsc, 14) in py3gpp.
    """
    carrier = _make_carrier(slot)

    # py3gpp erwartet Grid als (Nsc, 14)
    ref_grid_py3gpp = prs_grid.T   # (14, Nsc) → (Nsc, 14)

    # nrTimingEstimate gibt (offset, mag) zurück
    offset, mag = nrTimingEstimate(carrier, rx_waveform, ref_grid_py3gpp)

    # Truncation analog zu Matlab: mag(1:Nfft*SCS/15)
    corr_trunc    = np.abs(np.array(mag, dtype=np.complex64))[:max_delay_samples]
    delay_samples = int(np.argmax(corr_trunc))
    delay_s       = delay_samples / sample_rate

    return delay_samples, delay_s, corr_trunc


def _timing_eigen(rx_waveform, ref_waveform, sample_rate,
                  max_delay_samples) -> tuple:
    """Kreuzkorrelation als Fallback."""
    corr_full   = np.correlate(rx_waveform, ref_waveform, mode='full')
    mid         = len(ref_waveform) - 1
    corr_causal = np.abs(corr_full[mid:])
    corr_trunc  = corr_causal[:max_delay_samples]

    delay_samples = int(np.argmax(corr_trunc))
    delay_s       = delay_samples / sample_rate
    return delay_samples, delay_s, corr_trunc


def getRSTDValues(delay_estimates: np.ndarray,
                  sample_rate: float) -> np.ndarray:
    """
    RSTD-Matrix berechnen.
    Analog zu: rstdVals = getRSTDValues(delayEst, ofdmInfo.SampleRate)
    """
    n    = len(delay_estimates)
    rstd = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            rstd[i, j] = (delay_estimates[i] - delay_estimates[j]) / sample_rate
    return rstd
