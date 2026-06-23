# =============================================================================
#  signal_processing/toa_estimation.py
#  ToA via Kreuzkorrelation – analog zu nrTimingEstimate()
#
#  Matlab nutzt nrTimingEstimate(carrier, rxWaveform, prsGrid):
#  → Kreuzkorrelation der empfangenen Waveform mit dem Referenz-PRS-Grid
#  → Peak der Korrelation = ToA in Samples → ToA in Sekunden
#
#  Das ist robuster als IFFT weil:
#  - Kein Multipath-Bias: Korrelation findet den ERSTEN Peak (LOS)
#  - Kein Frequenzbereichs-Artefakte durch Windowing
#  - Direkt im Zeitbereich → arbeitet auf IQ-Samples
# =============================================================================

import numpy as np
import config


def nrTimingEstimate(rx_waveform: np.ndarray,
                     ref_waveform: np.ndarray,
                     sample_rate: float,
                     max_delay_samples: int = None) -> tuple:
    """
    ToA-Schätzung via Kreuzkorrelation.
    Analog zu: [offset, mag] = nrTimingEstimate(carrier, rxWaveform, prsGrid)

    Matlab-Code:
        [~,mag] = nrTimingEstimate(carrier, rxWaveform, prsGrid{gNBIdx});
        corr{gNBIdx} = mag(1:(ofdmInfo.Nfft*carrier.SubcarrierSpacing/15));
        delayEst(gNBIdx) = find(corr == max(corr), 1) - 1;

    Parameters
    ----------
    rx_waveform  : Empfangenes IQ-Signal (Zeitbereich), shape (N,)
    ref_waveform : Referenz-PRS Waveform (Zeitbereich), shape (M,)
    sample_rate  : Abtastrate [Hz] = SCS * NFFT
    max_delay_samples : Maximale Korrelationslänge (analog zu Matlab-Truncation)
                        Matlab nutzt: Nfft * SCS/15 Samples

    Returns
    -------
    delay_samples : int   – Verzögerung in Samples (Peak der Korrelation)
    delay_s       : float – Verzögerung in Sekunden
    corr_mag      : np.ndarray – Korrelationsbetrag (für Plot)
    """
    # Kreuzkorrelation: corr[k] = Σ rx[n+k] · conj(ref[n])
    # Äquivalent zu Matlab's nrTimingEstimate im Zeitbereich
    corr_full = np.correlate(rx_waveform, ref_waveform, mode='full')
    corr_mag  = np.abs(corr_full)

    # Nur kausale Seite (Delays ≥ 0)
    mid       = len(ref_waveform) - 1
    corr_causal = corr_mag[mid:]

    # Truncation analog zu Matlab:
    # mag(1 : Nfft * SubcarrierSpacing/15)
    if max_delay_samples is None:
        max_delay_samples = int(config.NFFT * config.SCS_HZ / 15e3)
    corr_trunc = corr_causal[:max_delay_samples]

    # Peak = ToA-Schätzung
    delay_samples = int(np.argmax(corr_trunc))
    delay_s       = delay_samples / sample_rate

    return delay_samples, delay_s, corr_trunc


def getRSTDValues(delay_estimates: np.ndarray,
                  sample_rate: float) -> np.ndarray:
    """
    Berechnet RSTD-Matrix aus ToA-Schätzwerten.
    Analog zu: rstdVals = getRSTDValues(delayEst, ofdmInfo.SampleRate)

    RSTD(i,j) = TOA(i) - TOA(j)  [in Sekunden]

    Parameters
    ----------
    delay_estimates : np.ndarray, shape (n_bs,) – Delays in Samples
    sample_rate     : float – Abtastrate [Hz]

    Returns
    -------
    rstd : np.ndarray, shape (n_bs, n_bs) – RSTD in Sekunden
    """
    n = len(delay_estimates)
    rstd = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            rstd[i, j] = (delay_estimates[i] - delay_estimates[j]) / sample_rate
    return rstd
