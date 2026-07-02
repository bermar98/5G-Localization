# =============================================================================
#  ue/receiver.py – UE + Positioning Server Signalverarbeitung
#
#  Verantwortlich für ALLES was das UE und der Server machen:
#  - RX-Waveform empfangen
#  - ToA-Schätzung via Kreuzkorrelation (nrTimingEstimate)
#  - RSTD berechnen (getRSTDValues)
#  - Positionsschätzung via Hyperbeln (getRSTDCurve + getEstimatedUEPosition)
#
#  Analog zu Matlab:
#    nrTimingEstimate → getRSTDValues → getRSTDCurve → getEstimatedUEPosition
# =============================================================================

import numpy as np
import config
from signal_processing.ofdm           import ofdm_modulate
from signal_processing.toa_estimation import nrTimingEstimate, getRSTDValues
from positioning.tdoa_ls              import estimate_position_otdoa


class UeReceiver:
    """
    Kapselt die komplette UE-seitige Signalverarbeitung und Positionsschätzung.

    Trennung vom Transmitter:
    - Kennt NICHT wie das Signal generiert wurde
    - Bekommt nur: rx_waveform + prs_grids (als Referenz für Korrelation)
    - Gibt zurück: geschätzte Position + Korrelationsdaten
    """

    def __init__(self, gnb_positions: np.ndarray,
                 cells_to_detect: int = config.CELLS_TO_DETECT):
        """
        Parameters
        ----------
        gnb_positions  : np.ndarray (n_bs, 3) – bekannte gNB-Positionen
        cells_to_detect: Anzahl gNBs für TDoA (min. 3 für 2D)
        """
        self.gnb_positions   = gnb_positions
        self.cells_to_detect = cells_to_detect
        self.n_bs            = len(gnb_positions)

        # Ergebnisse (nach process() befüllt)
        self.delay_est    = None
        self.max_corr_val = None
        self.corr_list    = None
        self.rstd_matrix  = None
        self.detected_gnbs = None
        self.est_pos      = None
        self.curve_x      = None
        self.curve_y      = None
        self.gnb_pairs    = None

    def process(self, rx_waveform: np.ndarray,
                prs_grids: list,
                sample_rate: float,
                sample_delays: np.ndarray = None) -> np.ndarray:
        """
        Vollständige UE-seitige Verarbeitung.

        Parameters
        ----------
        rx_waveform   : Empfangenes IQ-Signal (Summensignal aller gNBs)
        prs_grids     : list of np.ndarray – PRS-Referenzgrids pro gNB
                        (nur PRS, kein PDSCH – für Kreuzkorrelation)
        sample_rate   : Abtastrate [Hz]
        sample_delays : Wahre Delays in Samples (nur für Evaluierung, optional)

        Returns
        -------
        est_pos : np.ndarray (3,) – geschätzte UE-Position [x, y, z]
        """
        self._estimate_toa(rx_waveform, prs_grids, sample_rate, sample_delays)
        self._select_best_gnbs()
        self._compute_rstd(sample_rate)
        self._estimate_position()
        return self.est_pos

    def _estimate_toa(self, rx_waveform, prs_grids, sample_rate, sample_delays):
        """
        Schritt 1: ToA-Schätzung via Kreuzkorrelation pro gNB.
        Analog zu: [~,mag] = nrTimingEstimate(carrier, rxWaveform, prsGrid)
        """
        max_corr_len = int(config.NFFT * config.SCS_HZ / 15e3)

        self.delay_est    = np.zeros(self.n_bs, dtype=int)
        self.max_corr_val = np.zeros(self.n_bs)
        self.corr_list    = []

        print("\n--- UE: ToA-Schätzung (Kreuzkorrelation) ---")

        for gnb_idx in range(self.n_bs):
            # Referenz: NUR das PRS-Grid dieser gNB (kein PDSCH)
            ref_waveform = ofdm_modulate(prs_grids[gnb_idx], config.NFFT)

            d_samples, d_s, corr = nrTimingEstimate(
                rx_waveform, ref_waveform, sample_rate, max_corr_len)

            self.delay_est[gnb_idx]    = d_samples
            self.max_corr_val[gnb_idx] = corr.max()
            self.corr_list.append(corr)

            # Fehler berechnen (nur wenn wahre Delays bekannt)
            if sample_delays is not None:
                err_m = (abs(d_samples - sample_delays[gnb_idx])
                         / sample_rate * config.C)
                print(f"  gNB{gnb_idx+1}: "
                      f"Est={d_samples:4d} | Wahr={sample_delays[gnb_idx]:4d} | "
                      f"Fehler={err_m:.1f}m | Korr={corr.max():.3f}")
            else:
                print(f"  gNB{gnb_idx+1}: "
                      f"Est={d_samples:4d} Samples | "
                      f"Korr={corr.max():.3f}")

    def _select_best_gnbs(self):
        """
        Schritt 2: Beste gNBs nach Korrelationsstärke wählen.
        Analog zu: [~,detectedgNBs] = sort(maxCorr,'descend')
        """
        k = min(self.cells_to_detect, self.n_bs)
        self.detected_gnbs = np.argsort(self.max_corr_val)[::-1][:k].tolist()
        print(f"\n--- UE: Detektierte gNBs ---")
        print(f"  {[f'gNB{i+1}' for i in self.detected_gnbs]}")
        print(f"  Referenz-gNB: gNB{self.detected_gnbs[0]+1}")

    def _compute_rstd(self, sample_rate):
        """
        Schritt 3: RSTD-Matrix berechnen.
        Analog zu: rstdVals = getRSTDValues(delayEst, sampleRate)
        """
        self.rstd_matrix = getRSTDValues(self.delay_est, sample_rate)

    def _estimate_position(self):
        """
        Schritt 4: Positionsschätzung via Hyperbel-Multilateration.
        Analog zu: getRSTDCurve + getEstimatedUEPosition
        """
        print("\n--- Server: Positionsschätzung (Hyperbel-Multilateration) ---")
        ref_idx      = self.detected_gnbs[0]
        neighbor_idx = self.detected_gnbs[1:]

        self.est_pos, self.curve_x, self.curve_y, self.gnb_pairs = \
            estimate_position_otdoa(
                gnb_positions = self.gnb_positions,
                rstd_matrix   = self.rstd_matrix,
                ref_gnb_idx   = ref_idx,
                neighbor_idxs = neighbor_idx,
            )
