# =============================================================================
#  ran/simulated.py – Simulierter Kanal für DL-OTDOA
#  Analog zu: "Add Signal Delays and Apply Path Loss" im Matlab-Beispiel
#
#  Matlab:
#    radius{gNBIdx} = rangeangle(gNBPos{gNBIdx}', UEPos');
#    delay = radius / speedOfLight;
#    sampleDelay = round(delay * ofdmInfo.SampleRate);
#    rx = [zeros(sampleDelay,1); txWaveform; zeros(pad,1)] / sqrt(PL);
#    rxWaveform = sum(rx von allen gNBs)
# =============================================================================

import numpy as np
import config
from ran.base import DataSource


class SimulatedSource(DataSource):
    """
    Simulierter Kanal für DL-OTDOA mit PRS.

    Implementiert das Matlab-Kanalmodell:
    - Geometrischer LOS-Delay (sample-genau, dann gerundet wie in Matlab)
    - UMa Pathloss (nrPathLoss analog)
    - Additives weißes Rauschen

    KEIN Multipath-TDL – analog zum Matlab-Beispiel das nur LOS nutzt:
    "losFlag = true; % Assuming LOS flag as true, as we only consider LOS path delays"
    """

    def __init__(self, ue_pos: np.ndarray = None, rng_seed: int = config.RNG_SEED):
        self._rng   = np.random.default_rng(rng_seed)
        self._txPos = self._create_gnb_positions()
        self._rxPos = (ue_pos if ue_pos is not None
                       else config.UE_POS).reshape(1, 3)
        self._sample_rate = config.SCS_HZ * config.NFFT
        self._true_delays, self._sample_delays = self._compute_delays()

        print(f"[SimulatedSource] {self.n_bs} gNBs, UE @ "
              f"({self._rxPos[0,0]:.0f},{self._rxPos[0,1]:.0f},{self._rxPos[0,2]:.0f})m")
        print(f"[SimulatedSource] Abtastrate: {self._sample_rate/1e6:.2f} MHz")
        print(f"[SimulatedSource] Delays [Samples]: {self._sample_delays}")

    # -------------------------------------------------------------------------
    #  DataSource Interface
    # -------------------------------------------------------------------------

    def get_channel_matrix(self) -> np.ndarray:
        """Nicht verwendet in DL-OTDOA – Kanal wird über Waveform modelliert."""
        return None

    def get_true_delays(self) -> np.ndarray:
        return self._true_delays.reshape(-1, 1)

    def get_bs_positions(self) -> np.ndarray:
        return self._txPos.copy()

    def get_ue_positions(self) -> np.ndarray:
        return self._rxPos.copy()

    @property
    def n_bs(self) -> int:
        return len(self._txPos)

    @property
    def n_ue(self) -> int:
        return 1

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def sample_delays(self) -> np.ndarray:
        return self._sample_delays

    # -------------------------------------------------------------------------
    #  Kernfunktion: rxWaveform erzeugen
    #  Analog zu Matlab: "Model the received signal at the UE"
    # -------------------------------------------------------------------------

    def get_rx_waveform(self, tx_waveforms: list) -> np.ndarray:
        """
        Erzeugt empfangenes Summensignal am UE.

        Analog zu Matlab:
            rxWaveform = zeros(len(txWaveform{1}) + max(sampleDelay), 1)
            for gNBIdx:
                PL = nrPathLoss(...)
                rx = [zeros(sampleDelay,1); txWaveform; zeros(pad,1)] / sqrt(PL)
                rxWaveform += rx

        Parameters
        ----------
        tx_waveforms : list of np.ndarray – gesendete Waveforms je gNB

        Returns
        -------
        rx_waveform : np.ndarray – Summensignal am UE + Rauschen
        """
        max_delay = int(self._sample_delays.max())
        waveform_len = len(tx_waveforms[0])
        rx_total = np.zeros(waveform_len + max_delay, dtype=np.complex64)

        for nbs, tx in enumerate(tx_waveforms):
            d   = int(self._sample_delays[nbs])
            pad = max_delay - d
            pl  = self._pathloss_linear(nbs)

            # Delay + Padding + Dämpfung (analog zu Matlab)
            rx = np.concatenate([
                np.zeros(d,   dtype=np.complex64),
                tx.astype(np.complex64),
                np.zeros(pad, dtype=np.complex64)
            ]) / np.sqrt(pl)

            rx_total += rx

        # AWGN hinzufügen
        noise_std = np.sqrt(config.NOISE_POWER_W / 2)
        rx_total += noise_std * (
            self._rng.standard_normal(len(rx_total)) +
            1j * self._rng.standard_normal(len(rx_total))
        ).astype(np.complex64)

        return rx_total

    # -------------------------------------------------------------------------
    #  Hilfsmethoden
    # -------------------------------------------------------------------------

    def _create_gnb_positions(self) -> np.ndarray:
        """
        Zufällige gNB-Positionen analog zu getgNBPositions() in Matlab.

        Matlab:
            phi = gNBIdx*2*pi/numgNBs + rand()*2*pi/(2*numgNBs) - 2*pi/(2*numgNBs)
            r   = randi([0,1000]) + 4000 + gNBIdx*5000/numgNBs
            gNBPos = [x, y, 25]
        """
        n = config.N_BS
        rng = np.random.default_rng(config.RNG_SEED)
        positions = np.array([[ 50.0,  0.0, 25.0], [15.5,  47.5, 25.0],
                        [-40.45, 29.38, 25.0], [-40.45,  -29.38, 25.0], [15.5,  -47.5, 25.0]])
        '''for idx in range(1, n+1):
            phi = (idx * 2*np.pi/n
                   + rng.uniform() * 2*np.pi/(2*n)
                   - 2*np.pi/(2*n))
            r   = rng.integers(0, 101) + idx * 50/n
            x   = r * np.cos(phi)
            y   = r * np.sin(phi)
            positions.append([x, y, config.BS_HEIGHT_M])'''
        return np.array(positions)

    def _compute_delays(self):
        """
        Analog zu Matlab:
            radius = rangeangle(gNBPos', UEPos')  → euklidischer Abstand
            delay  = radius / speedOfLight
            sampleDelay = round(delay * sampleRate)
        """
        ue = self._rxPos[0]
        delays = np.zeros(self.n_bs)
        for nbs in range(self.n_bs):
            d3d = np.linalg.norm(self._txPos[nbs] - ue)
            delays[nbs] = d3d / config.C
        sample_delays = np.round(delays * self._sample_rate).astype(int)
        return delays, sample_delays

    def _pathloss_linear(self, nbs: int) -> float:
        """
        UMa LOS Pathloss analog zu nrPathLoss(plCfg, fc, losFlag, gNBPos, UEPos).
        3GPP TR 38.901 Tabelle 7.4.1-1, UMa LOS.
        """
        ue   = self._rxPos[0]
        d3d  = np.linalg.norm(self._txPos[nbs] - ue)
        fc   = config.CARRIER_FREQUENCY_HZ / 1e9
        d3d  = max(d3d, 1.0)
        pl_db = 32.4 + 20*np.log10(fc) + 21*np.log10(d3d)
        return 10**(pl_db/10)
