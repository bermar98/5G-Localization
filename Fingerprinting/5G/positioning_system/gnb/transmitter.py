# =============================================================================
#  gnb/transmitter.py – gNB Signalgenerierung
#
#  Verantwortlich für ALLES was die gNB macht:
#  - PRS Grid generieren  (nrPRS)
#  - PDSCH Grid generieren (nrPDSCH)
#  - OFDM Modulation      (nrOFDMModulate)
#  - Slot-Koordination    (wer sendet wann)
#
#  Analog zu Matlab:
#    nrPRS + nrPDSCH + nrOFDMModulate in der Slot-Schleife
# =============================================================================

import numpy as np
import config
from signal_processing.prs_generator   import nrPRS
from signal_processing.pdsch_generator import nrPDSCH_grid as nrPDSCH, should_transmit_pdsch
from signal_processing.ofdm            import ofdm_modulate


class GnbConfig:
    """
    Konfiguration einer einzelnen gNB.
    Analog zu nrCarrierConfig + nrPRSConfig in Matlab.
    """
    def __init__(self, gnb_idx: int, nprs_id: int,
                 slot_offset: int, position: np.ndarray):
        self.gnb_idx     = gnb_idx
        self.nprs_id     = nprs_id
        self.slot_offset = slot_offset
        self.position    = np.array(position)

    def __repr__(self):
        return (f"GnbConfig(idx={self.gnb_idx}, "
                f"nprs_id={self.nprs_id}, "
                f"slot={self.slot_offset})")


class GnbTransmitter:
    """
    Kapselt die komplette gNB-Signalgenerierung.

    Analog zu Matlab Slot-Schleife:
        for slotIdx = 0:totSlots-1
            nrPRS → prsGrid
            nrPDSCH → dataGrid
            nrOFDMModulate(prsGrid + dataGrid) → txWaveform
    """

    def __init__(self, gnb_configs: list):
        """
        Parameters
        ----------
        gnb_configs : list of GnbConfig – eine pro gNB
        """
        self.configs     = gnb_configs
        self.n_bs        = len(gnb_configs)
        self._prs_grids  = None
        self._data_grids = None
        self._waveforms  = None

    def generate(self) -> list:
        """
        Generiert PRS + PDSCH Grids und OFDM-Waveforms für alle gNBs.
        Gibt tx_waveforms zurück: list of np.ndarray, eine Waveform pro gNB.
        """
        total_slots  = config.N_FRAMES * config.SLOTS_PER_FRAME
        slot_offsets = [cfg.slot_offset for cfg in self.configs]

        # Leere Grids initialisieren
        self._prs_grids  = [np.zeros((14, config.NSC), dtype=np.complex64)
                            for _ in range(self.n_bs)]
        self._data_grids = [np.zeros((14, config.NSC), dtype=np.complex64)
                            for _ in range(self.n_bs)]

        print("--- gNB: PRS + PDSCH Grid generieren ---")

        # Slot-Schleife (analog zu Matlab: for slotIdx = 0:totSlots-1)
        for slot_idx in range(total_slots):

            # PRS: jede gNB in eigenem Slot
            for cfg in self.configs:
                if slot_idx == cfg.slot_offset:
                    prs_grid, _, _ = nrPRS(
                        Nsc          = config.NSC,
                        nprs_id      = cfg.nprs_id,
                        slot         = slot_idx,
                        comb_size    = config.PRS_COMB_SIZE,
                        num_symbols  = config.PRS_NUM_SYMBOLS,
                        symbol_start = config.PRS_SYMBOL_START,
                        num_rbs      = config.PRS_NUM_RBS,
                        rb_offset    = config.PRS_RB_OFFSET,
                    )
                    self._prs_grids[cfg.gnb_idx] = prs_grid
                    print(f"  Slot {slot_idx:2d}: gNB{cfg.gnb_idx+1} → PRS "
                          f"(NPRSID={cfg.nprs_id})")

            # PDSCH: alle gNBs in Slots ohne PRS
            if should_transmit_pdsch(slot_idx, slot_offsets):
                for cfg in self.configs:
                    data_grid, _, _ = nrPDSCH(
                        slot     = slot_idx,
                        rng_seed = cfg.gnb_idx * 1000 + slot_idx,
                    )
                    self._data_grids[cfg.gnb_idx] = data_grid
                print(f"  Slot {slot_idx:2d}: alle gNBs → PDSCH")

        # OFDM-Modulation: PRS + PDSCH → Waveform
        print("\n--- gNB: OFDM-Modulation ---")
        self._waveforms = []
        for cfg in self.configs:
            combined = self._prs_grids[cfg.gnb_idx] + self._data_grids[cfg.gnb_idx]
            waveform = ofdm_modulate(combined, config.NFFT)
            self._waveforms.append(waveform)
            print(f"  gNB{cfg.gnb_idx+1}: {len(waveform)} Samples "
                  f"| PRS-Slot={cfg.slot_offset}")

        return self._waveforms

    @property
    def prs_grids(self) -> list:
        """PRS-Grids (NUR PRS, kein PDSCH) – für ToA-Referenz am UE."""
        if self._prs_grids is None:
            raise RuntimeError("generate() muss zuerst aufgerufen werden.")
        return self._prs_grids

    @property
    def waveforms(self) -> list:
        """Gesendete Waveforms (PRS + PDSCH) – für Kanalübertragung."""
        if self._waveforms is None:
            raise RuntimeError("generate() muss zuerst aufgerufen werden.")
        return self._waveforms

    @property
    def positions(self) -> np.ndarray:
        """gNB-Positionen als (n_bs, 3) Array."""
        return np.array([cfg.position for cfg in self.configs])
