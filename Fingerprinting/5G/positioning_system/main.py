# =============================================================================
#  main.py – Orchestrierung DL-OTDOA
#
#  Klare Trennung:
#  ┌─────────────────────┐     Kanal      ┌──────────────────────┐
#  │  GnbTransmitter     │ ─────────────► │  UeReceiver          │
#  │  gnb/transmitter.py │                │  ue/receiver.py      │
#  │                     │                │                      │
#  │  • nrPRS()          │                │  • nrTimingEstimate()│
#  │  • nrPDSCH()        │                │  • getRSTDValues()   │
#  │  • ofdm_modulate()  │                │  • estimate_pos()    │
#  └─────────────────────┘                └──────────────────────┘
# =============================================================================

import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from gnb.transmitter  import GnbConfig, GnbTransmitter
from ue.receiver      import UeReceiver
from signal_processing.pathloss import nrPathLoss
from visualization.plots import (plotgNBAndUEPositions,
                                 plotPRSCorr,
                                 plotPositionsAndHyperbolaCurves,
                                 plot_prs_all)


def create_data_source():
    if config.MODE == "simulated":
        from ran.simulated import SimulatedSource
        return SimulatedSource(ue_pos=config.UE_POS)
    elif config.MODE == "srsran":
        from ran.srsran import SrsRanSource
        bs_positions = np.array([
            [ 0.0,  0.0, 3.0], [20.0,  0.0, 3.0],
            [10.0, 15.0, 3.0], [20.0, 15.0, 3.0],
            [ 0.0, 15.0, 3.0],
        ])
        return SrsRanSource(bs_positions, None)
    raise ValueError(f"Unbekannter Modus: {config.MODE!r}")


def run():
    print("\n" + "="*62)
    print("  NR Positioning Using PRS – DL-OTDOA")
    print(f"  Modus: {config.MODE.upper()} | "
          f"fc={config.CARRIER_FREQUENCY_HZ/1e9:.1f} GHz | "
          f"CombSize={config.PRS_COMB_SIZE}")
    print("="*62 + "\n")

    # =========================================================================
    #  1. Datenquelle – liefert Topologie und Kanal
    # =========================================================================
    source        = create_data_source()
    gnb_pos       = source.get_bs_positions()
    ue_pos        = source.get_ue_positions()[0]
    true_delays   = source.get_true_delays()[:, 0]
    sample_rate   = source.sample_rate
    sample_delays = source.sample_delays

    print(f"UE-Position : ({ue_pos[0]:.1f}, {ue_pos[1]:.1f}, {ue_pos[2]:.1f}) m")

    # Pathloss anzeigen
    print("\n--- Pathloss (nrPathLoss) ---")
    for nbs in range(source.n_bs):
        pl  = nrPathLoss(config.CARRIER_FREQUENCY_HZ, gnb_pos[nbs], ue_pos,
                         scenario=config.PATHLOSS_SCENARIO)
        d3d = np.linalg.norm(gnb_pos[nbs] - ue_pos)
        print(f"  gNB{nbs+1}: {d3d:.0f}m | {pl:.1f} dB")

    plotgNBAndUEPositions(gnb_pos, ue_pos)

    # PRS Slot-Konfiguration visualisieren
    plot_prs_all(n_bs=config.N_BS)

    # =========================================================================
    #  2. gNB-Konfigurationen erstellen
    #     analog zu: nrPRSConfig + prsIDs + prsSlotOffsets
    # =========================================================================
    rng      = np.random.default_rng(config.RNG_SEED)
    nprs_ids = rng.choice(4096, size=config.N_BS, replace=False).tolist()

    gnb_configs = [
        GnbConfig(
            gnb_idx     = i,
            nprs_id     = nprs_ids[i],
            slot_offset = config.PRS_SLOT_OFFSETS[i],
            position    = gnb_pos[i],
        )
        for i in range(config.N_BS)
    ]
    print(f"\nNPRS_IDs: {nprs_ids}")

    # =========================================================================
    #  3. gNB SENDET: PRS + PDSCH + OFDM-Modulation
    # =========================================================================
    transmitter  = GnbTransmitter(gnb_configs)
    tx_waveforms = transmitter.generate()  # ← komplette gNB-Signalgenerierung

    # =========================================================================
    #  4. KANAL: Delay + Pathloss + Rauschen
    # =========================================================================
    print("\n--- Kanal: Delay + Pathloss + Rauschen ---")
    rx_waveform = source.get_rx_waveform(tx_waveforms)
    print(f"  RX-Waveform: {len(rx_waveform)} Samples")

    # =========================================================================
    #  5. UE EMPFÄNGT + SERVER BERECHNET POSITION
    # =========================================================================
    receiver = UeReceiver(
        gnb_positions   = gnb_pos,
        cells_to_detect = config.CELLS_TO_DETECT,
    )
    est_pos = receiver.process(
        rx_waveform   = rx_waveform,
        prs_grids     = transmitter.prs_grids,  # NUR PRS (kein PDSCH)
        sample_rate   = sample_rate,
        sample_delays = sample_delays,           # für Evaluierung
    )

    # =========================================================================
    #  6. Ergebnisse
    # =========================================================================
    est_err = np.linalg.norm(est_pos[:2] - ue_pos[:2])
    print(f"\n{'='*55}")
    print(f"  Wahre UE-Position         : [{ue_pos[0]:.1f}, {ue_pos[1]:.1f}]")
    print(f"  Geschätzte UE-Position    : [{est_pos[0]:.1f}, {est_pos[1]:.1f}]")
    print(f"  Positionierungsfehler     : {est_err:.2f} m")
    print(f"{'='*55}\n")

    plotPRSCorr(receiver.corr_list, sample_rate,
                [f"gNB{i+1} (NPRSID={nprs_ids[i]})"
                 for i in range(config.N_BS)])

    plotPositionsAndHyperbolaCurves(
        gnb_pos, ue_pos, est_pos,
        receiver.curve_x, receiver.curve_y,
        receiver.gnb_pairs, receiver.detected_gnbs,
    )
    plt.show()
    return est_pos, est_err


if __name__ == "__main__":
    run()
