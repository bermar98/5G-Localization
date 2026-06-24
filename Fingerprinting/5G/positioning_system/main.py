# =============================================================================
#  main.py – DL-OTDOA mit PRS + PDSCH + Pathloss
#  Direkter Python-Port des Matlab-Beispiels "NR Positioning Using PRS"
#
#  Matlab → Python:
#  ─────────────────────────────────────────────────────────────────────
#  nrPRSConfig / nrPRS()          → prs_generator.py    :: nrPRS()
#  nrPDSCHConfig / nrPDSCH()      → pdsch_generator.py  :: nrPDSCH()
#  nrPathLossConfig / nrPathLoss() → pathloss.py         :: nrPathLoss()
#  nrOFDMModulate()               → ofdm.py             :: ofdm_modulate()
#  nrTimingEstimate()             → toa_estimation.py   :: nrTimingEstimate()
#  getRSTDValues()                → toa_estimation.py   :: getRSTDValues()
#  getRSTDCurve()                 → tdoa_ls.py          :: getRSTDCurve()
#  getEstimatedUEPosition()       → tdoa_ls.py          :: getEstimatedUEPosition()
# =============================================================================

import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from signal_processing.prs_generator   import nrPRS
from signal_processing.pdsch_generator import nrPDSCH_grid as nrPDSCH, should_transmit_pdsch
from signal_processing.pathloss        import nrPathLoss, nrPathLoss_linear
from signal_processing.ofdm            import ofdm_modulate
from signal_processing.toa_estimation  import nrTimingEstimate, getRSTDValues
from positioning.tdoa_ls               import estimate_position_otdoa
from visualization.plots               import (plotgNBAndUEPositions,
                                               plotPRSCorr,
                                               plotPositionsAndHyperbolaCurves)


def create_data_source():
    if config.MODE == "simulated":
        from ran.simulated import SimulatedSource
        return SimulatedSource(ue_pos=config.UE_POS)
    elif config.MODE == "srsran":
        from ran.srsran import SrsRanSource
        bs_positions = np.array([
            [ 0.0,  0.0, 3.0],
            [20.0,  0.0, 3.0],
            [10.0, 15.0, 3.0],
            [20.0, 15.0, 3.0],
            [ 0.0, 15.0, 3.0],
        ])
        return SrsRanSource(bs_positions, None)
    else:
        raise ValueError(f"Unbekannter Modus: {config.MODE!r}")


def run():
    print("\n" + "="*62)
    print("  NR Positioning Using PRS – Python Port des Matlab-Beispiels")
    print(f"  Modus    : {config.MODE.upper()}")
    print(f"  Szenario : {config.PATHLOSS_SCENARIO} | fc={config.CARRIER_FREQUENCY_HZ/1e9:.1f} GHz")
    print(f"  PRS      : CombSize={config.PRS_COMB_SIZE} | Symbole={config.PRS_NUM_SYMBOLS}")
    print("="*62 + "\n")

    # =========================================================================
    #  1. Datenquelle
    # =========================================================================
    source      = create_data_source()
    gnb_pos     = source.get_bs_positions()
    ue_pos      = source.get_ue_positions()[0]
    true_delays = source.get_true_delays()[:, 0]
    sample_rate = source.sample_rate
    sample_delays = source.sample_delays

    print(f"UE-Position : ({ue_pos[0]:.1f}, {ue_pos[1]:.1f}, {ue_pos[2]:.1f}) m")
    print(f"Wahre Delays: {(true_delays*1e6).round(3)} µs\n")

    # Pathloss pro gNB ausgeben (wie Matlab: PLdB für jede gNB)
    print("--- Pathloss (analog: nrPathLoss) ---")
    for nbs in range(source.n_bs):
        pl_db = nrPathLoss(
            config.CARRIER_FREQUENCY_HZ,
            gnb_pos[nbs], ue_pos,
            scenario = config.PATHLOSS_SCENARIO,
            los      = True,
        )
        d3d = np.linalg.norm(gnb_pos[nbs] - ue_pos)
        print(f"  gNB{nbs+1}: Distanz={d3d:.0f}m | Pathloss={pl_db:.1f} dB")

    # Plot Topologie
    plotgNBAndUEPositions(gnb_pos, ue_pos)

    # =========================================================================
    #  2. PRS-IDs (analog: prsIDs = randperm(4096, numgNBs) - 1)
    # =========================================================================
    rng = np.random.default_rng(config.RNG_SEED)
    nprs_ids = rng.choice(4096, size=config.N_BS, replace=False).tolist()
    print(f"\nNPRS_IDs: {nprs_ids}")

    # =========================================================================
    #  3. PRS + PDSCH Grid generieren (Slot-Schleife)
    #     Analog zur Matlab Slot-Schleife:
    #       for slotIdx = 0:totSlots-1
    #         PRS in zugewiesenem Slot → prsGrid
    #         PDSCH in Slots ohne PRS → dataGrid
    # =========================================================================
    print("\n--- PRS + PDSCH Grid generieren ---")

    total_slots = config.N_FRAMES * config.SLOTS_PER_FRAME  # 10 Slots
    prs_grids   = [np.zeros((14, config.NSC), dtype=np.complex64)
                   for _ in range(config.N_BS)]
    data_grids  = [np.zeros((14, config.NSC), dtype=np.complex64)
                   for _ in range(config.N_BS)]

    for slot_idx in range(total_slots):

        # --- PRS: jede gNB in eigenem Slot ---
        for gnb_idx in range(config.N_BS):
            if slot_idx == config.PRS_SLOT_OFFSETS[gnb_idx]:
                prs_grid, _, _ = nrPRS(
                    Nsc          = config.NSC,
                    nprs_id      = nprs_ids[gnb_idx],
                    slot         = slot_idx,
                    comb_size    = config.PRS_COMB_SIZE,
                    num_symbols  = config.PRS_NUM_SYMBOLS,
                    symbol_start = config.PRS_SYMBOL_START,
                    num_rbs      = config.PRS_NUM_RBS,
                    rb_offset    = config.PRS_RB_OFFSET,
                )
                prs_grids[gnb_idx] = prs_grid
                print(f"  Slot {slot_idx:2d}: gNB{gnb_idx+1} → PRS "
                      f"(NPRSID={nprs_ids[gnb_idx]})")

        # --- PDSCH: nur in Slots ohne PRS (analog zu Matlab) ---
        if should_transmit_pdsch(slot_idx, config.PRS_SLOT_OFFSETS):
            for gnb_idx in range(config.N_BS):
                data_grid, _, _ = nrPDSCH(
                    slot     = slot_idx,
                    rng_seed = gnb_idx * 1000 + slot_idx,
                )
                data_grids[gnb_idx] = data_grid
            print(f"  Slot {slot_idx:2d}: alle gNBs → PDSCH")

    # =========================================================================
    #  4. OFDM-Modulation (analog: nrOFDMModulate)
    #     PRS + PDSCH werden addiert wie in Matlab:
    #     txWaveform = nrOFDMModulate(carrier, prsGrid + dataGrid)
    # =========================================================================
    print("\n--- OFDM-Modulation (PRS + PDSCH) ---")
    tx_waveforms = []
    for gnb_idx in range(config.N_BS):
        # PRS + PDSCH addieren (analog zu Matlab: prsGrid + dataGrid)
        combined_grid = prs_grids[gnb_idx] + data_grids[gnb_idx]
        waveform = ofdm_modulate(combined_grid, config.NFFT)
        tx_waveforms.append(waveform)
        print(f"  gNB{gnb_idx+1}: Waveform {len(waveform)} Samples "
              f"| PRS-Slot={config.PRS_SLOT_OFFSETS[gnb_idx]}")

    # =========================================================================
    #  5. Kanal: Delay + Pathloss + Rauschen
    #     analog zu: rx = [zeros; txWaveform; zeros] / sqrt(PL)
    #                rxWaveform = sum(rx)
    # =========================================================================
    print("\n--- Kanal: Delay + Pathloss + Rauschen ---")
    rx_waveform = source.get_rx_waveform(tx_waveforms)
    print(f"  RX-Waveform: {len(rx_waveform)} Samples")

    # =========================================================================
    #  6. ToA-Schätzung via Kreuzkorrelation
    #     analog zu: nrTimingEstimate(carrier, rxWaveform, prsGrid)
    #     Wichtig: Referenz ist NUR das PRS-Grid (ohne PDSCH)
    # =========================================================================
    print("\n--- ToA-Schätzung (Kreuzkorrelation) ---")
    max_corr_len = int(config.NFFT * config.SCS_HZ / 15e3)
    corr_list    = []
    delay_est    = np.zeros(config.N_BS, dtype=int)
    max_corr_val = np.zeros(config.N_BS)

    for gnb_idx in range(config.N_BS):
        # Referenz: NUR PRS (kein PDSCH) – analog zu Matlab: prsGrid{gNBIdx}
        ref_waveform = ofdm_modulate(prs_grids[gnb_idx], config.NFFT)

        d_samples, d_s, corr = nrTimingEstimate(
            rx_waveform, ref_waveform, sample_rate, max_corr_len)

        corr_list.append(corr)
        delay_est[gnb_idx]    = d_samples
        max_corr_val[gnb_idx] = corr.max()

        err_m = abs(d_samples - sample_delays[gnb_idx]) / sample_rate * config.C
        print(f"  gNB{gnb_idx+1}: "
              f"Est={d_samples:4d} Samples | "
              f"Wahr={sample_delays[gnb_idx]:4d} | "
              f"Fehler={err_m:.1f} m | "
              f"maxCorr={corr.max():.3f}")

    # Plot Korrelation
    plotPRSCorr(corr_list, sample_rate,
                [f"gNB{i+1} (NPRSID={nprs_ids[i]})" for i in range(config.N_BS)])

    # =========================================================================
    #  7. Detektierte gNBs (analog: sort(maxCorr,'descend'))
    # =========================================================================
    cells_to_detect = min(config.CELLS_TO_DETECT, config.N_BS)
    detected_gnbs   = np.argsort(max_corr_val)[::-1][:cells_to_detect].tolist()
    print(f"\nDetektierte gNBs: {[f'gNB{i+1}' for i in detected_gnbs]}")
    print(f"Referenz-gNB    : gNB{detected_gnbs[0]+1}")

    # =========================================================================
    #  8. RSTD berechnen (analog: getRSTDValues)
    # =========================================================================
    rstd_matrix = getRSTDValues(delay_est, sample_rate)

    # =========================================================================
    #  9. Positionsschätzung via Hyperbeln
    #     analog zu: getRSTDCurve + getEstimatedUEPosition
    # =========================================================================
    print("\n--- Positionsschätzung (Hyperbel-Multilateration) ---")
    ref_idx      = detected_gnbs[0]
    neighbor_idx = detected_gnbs[1:]

    est_pos, curve_x, curve_y, gnb_pairs = estimate_position_otdoa(
        gnb_positions = gnb_pos,
        rstd_matrix   = rstd_matrix,
        ref_gnb_idx   = ref_idx,
        neighbor_idxs = neighbor_idx,
    )

    # =========================================================================
    #  10. Ergebnisse
    # =========================================================================
    est_err = np.linalg.norm(est_pos[:2] - ue_pos[:2])
    print(f"\n  Wahre UE-Position          : [{ue_pos[0]:.1f}, {ue_pos[1]:.1f}]")
    print(f"  Geschätzte UE-Position     : [{est_pos[0]:.1f}, {est_pos[1]:.1f}]")
    print(f"  UE Position Estimation Error: {est_err:.2f} meters")

    # Plot Hyperbeln
    plotPositionsAndHyperbolaCurves(
        gnb_pos, ue_pos, est_pos,
        curve_x, curve_y, gnb_pairs, detected_gnbs
    )
    plt.show()
    return est_pos, est_err


if __name__ == "__main__":
    run()
