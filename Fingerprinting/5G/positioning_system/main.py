# =============================================================================
#  main.py – DL-OTDOA mit PRS
#  Direkter Python-Port des Matlab-Beispiels "NR Positioning Using PRS"
#
#  Matlab-Struktur → Python-Äquivalent:
#  ─────────────────────────────────────────────────────────────────
#  nrCarrierConfig          → config.py Parameter
#  nrPRSConfig              → config.py PRS_* Parameter
#  nrPRS() / nrPRSIndices() → signal_processing/prs_generator.py :: nrPRS()
#  nrOFDMModulate()         → signal_processing/ofdm.py :: ofdm_modulate()
#  rangeangle() + Delay     → ran/simulated.py :: get_rx_waveform()
#  nrTimingEstimate()       → signal_processing/toa_estimation.py :: nrTimingEstimate()
#  getRSTDValues()          → signal_processing/toa_estimation.py :: getRSTDValues()
#  getRSTDCurve()           → positioning/tdoa_ls.py :: getRSTDCurve()
#  getEstimatedUEPosition() → positioning/tdoa_ls.py :: getEstimatedUEPosition()
# =============================================================================

import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from signal_processing.prs_generator  import nrPRS
from signal_processing.ofdm           import ofdm_modulate
from signal_processing.toa_estimation import nrTimingEstimate, getRSTDValues
from positioning.tdoa_ls              import estimate_position_otdoa
from visualization.plots              import (plotgNBAndUEPositions,
                                              plotPRSCorr,
                                              plotPositionsAndHyperbolaCurves)


def run():
    print("\n" + "="*62)
    print("  NR Positioning Using PRS – Python Port des Matlab-Beispiels")
    print(f"  fc={config.CARRIER_FREQUENCY_HZ/1e9:.1f} GHz | "
          f"SCS={config.SCS_HZ/1e3:.0f} kHz | "
          f"CombSize={config.PRS_COMB_SIZE} | "
          f"Symbole={config.PRS_NUM_SYMBOLS}")
    print("="*62 + "\n")

    # =========================================================================
    #  1. Topologie (analog: UEPos, gNBPos = getgNBPositions(numgNBs))
    # =========================================================================
    from ran.simulated import SimulatedSource
    source = SimulatedSource(ue_pos=config.UE_POS)

    gnb_pos    = source.get_bs_positions()    # (n_bs, 3)
    ue_pos     = source.get_ue_positions()[0] # (3,)
    true_delays = source.get_true_delays()[:, 0]
    sample_rate = source.sample_rate
    sample_delays = source.sample_delays

    print(f"UE-Position: {ue_pos}")
    print(f"Wahre Delays [µs]: {(true_delays*1e6).round(3)}")

    # Plot Topologie (analog: plotgNBAndUEPositions)
    plotgNBAndUEPositions(gnb_pos, ue_pos)

    # =========================================================================
    #  2. PRS-IDs (analog: prsIDs = randperm(4096, numgNBs) - 1)
    # =========================================================================
    rng = np.random.default_rng(config.RNG_SEED)
    nprs_ids = rng.choice(4096, size=config.N_BS, replace=False).tolist()
    print(f"\nNPRS_IDs: {nprs_ids}")

    # =========================================================================
    #  3. PRS-Grids generieren + OFDM-Modulation
    #     Analog: nrPRS() → slotGrid → prsGrid → nrOFDMModulate()
    # =========================================================================
    print("\n--- PRS-Generierung & OFDM-Modulation ---")

    total_slots = config.N_FRAMES * config.SLOTS_PER_FRAME  # 10 Slots
    prs_grids   = [np.zeros((14, config.NSC), dtype=np.complex64)
                   for _ in range(config.N_BS)]

    for slot_idx in range(total_slots):
        for gnb_idx in range(config.N_BS):
            # Slot-Offset prüfen (analog: PRSResourceOffset)
            slot_offset = config.PRS_SLOT_OFFSETS[gnb_idx]
            if slot_idx == slot_offset:
                prs_grid, indices, symbols = nrPRS(
                    Nsc         = config.NSC,
                    nprs_id     = nprs_ids[gnb_idx],
                    slot        = slot_idx,
                    comb_size   = config.PRS_COMB_SIZE,
                    num_symbols = config.PRS_NUM_SYMBOLS,
                    symbol_start = config.PRS_SYMBOL_START,
                    num_rbs     = config.PRS_NUM_RBS,
                    rb_offset   = config.PRS_RB_OFFSET,
                )
                prs_grids[gnb_idx] = prs_grid

    # OFDM-Modulation (analog: nrOFDMModulate)
    tx_waveforms = []
    for gnb_idx in range(config.N_BS):
        waveform = ofdm_modulate(prs_grids[gnb_idx], config.NFFT)
        tx_waveforms.append(waveform)
        print(f"  gNB{gnb_idx+1}: NPRSID={nprs_ids[gnb_idx]:4d} | "
              f"Slot-Offset={config.PRS_SLOT_OFFSETS[gnb_idx]} | "
              f"Waveform-Länge={len(waveform)}")

    # =========================================================================
    #  4. Kanal: Delay + Pathloss (analog: sampleDelay, rx / sqrt(PL))
    # =========================================================================
    print("\n--- Kanal: Delay + Pathloss + Rauschen ---")
    rx_waveform = source.get_rx_waveform(tx_waveforms)
    print(f"  RX-Waveform Länge: {len(rx_waveform)} Samples")

    # =========================================================================
    #  5. ToA-Schätzung via Kreuzkorrelation
    #     Analog: [~,mag] = nrTimingEstimate(carrier, rxWaveform, prsGrid)
    #             corr = mag(1:Nfft*SCS/15)
    #             delayEst = find(corr==max(corr),1) - 1
    # =========================================================================
    print("\n--- ToA-Schätzung (Kreuzkorrelation, analog nrTimingEstimate) ---")

    max_corr_len = int(config.NFFT * config.SCS_HZ / 15e3)
    corr_list    = []
    delay_est    = np.zeros(config.N_BS, dtype=int)
    max_corr_val = np.zeros(config.N_BS)

    for gnb_idx in range(config.N_BS):
        ref_waveform = ofdm_modulate(prs_grids[gnb_idx], config.NFFT)

        d_samples, d_s, corr = nrTimingEstimate(
            rx_waveform, ref_waveform, sample_rate, max_corr_len)

        corr_list.append(corr)
        delay_est[gnb_idx]    = d_samples
        max_corr_val[gnb_idx] = corr.max()

        err_m = abs(d_samples - sample_delays[gnb_idx]) / sample_rate * config.C
        print(f"  gNB{gnb_idx+1}: delayEst={d_samples:4d} Samples "
              f"| Wahr={sample_delays[gnb_idx]:4d} | "
              f"ToA-Fehler={err_m:.1f} m | maxCorr={corr.max():.3f}")

    # Plot Korrelation (analog: plotPRSCorr)
    plotPRSCorr(corr_list, sample_rate,
                [f"gNB{i+1} (NPRSID={nprs_ids[i]})" for i in range(config.N_BS)])

    # =========================================================================
    #  6. Beste Zellen wählen (analog: detectedgNBs)
    #     Matlab: [~,detectedgNBs] = sort(maxCorr,'descend'); detectedgNBs(1:k)
    # =========================================================================
    cells_to_detect = min(config.CELLS_TO_DETECT, config.N_BS)
    detected_gnbs   = np.argsort(max_corr_val)[::-1][:cells_to_detect].tolist()
    print(f"\nDetektierte gNBs (nach Korrelationsstärke): "
          f"{[f'gNB{i+1}' for i in detected_gnbs]}")
    print(f"Referenz-gNB: gNB{detected_gnbs[0]+1}")

    # =========================================================================
    #  7. RSTD berechnen (analog: rstdVals = getRSTDValues(delayEst, SampleRate))
    # =========================================================================
    rstd_matrix = getRSTDValues(delay_est, sample_rate)
    print(f"\nRSTD-Matrix [µs] (detektierte gNBs):")
    for i in detected_gnbs:
        row = [f"{rstd_matrix[i,j]*1e6:7.3f}" for j in detected_gnbs]
        print(f"  gNB{i+1}: [{', '.join(row)}]")

    # =========================================================================
    #  8. Hyperbeln + Positionsschätzung
    #     Analog: getRSTDCurve() → getEstimatedUEPosition()
    # =========================================================================
    print("\n--- Positionsschätzung (Hyperbel-Multilateration) ---")

    ref_idx      = detected_gnbs[0]
    neighbor_idx = detected_gnbs[1:]

    # Nur detektierte gNBs für Positionsschätzung nutzen
    est_pos, curve_x, curve_y, gnb_pairs = estimate_position_otdoa(
        gnb_positions = gnb_pos,
        rstd_matrix   = rstd_matrix,
        ref_gnb_idx   = ref_idx,
        neighbor_idxs = neighbor_idx,
    )

    # =========================================================================
    #  9. Ergebnisse (analog: disp(['Estimated UE Position...']))
    # =========================================================================
    est_err = np.linalg.norm(est_pos[:2] - ue_pos[:2])
    print(f"\n  Wahre UE-Position         : [{ue_pos[0]:.1f}, {ue_pos[1]:.1f}]")
    print(f"  Geschätzte UE-Position    : [{est_pos[0]:.1f}, {est_pos[1]:.1f}]")
    print(f"  UE Position Estimation Error: {est_err:.2f} meters")

    # Plot Hyperbeln (analog: plotPositionsAndHyperbolaCurves)
    plotPositionsAndHyperbolaCurves(
        gnb_pos, ue_pos, est_pos,
        curve_x, curve_y, gnb_pairs, detected_gnbs
    )

    plt.show()
    return est_pos, est_err


if __name__ == "__main__":
    run()
