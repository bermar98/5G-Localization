# =============================================================================
#  evaluate_grid.py – Systematische Positionsauswertung im Raster
#
#  Bestimmt die Positionierungsgenauigkeit für UE-Positionen im Raster
#  von [-30, -30] bis [30, 30] in 10m-Schritten.
#
#  Ergebnis wird in output/grid_results.json gespeichert.
#
#  Starten: python3 evaluate_grid.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config
from gnb.transmitter  import GnbConfig, GnbTransmitter
from ue.receiver      import UeReceiver
from ran.simulated    import SimulatedSource
from signal_processing.pathloss import nrPathLoss


# =============================================================================
#  Raster-Parameter
# =============================================================================

GRID_MIN  = -30    # [m]
GRID_MAX  =  30    # [m]
GRID_STEP =  10    # [m]

OUTPUT_FILE = os.path.join(config.VIZ_OUTPUT_DIR, "grid_results.json")


# =============================================================================
#  Hilfsfunktion: Einzelne Position auswerten
# =============================================================================

def evaluate_position(ue_xy: np.ndarray, gnb_configs: list,
                      gnb_pos: np.ndarray) -> dict:
    """
    Führt die komplette DL-OTDOA Lokalisierung für eine UE-Position durch.

    Parameters
    ----------
    ue_xy       : np.ndarray (2,) – UE-Position [x, y] in Metern
    gnb_configs : list of GnbConfig
    gnb_pos     : np.ndarray (n_bs, 3)

    Returns
    -------
    dict mit wahre Position, geschätzter Position, Fehler und Metadaten
    """
    ue_pos = np.array([ue_xy[0], ue_xy[1], config.UE_HEIGHT_M])

    # Datenquelle mit dieser UE-Position
    source = SimulatedSource(ue_pos=ue_pos)

    true_delays   = source.get_true_delays()[:, 0]  # (n_bs,)
    sample_rate   = source.sample_rate
    sample_delays = source.sample_delays

    # gNB sendet
    transmitter  = GnbTransmitter(gnb_configs)
    tx_waveforms = transmitter.generate()

    # Kanal
    rx_waveform = source.get_rx_waveform(tx_waveforms)

    # UE empfängt + Server berechnet
    receiver = UeReceiver(
        gnb_positions   = gnb_pos,
        cells_to_detect = config.CELLS_TO_DETECT,
    )

    try:
        est_pos = receiver.process(
            rx_waveform   = rx_waveform,
            prs_grids     = transmitter.prs_grids,
            sample_rate   = sample_rate,
            sample_delays = sample_delays,  # (n_bs,)
        )
        error_2d = float(np.linalg.norm(est_pos[:2] - ue_pos[:2]))
        success  = True
    except Exception as e:
        est_pos  = np.array([float('nan'), float('nan'), float('nan')])
        error_2d = float('nan')
        success  = False

    # Pathloss pro gNB
    pathloss = [
        round(nrPathLoss(config.CARRIER_FREQUENCY_HZ,
                         gnb_pos[i], ue_pos,
                         scenario=config.PATHLOSS_SCENARIO), 1)
        for i in range(len(gnb_pos))
    ]

    return {
        "ue_true":      [round(float(ue_pos[0]), 2),
                         round(float(ue_pos[1]), 2),
                         round(float(ue_pos[2]), 2)],
        "ue_estimated": [round(float(est_pos[0]), 2),
                         round(float(est_pos[1]), 2),
                         round(float(est_pos[2]), 2)],
        "error_2d_m":   round(error_2d, 3),
        "success":      success,
        "toa_true_us":  [round(float(t)*1e6, 4) for t in true_delays],
        "pathloss_db":  pathloss,
        "detected_gnbs": [int(i)+1 for i in (receiver.detected_gnbs or [])],
    }


# =============================================================================
#  Hauptprogramm
# =============================================================================

def run_grid_evaluation():
    os.makedirs(config.VIZ_OUTPUT_DIR, exist_ok=True)

    # Raster aufbauen
    grid_values = np.arange(GRID_MIN, GRID_MAX + GRID_STEP, GRID_STEP)
    n_points    = len(grid_values)
    total       = n_points * n_points

    print("\n" + "="*60)
    print("  DL-OTDOA Raster-Auswertung")
    print(f"  Raster: {GRID_MIN} bis {GRID_MAX} m, Schritte: {GRID_STEP} m")
    print(f"  Punkte: {n_points} × {n_points} = {total}")
    print(f"  fc={config.CARRIER_FREQUENCY_HZ/1e9:.1f} GHz | "
          f"SCS={config.SCS_HZ/1e3:.0f} kHz | "
          f"{config.N_BS} gNBs")
    print("="*60 + "\n")

    # gNB-Konfiguration einmalig erstellen
    # (bleibt für alle UE-Positionen identisch)
    source_init = SimulatedSource(ue_pos=np.array([0.0, 0.0, config.UE_HEIGHT_M]))
    gnb_pos     = source_init.get_bs_positions()

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

    print("gNB-Positionen:")
    for i, pos in enumerate(gnb_pos):
        print(f"  gNB{i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m "
              f"| NPRS_ID={nprs_ids[i]}")
    print()

    # Auswertung
    results   = []
    errors    = []
    t_start   = time.time()
    count     = 0

    for ix, x in enumerate(grid_values):
        for iy, y in enumerate(grid_values):
            count += 1
            ue_xy = np.array([float(x), float(y)])

            t0  = time.time()
            res = evaluate_position(ue_xy, gnb_configs, gnb_pos)
            dt  = time.time() - t0

            results.append(res)
            if res["success"] and not np.isnan(res["error_2d_m"]):
                errors.append(res["error_2d_m"])

            status = f"✓ {res['error_2d_m']:.1f}m" if res["success"] else "✗ Fehler"
            print(f"  [{count:3d}/{total}] "
                  f"UE=({x:+5.1f},{y:+5.1f}) → "
                  f"Est=({res['ue_estimated'][0]:+7.1f},{res['ue_estimated'][1]:+7.1f}) | "
                  f"{status} | {dt:.1f}s")

    t_total = time.time() - t_start
    errors  = np.array(errors)

    # Statistik
    stats = {
        "n_total":        total,
        "n_success":      int(len(errors)),
        "n_failed":       int(total - len(errors)),
        "median_m":       round(float(np.median(errors)), 3),
        "mean_m":         round(float(np.mean(errors)), 3),
        "p25_m":          round(float(np.percentile(errors, 25)), 3),
        "p75_m":          round(float(np.percentile(errors, 75)), 3),
        "p90_m":          round(float(np.percentile(errors, 90)), 3),
        "rmse_m":         round(float(np.sqrt(np.mean(errors**2))), 3),
        "min_m":          round(float(np.min(errors)), 3),
        "max_m":          round(float(np.max(errors)), 3),
        "runtime_s":      round(t_total, 1),
    }

    # JSON speichern
    output = {
        "metadata": {
            "grid_min_m":             GRID_MIN,
            "grid_max_m":             GRID_MAX,
            "grid_step_m":            GRID_STEP,
            "carrier_frequency_hz":   config.CARRIER_FREQUENCY_HZ,
            "scs_hz":                 config.SCS_HZ,
            "num_rbs":                config.NUM_RBS,
            "bandwidth_mhz":          round(config.NSC * config.SCS_HZ / 1e6, 2),
            "n_bs":                   config.N_BS,
            "prs_comb_size":          config.PRS_COMB_SIZE,
            "prs_num_symbols":        config.PRS_NUM_SYMBOLS,
            "pt_dbm":                 config.PT_DBM,
            "noise_figure_db":        config.NOISE_FIGURE_DB,
            "pathloss_scenario":      config.PATHLOSS_SCENARIO,
            "isd_m":                  config.ISD_M,
            "gnb_positions":          gnb_pos.tolist(),
            "nprs_ids":               nprs_ids,
        },
        "statistics": stats,
        "results":    results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Zusammenfassung
    print(f"\n{'='*60}")
    print(f"  Ergebnisse gespeichert: {OUTPUT_FILE}")
    print(f"  Laufzeit: {t_total:.1f}s ({t_total/total:.1f}s pro Punkt)")
    print(f"  Erfolgreiche Schätzungen: {len(errors)}/{total}")
    print(f"  Median-Fehler : {stats['median_m']:.2f} m")
    print(f"  90%-Fehler    : {stats['p90_m']:.2f} m")
    print(f"  RMSE          : {stats['rmse_m']:.2f} m")
    print(f"{'='*60}\n")

    # Fehler-Heatmap
    _plot_heatmap(results, grid_values, stats)

    return output


# =============================================================================
#  Heatmap-Visualisierung
# =============================================================================

def _plot_heatmap(results: list, grid_values: np.ndarray, stats: dict):
    """Erstellt eine Fehler-Heatmap über das Auswertungsraster."""
    n   = len(grid_values)
    err_grid = np.full((n, n), np.nan)

    for res in results:
        x, y = res["ue_true"][0], res["ue_true"][1]
        ix   = int(np.where(grid_values == x)[0][0])
        iy   = int(np.where(grid_values == y)[0][0])
        if res["success"]:
            err_grid[iy, ix] = res["error_2d_m"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"DL-OTDOA Positionierungsfehler – Rasterauswertung\n"
        f"fc={config.CARRIER_FREQUENCY_HZ/1e9:.1f} GHz | "
        f"BW={config.NSC*config.SCS_HZ/1e6:.1f} MHz | "
        f"{config.N_BS} gNBs | ISD={config.ISD_M:.0f} m",
        fontsize=11, fontweight='bold'
    )

    # --- Plot 1: Heatmap ---
    ax = axes[0]
    vmax = np.nanpercentile(err_grid, 90)
    im = ax.imshow(
        err_grid,
        origin='lower',
        extent=[GRID_MIN - GRID_STEP/2, GRID_MAX + GRID_STEP/2,
                GRID_MIN - GRID_STEP/2, GRID_MAX + GRID_STEP/2],
        cmap='RdYlGn_r',
        vmin=0, vmax=vmax,
        aspect='equal'
    )
    plt.colorbar(im, ax=ax, label='2D-Fehler [m]')

    # gNB-Positionen einzeichnen
    source_tmp = SimulatedSource(ue_pos=np.array([0.0, 0.0, config.UE_HEIGHT_M]))
    gnb_pos    = source_tmp.get_bs_positions()
    ax.scatter(gnb_pos[:, 0], gnb_pos[:, 1],
               marker='^', color='black', s=120, zorder=5, label='gNBs')

    # Fehlerwerte beschriften
    for res in results:
        if res["success"]:
            ax.text(res["ue_true"][0], res["ue_true"][1],
                    f'{res["error_2d_m"]:.1f}',
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold')

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Fehler-Heatmap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # --- Plot 2: CDF ---
    ax2 = axes[1]
    errors_flat = err_grid.flatten()
    errors_flat = errors_flat[~np.isnan(errors_flat)]
    errors_sorted = np.sort(errors_flat)
    cdf = np.arange(1, len(errors_sorted)+1) / len(errors_sorted)

    ax2.plot(errors_sorted, cdf, color='#1D9E75', lw=2)
    ax2.axhline(0.50, color='crimson',   lw=1.2, ls=':', label=f"Median: {stats['median_m']:.1f} m")
    ax2.axhline(0.90, color='royalblue', lw=1.2, ls=':', label=f"P90: {stats['p90_m']:.1f} m")
    ax2.axvline(stats['median_m'], color='crimson',   lw=1.2, ls='--', alpha=0.7)
    ax2.axvline(stats['p90_m'],    color='royalblue', lw=1.2, ls='--', alpha=0.7)

    ax2.set_xlabel("2D-Positionierungsfehler [m]")
    ax2.set_ylabel("Kumulative Häufigkeit")
    ax2.set_title("CDF des Positionierungsfehlers")
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()

    plot_path = os.path.join(config.VIZ_OUTPUT_DIR, "grid_heatmap.png")
    if config.VIZ_SAVE_PLOTS:
        os.makedirs(config.VIZ_OUTPUT_DIR, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] {plot_path}")
    plt.show()


# =============================================================================
#  Einstiegspunkt
# =============================================================================

if __name__ == "__main__":
    import io, contextlib

    # interne Prints der Simulation unterdrücken
    _orig_evaluate = evaluate_position
    def evaluate_position(ue_xy, gnb_configs, gnb_pos):
        with contextlib.redirect_stdout(io.StringIO()):
            return _orig_evaluate(ue_xy, gnb_configs, gnb_pos)

    run_grid_evaluation()