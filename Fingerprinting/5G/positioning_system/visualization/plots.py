# =============================================================================
#  visualization/plots.py
#  Analog zu: plotgNBAndUEPositions(), plotPRSCorr(), plotPositionsAndHyperbolaCurves()
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import config

_COLORS = ['#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F',
           '#0072BD','#FF69B4','#8B4513','#006400']


def plotgNBAndUEPositions(gnb_pos, ue_pos, title="gNB und UE Positionen"):
    """Analog zu plotgNBAndUEPositions() in Matlab."""
    fig, ax = plt.subplots(figsize=(8,8))
    for i, pos in enumerate(gnb_pos):
        ax.scatter(pos[0], pos[1], marker='^', color=_COLORS[i%len(_COLORS)],
                   s=180, zorder=4)
        ax.annotate(f"gNB{i+1}", (pos[0], pos[1]),
                    xytext=(8,8), textcoords='offset points', fontsize=9,
                    color=_COLORS[i%len(_COLORS)])
    ax.scatter(ue_pos[0], ue_pos[1], marker='o', color='black',
               s=120, zorder=5, label='UE')
    ax.set_xlabel("X Position (meters)"); ax.set_ylabel("Y Position (meters)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    if config.VIZ_SAVE_PLOTS:
        _save(fig, "gnb_ue_positions.png")
    return fig


def plotPRSCorr(corr_list, sample_rate, gnb_labels=None):
    """Analog zu plotPRSCorr() in Matlab."""
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(corr_list[0])) / sample_rate * 1e6  # µs
    for i, corr in enumerate(corr_list):
        lbl = gnb_labels[i] if gnb_labels else f"gNB{i+1}"
        ax.plot(t, np.abs(corr), color=_COLORS[i%len(_COLORS)],
                lw=1.5, label=lbl)
        peak = np.argmax(np.abs(corr))
        ax.scatter(t[peak], np.abs(corr[peak]),
                   color=_COLORS[i%len(_COLORS)], s=60, zorder=5)
    ax.set_xlabel("Zeit (µs)"); ax.set_ylabel("Korrelationsbetrag")
    ax.set_title("PRS-Kreuzkorrelation aller gNBs")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if config.VIZ_SAVE_PLOTS:
        _save(fig, "prs_correlation.png")
    return fig


def plotPositionsAndHyperbolaCurves(gnb_pos, ue_pos_true, est_pos,
                                     curve_x, curve_y, gnb_pairs,
                                     detected_gnbs):
    """Analog zu plotPositionsAndHyperbolaCurves() in Matlab."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # gNBs
    for i in detected_gnbs:
        ax.scatter(gnb_pos[i,0], gnb_pos[i,1], marker='^',
                   color=_COLORS[i%len(_COLORS)], s=180, zorder=4)
        lbl = f"gNB{i+1}" + (" (Referenz)" if i == detected_gnbs[0] else "")
        ax.annotate(lbl, (gnb_pos[i,0], gnb_pos[i,1]),
                    xytext=(8,8), textcoords='offset points', fontsize=9,
                    color=_COLORS[i%len(_COLORS)])

    # Hyperbeln (analog zu Matlab: '--','LineWidth',0.9,'Color','k')
    for idx, (x, y) in enumerate(zip(curve_x, curve_y)):
        ref, nb = gnb_pairs[idx]
        ax.plot(x, y, '--', color='black', lw=0.9, alpha=0.7,
                label=f"Hyperbel gNB{ref+1}–gNB{nb+1}")

    # Wahre UE-Position
    ax.scatter(ue_pos_true[0], ue_pos_true[1], marker='o',
               color='black', s=120, zorder=5, label='Wahre UE-Position',
               facecolors='white', edgecolors='black', linewidths=2)

    # Geschätzte Position (analog zu Matlab: '+','Color','#D95319')
    ax.scatter(est_pos[0], est_pos[1], marker='+',
               color='#D95319', s=200, zorder=6, linewidths=2.5,
               label=f"Geschätzte Position")

    err = np.linalg.norm(est_pos[:2] - ue_pos_true[:2])
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    ax.set_title(f"DL-OTDOA mit PRS – Positionierungsfehler: {err:.1f} m",
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if config.VIZ_SAVE_PLOTS:
        _save(fig, "otdoa_hyperbolas.png")
    return fig


def _save(fig, filename):
    os.makedirs(config.VIZ_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.VIZ_OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] {path}")
