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


# =============================================================================
#  PRS Slot-Konfiguration visualisieren
#  Drei Plots analog zum interaktiven Widget:
#  1. Slot-Übersicht (Frame)
#  2. Symbol-Ebene (Slot-Zoom)
#  3. RE-Ebene (Comb-Muster)
# =============================================================================

_STAGGER = {
    2:  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    4:  [0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3],
    6:  [0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5],
    12: [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11],
}

_GNB_COLORS = ['#1D9E75', '#D85A30', '#534AB7',
               '#BA7517', '#185FA5', '#A32D2D', '#3B6D11']


def plot_prs_slot_overview(
    n_bs:              int   = None,
    total_slots:       int   = None,
    prs_slot_offsets:  list  = None,
    prs_num_symbols:   int   = None,
    prs_symbol_start:  int   = None,
    scs_hz:            float = None,
):
    """
    Plot 1: Slot-Übersicht über einen kompletten Frame.
    Zeigt welche gNB in welchem Slot PRS sendet und welche Slots PDSCH tragen.
    Analog zu Matlab plotGrid() im Beispiel.
    """
    import config as cfg
    n_bs             = n_bs             or cfg.N_BS
    total_slots      = total_slots      or cfg.SLOTS_PER_FRAME * cfg.N_FRAMES
    prs_slot_offsets = prs_slot_offsets or cfg.PRS_SLOT_OFFSETS
    prs_num_symbols  = prs_num_symbols  or cfg.PRS_NUM_SYMBOLS
    prs_symbol_start = prs_symbol_start or cfg.PRS_SYMBOL_START
    scs_hz           = scs_hz           or cfg.SCS_HZ

    slot_ms = 1000 / (scs_hz / 1e3)   # Slot-Länge in ms

    fig, ax = plt.subplots(figsize=(12, 4))

    for gnb_idx in range(n_bs):
        for slot_idx in range(total_slots):
            color = _GNB_COLORS[gnb_idx % len(_GNB_COLORS)]

            if slot_idx == prs_slot_offsets[gnb_idx % len(prs_slot_offsets)]:
                # PRS-Slot dieser gNB
                ax.add_patch(plt.Rectangle(
                    (slot_idx, gnb_idx), 1, 1,
                    color=color, alpha=0.85, zorder=2
                ))
                ax.text(slot_idx + 0.5, gnb_idx + 0.5, 'PRS',
                        ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold', zorder=3)

            elif slot_idx not in prs_slot_offsets[:n_bs]:
                # PDSCH-Slot
                ax.add_patch(plt.Rectangle(
                    (slot_idx, gnb_idx), 1, 1,
                    color='gray', alpha=0.15, zorder=1
                ))

    # y-Achse: gNB-Labels
    ax.set_yticks(np.arange(n_bs) + 0.5)
    ax.set_yticklabels([f'gNB {i+1}  (Slot {prs_slot_offsets[i % len(prs_slot_offsets)]})'
                        for i in range(n_bs)], fontsize=9)

    # x-Achse: Slot-Nummern
    ax.set_xticks(np.arange(total_slots) + 0.5)
    ax.set_xticklabels([str(s) for s in range(total_slots)], fontsize=8)
    ax.set_xlabel(f"Slot-Index  (Slot-Länge = {slot_ms:.2f} ms)", fontsize=10)
    ax.set_title(
        f"PRS Slot-Konfiguration – 1 Frame ({total_slots * slot_ms:.0f} ms) | "
        f"SCS = {scs_hz/1e3:.0f} kHz | {n_bs} gNBs",
        fontsize=11, fontweight='bold'
    )

    ax.set_xlim(0, total_slots)
    ax.set_ylim(0, n_bs)
    ax.grid(True, which='major', axis='x', alpha=0.3, linewidth=0.5)

    # Legende
    handles = [plt.Rectangle((0,0), 1, 1, color=_GNB_COLORS[i], alpha=0.85)
               for i in range(n_bs)]
    handles.append(plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.15))
    labels  = [f'gNB {i+1} PRS' for i in range(n_bs)] + ['PDSCH']
    ax.legend(handles, labels, loc='upper right', fontsize=8,
              ncol=min(n_bs+1, 4), framealpha=0.9)

    plt.tight_layout()
    if config.VIZ_SAVE_PLOTS:
        _save(fig, "prs_slot_overview.png")
    return fig


def plot_prs_symbol_level(
    gnb_idx:           int   = 0,
    prs_slot_offsets:  list  = None,
    prs_num_symbols:   int   = None,
    prs_symbol_start:  int   = None,
):
    """
    Plot 2: Symbol-Ebene – Zoom auf einen PRS-Slot.
    Zeigt welche der 14 OFDM-Symbole PRS vs. PDSCH/DM-RS tragen.
    """
    import config as cfg
    prs_slot_offsets = prs_slot_offsets or cfg.PRS_SLOT_OFFSETS
    prs_num_symbols  = prs_num_symbols  or cfg.PRS_NUM_SYMBOLS
    prs_symbol_start = prs_symbol_start or cfg.PRS_SYMBOL_START

    n_sym    = 14
    slot_idx = prs_slot_offsets[gnb_idx % len(prs_slot_offsets)]
    color    = _GNB_COLORS[gnb_idx % len(_GNB_COLORS)]

    fig, ax = plt.subplots(figsize=(12, 2.5))

    dmrs_symbols = [2, 11]   # Standard PDSCH DM-RS Positionen

    for sym in range(n_sym):
        is_prs  = prs_symbol_start <= sym < prs_symbol_start + prs_num_symbols
        is_dmrs = sym in dmrs_symbols and not is_prs

        if is_prs:
            c, alpha, label = color, 0.85, 'PRS'
            txt_color = 'white'
        elif is_dmrs:
            c, alpha, label = '#BA7517', 0.7, 'DM-RS'
            txt_color = 'white'
        else:
            c, alpha, label = 'gray', 0.12, ''
            txt_color = 'gray'

        ax.add_patch(plt.Rectangle((sym, 0), 1, 1,
                                   color=c, alpha=alpha, zorder=2))
        if label:
            ax.text(sym + 0.5, 0.5, label,
                    ha='center', va='center',
                    fontsize=8, color=txt_color,
                    fontweight='bold', zorder=3)

        # CP-Markierung (erstes Symbol hat längeren CP)
        cp_label = 'CP+' if sym == 0 else 'CP'
        ax.text(sym + 0.5, 0.1, cp_label,
                ha='center', va='bottom',
                fontsize=6, color='gray', alpha=0.6, zorder=3)

    ax.set_xlim(0, n_sym)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(n_sym) + 0.5)
    ax.set_xticklabels([f'Symbol {s}' for s in range(n_sym)], fontsize=8)
    ax.set_yticks([])
    ax.set_title(
        f"Symbol-Ebene – gNB {gnb_idx+1} | Slot {slot_idx} | "
        f"PRS: Symbol {prs_symbol_start}–{prs_symbol_start+prs_num_symbols-1}",
        fontsize=11, fontweight='bold'
    )
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)

    # Legende
    handles = [
        plt.Rectangle((0,0), 1, 1, color=color,    alpha=0.85),
        plt.Rectangle((0,0), 1, 1, color='#BA7517', alpha=0.7),
        plt.Rectangle((0,0), 1, 1, color='gray',    alpha=0.12),
    ]
    ax.legend(handles, ['PRS', 'DM-RS', 'Leer / PDSCH'],
              loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if config.VIZ_SAVE_PLOTS:
        _save(fig, f"prs_symbol_level_gnb{gnb_idx+1}.png")
    return fig


def plot_prs_re_pattern(
    prs_comb_size:    int = None,
    prs_num_symbols:  int = None,
    prs_symbol_start: int = None,
    n_prbs:           int = 4,
):
    """
    Plot 3: RE-Ebene – Comb-Muster mit staggered Offset.
    Zeigt welche Subcarrier in welchem Symbol PRS tragen.
    Analog zu plotGrid(..., 'REFill') in Matlab.
    """
    import config as cfg
    prs_comb_size    = prs_comb_size    or cfg.PRS_COMB_SIZE
    prs_num_symbols  = prs_num_symbols  or cfg.PRS_NUM_SYMBOLS
    prs_symbol_start = prs_symbol_start or cfg.PRS_SYMBOL_START

    stagger  = _STAGGER.get(prs_comb_size, _STAGGER[12])
    n_re     = n_prbs * 12   # Subcarrier die gezeigt werden
    color    = _GNB_COLORS[0]

    # Grid aufbauen: (n_re, prs_num_symbols) – 1 = PRS, 0 = leer
    grid = np.zeros((n_re, prs_num_symbols))
    for sym_local in range(prs_num_symbols):
        kp = stagger[sym_local % len(stagger)]
        for m in range(n_re // prs_comb_size + 1):
            re = m * prs_comb_size + kp
            if re < n_re:
                grid[re, sym_local] = 1

    fig, ax = plt.subplots(figsize=(10, 6))

    # Hintergrund
    ax.imshow(np.zeros((n_re, prs_num_symbols)),
              aspect='auto', cmap='Greys', alpha=0.05,
              extent=[-0.5, prs_num_symbols-0.5, -0.5, n_re-0.5])

    # PRS-REs einfärben
    for sym_local in range(prs_num_symbols):
        kp = stagger[sym_local % len(stagger)]
        for m in range(n_re // prs_comb_size + 1):
            re = m * prs_comb_size + kp
            if re < n_re:
                ax.add_patch(plt.Rectangle(
                    (sym_local - 0.5, re - 0.5), 1, 1,
                    color=color, alpha=0.8, zorder=2
                ))
        # k'-Label unten
        ax.text(sym_local, -1.2, f"k'={kp}",
                ha='center', va='top', fontsize=8,
                color=color, fontweight='bold')

    # PRB-Trennlinien
    for prb in range(n_prbs + 1):
        ax.axhline(y=prb*12 - 0.5, color='gray', linewidth=0.8,
                   alpha=0.5, linestyle='--')
        if prb < n_prbs:
            ax.text(-0.7, prb*12 + 5.5, f'RB {prb}',
                    ha='right', va='center', fontsize=8, color='gray')

    ax.set_xlim(-0.5, prs_num_symbols - 0.5)
    ax.set_ylim(-0.5, n_re - 0.5)
    ax.set_xticks(range(prs_num_symbols))
    ax.set_xticklabels([f'l={prs_symbol_start+s}' for s in range(prs_num_symbols)],
                       fontsize=8)
    ax.set_yticks(range(0, n_re, prs_comb_size))
    ax.set_xlabel("OFDM-Symbol", fontsize=10)
    ax.set_ylabel("Subcarrier (RE-Index)", fontsize=10)
    ax.set_title(
        f"RE-Ebene – Comb-{prs_comb_size} Muster | "
        f"{n_prbs} PRBs | Staggered k'-Offset nach TS 38.211",
        fontsize=11, fontweight='bold'
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)

    handles = [
        plt.Rectangle((0,0), 1, 1, color=color, alpha=0.8),
        plt.Rectangle((0,0), 1, 1, color='white',
                       ec='gray', linewidth=0.5),
    ]
    ax.legend(handles, ['PRS RE', 'Leer'],
              loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if config.VIZ_SAVE_PLOTS:
        _save(fig, "prs_re_pattern.png")
    return fig


def plot_prs_all(n_bs=None, gnb_idx_symbol=0):
    """
    Alle drei PRS-Plots auf einmal.
    Bequeme Funktion für main.py.
    """
    fig1 = plot_prs_slot_overview(n_bs=n_bs)
    fig2 = plot_prs_symbol_level(gnb_idx=gnb_idx_symbol)
    fig3 = plot_prs_re_pattern()
    return fig1, fig2, fig3
