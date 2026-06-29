# =============================================================================
#  dashboard.py – Interaktives Parameter-Dashboard
#
#  Zeigt in Echtzeit die Auswirkungen von Parameteränderungen auf:
#  1. PRS-Waveform und PDSCH-Waveform
#  2. Korrelationsbetrag (ToA-Schätzung)
#  3. Positionierungsfehler
#  4. Slot-Konfiguration
#
#  Steuerbare Parameter:
#  - Sendeleistung (PT_DBM)
#  - Rauschzahl (NOISE_FIGURE_DB)
#  - Subcarrier-Abstand (SCS)
#  - Bandbreite (NUM_RBS)
#  - CombSize
#  - PRS-Symbole
#
#  Starten: python3 dashboard.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
import warnings
warnings.filterwarnings('ignore')

# Projektmodule
import config as cfg
from signal_processing.prs_generator   import nrPRS
from signal_processing.pdsch_generator import nrPDSCH_grid as nrPDSCH
from signal_processing.ofdm            import ofdm_modulate
from signal_processing.pathloss        import nrPathLoss
from signal_processing.toa_estimation  import nrTimingEstimate, getRSTDValues
from ran.simulated                     import SimulatedSource
from gnb.transmitter                   import GnbConfig, GnbTransmitter
from ue.receiver                       import UeReceiver
from positioning.tdoa_ls               import estimate_position_otdoa


# =============================================================================
#  Simulationsparameter (live veränderbar)
# =============================================================================

class DashParams:
    """Aktuelle Dashboard-Parameter – werden durch Slider geändert."""
    def __init__(self):
        self.pt_dbm        = cfg.PT_DBM           # Sendeleistung [dBm]
        self.noise_fig_db  = cfg.NOISE_FIGURE_DB  # Rauschzahl [dB]
        self.scs_khz       = cfg.SCS_HZ / 1e3     # SCS [kHz]
        self.num_rbs       = cfg.NUM_RBS           # Bandbreite in RBs
        self.comb_size     = cfg.PRS_COMB_SIZE     # CombSize
        self.num_symbols   = cfg.PRS_NUM_SYMBOLS   # PRS-Symbole
        self.n_gnbs        = cfg.N_BS              # Anzahl gNBs

params = DashParams()


# =============================================================================
#  Simulation ausführen
# =============================================================================

def run_simulation(p: DashParams):
    """
    Führt eine vollständige Simulation mit den aktuellen Parametern durch.
    Gibt alle für die Visualisierung nötigen Daten zurück.
    """
    scs_hz      = p.scs_khz * 1e3
    nsc         = p.num_rbs * 12
    nfft        = int(2 ** np.ceil(np.log2(nsc * 1.5)))
    sample_rate = scs_hz * nfft
    pt_w        = 10 ** (0.1 * (p.pt_dbm - 30))
    noise_power = 1.380649e-23 * 290 * scs_hz * 10 ** (p.noise_fig_db / 10)
    n_bs        = int(p.n_gnbs)
    comb_size   = int(p.comb_size)
    num_symbols = int(p.num_symbols)

    # SNR berechnen (für Anzeige)
    snr_db = p.pt_dbm - 30 - 10*np.log10(noise_power) - 100  # grob

    # --- Topologie ---
    rng_seed = cfg.RNG_SEED
    rng      = np.random.default_rng(rng_seed)

    # gNB-Positionen (hexagonal)
    gnb_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.pi/6 + k*np.pi/3
        gnb_positions.append((cfg.ISD_M*np.cos(angle), cfg.ISD_M*np.sin(angle)))
    gnb_pos = np.array([[p[0], p[1], cfg.BS_HEIGHT_M]
                        for p in gnb_positions[:n_bs]])
    ue_pos  = cfg.UE_POS

    # --- PRS-IDs und Slot-Offsets ---
    nprs_ids     = rng.choice(4096, size=n_bs, replace=False).tolist()
    slot_offsets = list(range(0, 2*n_bs, 2))

    # --- PRS Grid für gNB 0 (für Waveform-Plot) ---
    prs_grid_0, _, _ = nrPRS(
        Nsc          = nsc,
        nprs_id      = nprs_ids[0],
        slot         = slot_offsets[0],
        comb_size    = comb_size,
        num_symbols  = num_symbols,
        symbol_start = cfg.PRS_SYMBOL_START,
        num_rbs      = p.num_rbs,
        rb_offset    = cfg.PRS_RB_OFFSET,
    )

    pdsch_grid_0, _, _ = nrPDSCH(slot=slot_offsets[0]+1, rng_seed=0)
    # Auf aktuelle NSC zuschneiden
    if pdsch_grid_0.shape[1] != nsc:
        tmp = np.zeros((14, nsc), dtype=np.complex64)
        n   = min(nsc, pdsch_grid_0.shape[1])
        tmp[:, :n] = pdsch_grid_0[:, :n]
        pdsch_grid_0 = tmp

    prs_wave   = ofdm_modulate(prs_grid_0,   nfft)
    pdsch_wave = ofdm_modulate(pdsch_grid_0, nfft)

    # --- Kanalmatrix (vereinfacht, ohne SimulatedSource für Geschwindigkeit) ---
    TDL_D = cfg.TDL_EXCESS_DELAYS_S
    TDL_P = 10**(cfg.TDL_POWERS_DB/10); TDL_P /= TDL_P.sum()
    f     = np.arange(nsc) * scs_hz

    H       = np.zeros((n_bs, nsc), dtype=np.complex64)
    t_delay = np.zeros(n_bs)

    for nbs in range(n_bs):
        dx  = ue_pos[0] - gnb_pos[nbs, 0]
        dy  = ue_pos[1] - gnb_pos[nbs, 1]
        dz  = ue_pos[2] - gnb_pos[nbs, 2]
        d3d = np.sqrt(dx**2 + dy**2 + dz**2)
        toa = d3d / cfg.C

        pl_db  = nrPathLoss(cfg.CARRIER_FREQUENCY_HZ, gnb_pos[nbs], ue_pos,
                            scenario=cfg.PATHLOSS_SCENARIO)
        pl_lin = 10**(pl_db/10)

        rng_link = np.random.default_rng(nbs * 10000 + rng_seed)
        amps     = (np.sqrt(TDL_P/2) *
                    (rng_link.standard_normal(len(TDL_P)) +
                     1j*rng_link.standard_normal(len(TDL_P))))

        H_link = np.zeros(nsc, dtype=np.complex64)
        for amp, tau in zip(amps, TDL_D + toa):
            H_link += amp * np.exp(-1j*2*np.pi*f*tau)
        H_link    /= np.sqrt(pl_lin)
        H[nbs]     = H_link
        t_delay[nbs] = toa

    # --- Empfangssignal aufbauen ---
    prs_grids_all = []
    tx_waves      = []
    rng_noise = np.random.default_rng(0)

    for nbs in range(n_bs):
        slot = slot_offsets[nbs]
        pg, _, _ = nrPRS(nsc, nprs_ids[nbs], slot, comb_size,
                         num_symbols, cfg.PRS_SYMBOL_START,
                         p.num_rbs, cfg.PRS_RB_OFFSET)

        # Auf aktuellen Nfft modulieren
        pg_full = np.zeros((14, nsc), dtype=np.complex64)
        pg_full[:, :min(pg.shape[1], nsc)] = pg[:, :min(pg.shape[1], nsc)]

        dg, _, _ = nrPDSCH(slot=slot+1, rng_seed=nbs*1000)
        dg_full = np.zeros((14, nsc), dtype=np.complex64)
        n = min(nsc, dg.shape[1])
        dg_full[:, :n] = dg[:, :n]

        combined = pg_full + dg_full
        w = ofdm_modulate(combined, nfft)
        prs_grids_all.append(pg_full)
        tx_waves.append(w)

    # Kanal anwenden
    max_delay_s  = int(np.round(t_delay.max() * sample_rate)) + nfft
    rx_total     = np.zeros(len(tx_waves[0]) + max_delay_s, dtype=np.complex64)

    for nbs in range(n_bs):
        d_samp = int(np.round(t_delay[nbs] * sample_rate))
        pl_db  = nrPathLoss(cfg.CARRIER_FREQUENCY_HZ, gnb_pos[nbs], ue_pos)
        pl_lin = 10**(pl_db/10)
        pad    = max_delay_s - d_samp
        rx     = np.concatenate([
            np.zeros(d_samp, dtype=np.complex64),
            tx_waves[nbs].astype(np.complex64),
            np.zeros(pad, dtype=np.complex64)
        ]) / np.sqrt(pl_lin) * np.sqrt(pt_w)
        rx_total += rx

    # AWGN
    rx_total += (np.sqrt(noise_power/2) *
                 (rng_noise.standard_normal(len(rx_total)) +
                  1j*rng_noise.standard_normal(len(rx_total)))).astype(np.complex64)

    # --- ToA-Schätzung ---
    max_corr_len = int(nfft * scs_hz / 15e3)
    delay_est    = np.zeros(n_bs, dtype=int)
    max_corr_val = np.zeros(n_bs)
    corr_gNB0    = None

    for nbs in range(n_bs):
        ref_wave = ofdm_modulate(prs_grids_all[nbs], nfft)
        d_s, _, corr = nrTimingEstimate(rx_total, ref_wave, sample_rate, max_corr_len)
        delay_est[nbs]    = d_s
        max_corr_val[nbs] = corr.max()
        if nbs == 0:
            corr_gNB0 = corr

    # --- Positionsschätzung ---
    k      = min(3, n_bs)
    d_idx  = np.argsort(max_corr_val)[::-1][:k]
    rstd   = getRSTDValues(delay_est, sample_rate)
    est_pos, _, _, _ = estimate_position_otdoa(gnb_pos, rstd,
                                                ref_gnb_idx=d_idx[0],
                                                neighbor_idxs=d_idx[1:].tolist())
    pos_err = np.linalg.norm(est_pos[:2] - ue_pos[:2])

    # Auflösung
    resolution_m = cfg.C / (p.num_rbs * 12 * scs_hz)

    return {
        'prs_wave':    np.real(prs_wave[:300]),
        'pdsch_wave':  np.real(pdsch_wave[:300]),
        'prs_abs':     np.abs(prs_wave[:300]),
        'pdsch_abs':   np.abs(pdsch_wave[:300]),
        'corr':        corr_gNB0,
        'delay_true':  t_delay,
        'delay_est':   delay_est,
        'sample_rate': sample_rate,
        'pos_err':     pos_err,
        'est_pos':     est_pos,
        'ue_pos':      ue_pos,
        'gnb_pos':     gnb_pos,
        'resolution_m': resolution_m,
        'snr_db':      p.pt_dbm - p.noise_fig_db,
        'prs_power':   float(np.mean(np.abs(prs_wave)**2)),
        'pdsch_power': float(np.mean(np.abs(pdsch_wave)**2)),
        'nsc':         nsc,
        'nfft':        nfft,
        'scs_hz':      scs_hz,
        'slot_offsets': slot_offsets,
        'comb_size':   comb_size,
        'num_symbols': num_symbols,
        'n_bs':        n_bs,
    }


# =============================================================================
#  Dashboard aufbauen
# =============================================================================

plt.rcParams.update({'font.size': 9})
fig = plt.figure(figsize=(16, 10))
fig.suptitle('5G PRS Positioning – Interaktives Parameter-Dashboard',
             fontsize=13, fontweight='bold', y=0.98)

# --- Layout ---
gs_main = gridspec.GridSpec(3, 3, figure=fig,
                             left=0.07, right=0.72,
                             top=0.93, bottom=0.08,
                             hspace=0.45, wspace=0.35)
gs_ctrl = gridspec.GridSpec(1, 1, figure=fig,
                             left=0.75, right=0.98,
                             top=0.93, bottom=0.08)

# Plot-Achsen
ax_prs   = fig.add_subplot(gs_main[0, :2])   # PRS Waveform
ax_pdsch = fig.add_subplot(gs_main[1, :2])   # PDSCH Waveform
ax_corr  = fig.add_subplot(gs_main[2, :2])   # Korrelation
ax_topo  = fig.add_subplot(gs_main[0:2, 2])  # Topologie
ax_info  = fig.add_subplot(gs_main[2, 2])    # Info-Panel

# Steuerbereich
ax_ctrl = fig.add_subplot(gs_ctrl[0])
ax_ctrl.set_visible(False)

# --- Slider ---
slider_configs = [
    ('PT [dBm]',    10, 50, params.pt_dbm,       1,   'pt_dbm'),
    ('NF [dB]',      0, 15, params.noise_fig_db,  0.5, 'noise_fig_db'),
    ('SCS [kHz]',   15, 120,[15,30,60,120],        1,   'scs_khz'),
    ('RBs',         25, 264,[25,52,106,162,264],   1,   'num_rbs'),
    ('CombSize',     2, 12, [2,4,6,12],             1,   'comb_size'),
    ('PRS Symbole',  1, 12, [1,2,4,6,12],           1,   'num_symbols'),
    ('Anz. gNBs',    3,  7, params.n_gnbs,          1,   'n_gnbs'),
]

sliders = {}
slider_axes = []
ctrl_x = 0.76
ctrl_y = 0.88
ctrl_h = 0.065

for i, (label, vmin, vmax, vinit, vstep, attr) in enumerate(slider_configs):
    y = ctrl_y - i * ctrl_h
    ax_s = fig.add_axes([ctrl_x, y, 0.20, 0.018])
    init = vinit if not isinstance(vinit, list) else vinit[0]
    sl   = Slider(ax_s, label, vmin, vmax, valinit=init, valstep=vstep,
                  color='#1D9E75')
    sl.label.set_fontsize(9)
    sliders[attr] = sl
    slider_axes.append(ax_s)

# Reset-Button
ax_reset = fig.add_axes([ctrl_x + 0.05, ctrl_y - len(slider_configs)*ctrl_h - 0.02,
                          0.08, 0.025])
btn_reset = Button(ax_reset, 'Reset', color='#f0ede6')

# --- Erste Simulation ---
result = run_simulation(params)


# =============================================================================
#  Plots zeichnen
# =============================================================================

def update_plots(res):
    N = 300
    t = np.arange(N) / res['sample_rate'] * 1e6  # µs

    # --- PRS Waveform ---
    ax_prs.cla()
    ax_prs.plot(t, res['prs_wave'],  color='#1D9E75', lw=0.8, label='gNB 1 PRS (Realteil)')
    ax_prs.plot(t, res['prs_abs'],   color='#1D9E75', lw=1.2, ls='--', alpha=0.5,
                label='|PRS| Hüllkurve')
    ax_prs.axhline(0, color='gray', lw=0.4, ls=':')
    ax_prs.set_title(f'PRS-Signal  |  Leistung: {res["prs_power"]:.4f}  |  '
                     f'NSC={res["nsc"]}  |  Comb={res["comb_size"]}  |  '
                     f'Symbole={res["num_symbols"]}',
                     fontsize=8)
    ax_prs.set_ylabel('Amplitude')
    ax_prs.legend(fontsize=7, loc='upper right')
    ax_prs.grid(True, alpha=0.3)

    # --- PDSCH Waveform ---
    ax_pdsch.cla()
    ax_pdsch.plot(t, res['pdsch_wave'], color='#D85A30', lw=0.8, label='gNB 1 PDSCH (Realteil)')
    ax_pdsch.plot(t, res['pdsch_abs'],  color='#D85A30', lw=1.2, ls='--', alpha=0.5,
                  label='|PDSCH| Hüllkurve')
    ax_pdsch.axhline(0, color='gray', lw=0.4, ls=':')
    ax_pdsch.set_title(f'PDSCH-Signal  |  Leistung: {res["pdsch_power"]:.3f}  |  '
                       f'Verhältnis PDSCH/PRS: {res["pdsch_power"]/res["prs_power"]:.1f}×',
                       fontsize=8)
    ax_pdsch.set_ylabel('Amplitude')
    ax_pdsch.set_xlabel('Zeit [µs]')
    ax_pdsch.legend(fontsize=7, loc='upper right')
    ax_pdsch.grid(True, alpha=0.3)

    # --- Korrelation ---
    ax_corr.cla()
    if res['corr'] is not None:
        corr     = res['corr']
        sr       = res['sample_rate']
        t_corr   = np.arange(len(corr)) / sr * 1e6
        ax_corr.plot(t_corr, corr, color='#534AB7', lw=1.0)

        # Wahrer Delay
        true_d = res['delay_true'][0] * 1e6
        est_d  = res['delay_est'][0]  / sr * 1e6
        ax_corr.axvline(true_d, color='#1D9E75', lw=1.5, ls='--',
                        label=f'Wahr: {true_d:.3f}µs')
        ax_corr.axvline(est_d,  color='#D85A30', lw=1.5, ls=':',
                        label=f'Est:  {est_d:.3f}µs')
        ax_corr.set_title(f'Kreuzkorrelation |R(τ)|  |  '
                          f'ToA-Fehler: {abs(est_d-true_d)*1e3:.1f}ns  ≈  '
                          f'{abs(est_d-true_d)*1e-6*3e8:.1f}m  |  '
                          f'Auflösung: {res["resolution_m"]:.1f}m',
                          fontsize=8)
        ax_corr.set_xlabel('Delay [µs]')
        ax_corr.set_ylabel('|R(τ)|')
        ax_corr.legend(fontsize=7)
        ax_corr.grid(True, alpha=0.3)

    # --- Topologie ---
    ax_topo.cla()
    gnb = res['gnb_pos']
    ue  = res['ue_pos']
    est = res['est_pos']

    colors = ['#1D9E75','#D85A30','#534AB7','#BA7517','#185FA5','#A32D2D','#3B6D11']
    for i, pos in enumerate(gnb):
        ax_topo.scatter(pos[0], pos[1], marker='^',
                        color=colors[i%len(colors)], s=80, zorder=3)
        ax_topo.text(pos[0]+15, pos[1]+15, f'gNB{i+1}',
                     fontsize=7, color=colors[i%len(colors)])

    ax_topo.scatter(ue[0], ue[1], marker='*', color='black', s=120, zorder=4,
                    label='Wahre UE-Pos.')
    ax_topo.scatter(est[0], est[1], marker='o', color='red', s=60, zorder=4,
                    label=f'Geschätzt\n({res["pos_err"]:.1f}m Fehler)')
    ax_topo.plot([ue[0], est[0]], [ue[1], est[1]],
                 color='red', lw=1.0, ls='--', alpha=0.7)

    ax_topo.set_title(f'Topologie  |  Fehler: {res["pos_err"]:.1f} m', fontsize=9)
    ax_topo.set_aspect('equal')
    ax_topo.legend(fontsize=7, loc='upper right')
    ax_topo.grid(True, alpha=0.2)

    # --- Info ---
    ax_info.cla()
    ax_info.set_xticks([]); ax_info.set_yticks([])
    info = [
        ('Frequenz',     f'{cfg.CARRIER_FREQUENCY_HZ/1e9:.1f} GHz'),
        ('SCS',          f'{res["scs_hz"]/1e3:.0f} kHz'),
        ('Bandbreite',   f'{res["nsc"]*res["scs_hz"]/1e6:.1f} MHz'),
        ('NFFT',         f'{res["nfft"]}'),
        ('Auflösung',    f'{res["resolution_m"]:.2f} m'),
        ('Abtastrate',   f'{res["sample_rate"]/1e6:.2f} MHz'),
        ('SNR (grob)',   f'{res["snr_db"]:.1f} dB'),
        ('Pos.-Fehler',  f'{res["pos_err"]:.1f} m'),
        ('gNBs',         str(res["n_bs"])),
        ('Slot-Offsets', str(res["slot_offsets"][:res["n_bs"]])),
    ]
    for row, (label, val) in enumerate(info):
        ax_info.text(0.02, 0.95 - row*0.095, label + ':',
                     transform=ax_info.transAxes, fontsize=8,
                     color='gray', va='top')
        ax_info.text(0.55, 0.95 - row*0.095, val,
                     transform=ax_info.transAxes, fontsize=8,
                     fontweight='bold', va='top')
    ax_info.set_title('Parameter', fontsize=9)
    ax_info.set_xlim(0,1); ax_info.set_ylim(0,1)

    fig.canvas.draw_idle()


update_plots(result)


# =============================================================================
#  Slider-Callbacks
# =============================================================================

def on_slider_change(val):
    """Wird aufgerufen wenn ein Slider bewegt wird."""
    params.pt_dbm       = sliders['pt_dbm'].val
    params.noise_fig_db = sliders['noise_fig_db'].val
    params.scs_khz      = sliders['scs_khz'].val
    params.num_rbs      = int(sliders['num_rbs'].val)
    params.comb_size    = int(sliders['comb_size'].val)
    params.num_symbols  = int(sliders['num_symbols'].val)
    params.n_gnbs       = int(sliders['n_gnbs'].val)

    try:
        res = run_simulation(params)
        update_plots(res)
    except Exception as e:
        print(f"Simulationsfehler: {e}")

for sl in sliders.values():
    sl.on_changed(on_slider_change)


def on_reset(event):
    params.pt_dbm       = cfg.PT_DBM
    params.noise_fig_db = cfg.NOISE_FIGURE_DB
    params.scs_khz      = cfg.SCS_HZ / 1e3
    params.num_rbs      = cfg.NUM_RBS
    params.comb_size    = cfg.PRS_COMB_SIZE
    params.num_symbols  = cfg.PRS_NUM_SYMBOLS
    params.n_gnbs       = cfg.N_BS
    for attr, sl in sliders.items():
        sl.set_val(getattr(params, attr))

btn_reset.on_clicked(on_reset)


# =============================================================================
#  Annotationen
# =============================================================================

fig.text(0.76, 0.03,
         '← Slider bewegen für Live-Update\n'
         '   Reset = Standardwerte aus config.py',
         fontsize=8, color='gray', va='bottom')

plt.show()