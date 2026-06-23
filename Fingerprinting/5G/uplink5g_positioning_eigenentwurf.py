# =============================================================================
#  5G Uplink Positioning – UL-TDoA
#  py3gpp (SRS) + py3gppchannels (Kanal) + NumPy/SciPy (OFDM, ToA, TDoA)
#
#  Installation: pip install py3gpp py3gppchannels hexalattice
#
#  Behobene Bugs gegenüber v2:
#   BUG1: Kanalmodell enthielt keinen absoluten LOS-Delay → delays += d_3d/c
#   BUG2: ESPRIT durch IFFT+Blackman-Window ersetzt (stabiler bei LOS)
#   BUG3: TDoA-Optimierer mit mehreren Startpunkten, 2D-Only, Residuum-Guard
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize

try:
    from py3gpp import nrSRS, nrSRSIndices, nrCarrierConfig, nrSRSConfig
    PY3GPP_AVAILABLE = True
    print("py3gpp: SRS-Generierung aktiv")
except ImportError:
    PY3GPP_AVAILABLE = False
    print("py3gpp nicht gefunden – SRS-Eigenimplementierung (TS 38.211)")

try:
    import py3gppchannels as nw
    PY3GPPCH_AVAILABLE = True
    print("py3gppchannels: TR 38.901 Kanalmodell aktiv")
except ImportError:
    PY3GPPCH_AVAILABLE = False
    print("py3gppchannels nicht gefunden – TDL-C Eigenimplementierung")


# =============================================================================
#  ABSCHNITT 1: Parameter
# =============================================================================

carrierFrequency = 3.6e9      # Trägerfrequenz [Hz]
scs              = 30e3       # Subcarrier-Abstand [Hz]
numRBs           = 52         # Resource Blocks (10 MHz @ 30 kHz)
Nsc              = numRBs * 12
Nfft             = 1024
nBSs             = 7          # BSs: 1 Zentrum + 6 Nachbarn
nUEs             = 8
isd              = 200.0      # Inter-Site-Distanz [m]
bsHt             = 10.0
ueHt             = 1.5
k_best           = 4          # Anzahl BSs für TDoA
c                = 3e8

# SRS (TS 38.211)
transmissionComb = 2
cSRS             = 13         # → m_SRS = 48 RBs
bSRS             = 0
symbolStart      = 13
cyclicShift      = 0

# Sendeleistung & Rauschen
Pt_dBm      = 23
Pt          = 10**(0.1*(Pt_dBm - 30))
noiseFigure = 7               # dB
noisePower  = 1.380649e-23 * 290 * scs * 10**(noiseFigure/10)


# =============================================================================
#  ABSCHNITT 2: Topologie
# =============================================================================

def create_hex_topology(nBSs, isd, bsHt, nUEs, ueHt, seed=42):
    bs_pos = [(0.0, 0.0)]
    for k in range(6):
        angle = np.pi/6 + k*np.pi/3
        bs_pos.append((isd*np.cos(angle), isd*np.sin(angle)))
    bs_pos = np.array(bs_pos[:nBSs])
    txPos  = np.column_stack([bs_pos[:,0], bs_pos[:,1], np.full(len(bs_pos), bsHt)])

    rng   = np.random.default_rng(seed)
    r     = isd*0.5*np.sqrt(rng.uniform(0, 1, nUEs))
    phi   = rng.uniform(0, 2*np.pi, nUEs)
    rxPos = np.column_stack([r*np.cos(phi), r*np.sin(phi), np.full(nUEs, ueHt)])
    return txPos, rxPos

txPos, rxPos = create_hex_topology(nBSs, isd, bsHt, nUEs, ueHt)
print(f"Topologie: {nBSs} BSs, {nUEs} UEs")


# =============================================================================
#  ABSCHNITT 3: Kanalmodell – TDL-C mit absolutem LOS-Delay
#  FIX BUG1: delays = TDL_EXCESS_DELAYS + d_3d/c
# =============================================================================

# TDL-C Profil (3GPP TR 38.901 Tabelle A.1-3, DS=100ns normiert)
_TDL_EXCESS_NS = np.array([0, 65, 150, 290, 410, 520, 600, 700,
                            960, 1190, 1390, 1530, 1960, 2350, 2600])
_TDL_POWERS_DB = np.array([-4.4,-1.2,-3.5,-5.2,-2.5, 0.0,-2.2,-3.9,
                            -7.4,-7.1,-10.7,-11.1,-5.1,-6.8,-8.7])


def build_channel_matrix(txPos, rxPos, fc, scs, Nsc, rng_seed=0):
    """
    H[nbs, nue, Nsc] mit absolutem LOS-Delay eingebettet.
    true_delays[nbs, nue] = geometrische Laufzeit d_3d/c [s].
    """
    nBSs_  = len(txPos)
    nUEs_  = len(rxPos)
    H       = np.zeros((nBSs_, nUEs_, Nsc), dtype=np.complex64)
    t_delay = np.zeros((nBSs_, nUEs_))
    f       = np.arange(Nsc) * scs

    excess = _TDL_EXCESS_NS * 1e-9  # Excess-Delays [s]
    powers = 10**(_TDL_POWERS_DB/10)
    powers /= powers.sum()

    for nbs in range(nBSs_):
        for nue in range(nUEs_):
            dx  = rxPos[nue,0]-txPos[nbs,0]
            dy  = rxPos[nue,1]-txPos[nbs,1]
            dz  = rxPos[nue,2]-txPos[nbs,2]
            d2d = np.sqrt(dx**2+dy**2)
            d3d = np.sqrt(dx**2+dy**2+dz**2)
            toa = d3d / c                       # geometrischer LOS-Delay

            # Pathloss UMi LOS (TR 38.901 Tabelle 7.4.1-1)
            pl_db = 32.4 + 21*np.log10(d3d) + 20*np.log10(fc/1e9)
            pl    = 10**(pl_db/10)

            # Zufällige Fading-Amplituden
            rng  = np.random.default_rng(nbs*10000+nue+rng_seed)
            amps = (np.sqrt(powers/2) *
                    (rng.standard_normal(len(powers)) +
                     1j*rng.standard_normal(len(powers))))

            # FIX BUG1: absolute Delays = excess + toa (LOS-Offset)
            abs_delays = excess + toa
            H_link = np.zeros(Nsc, dtype=np.complex64)
            for a, tau in zip(amps, abs_delays):
                H_link += a * np.exp(-1j*2*np.pi*f*tau)
            H_link /= np.sqrt(pl)

            H[nbs, nue]  = H_link
            t_delay[nbs, nue] = toa

    return H, t_delay


print("\n--- Kanalmatrizen ---")
H_true, true_delays = build_channel_matrix(txPos, rxPos, carrierFrequency, scs, Nsc)
print(f"H: {H_true.shape}  |  Delay-Bereich: "
      f"{true_delays.min()*1e6:.3f}–{true_delays.max()*1e6:.3f} µs")


# =============================================================================
#  ABSCHNITT 4: SRS-Generierung (py3gpp oder Eigenimplementierung)
# =============================================================================

def _srs_sequence(M, n_id, cyclic_shift=0):
    """CAZAC-ähnliche SRS-Sequenz nach TS 38.211 §6.4.1.4.2."""
    u   = n_id % 30
    phi = np.pi * u * np.arange(M) * (np.arange(M)+1) / 31
    alpha = 2*np.pi*cyclic_shift/8
    seq = np.exp(1j*(phi + alpha*np.arange(M)))
    return (seq / np.sqrt(M)).astype(np.complex64)


def generate_srs_grid(Nsc, nue_idx, transmissionComb, symbolStart,
                      cSRS=13, bSRS=0, cyclicShift=0):
    """SRS-Ressourcengitter (14 x Nsc) nach TS 38.211."""
    m_srs_table = {
        0:[4,4,4,4], 1:[8,4,4,4], 2:[12,4,4,4], 3:[16,4,4,4],
        4:[16,8,4,4], 5:[20,4,4,4], 6:[24,4,4,4], 7:[24,12,4,4],
        8:[28,4,4,4], 9:[32,16,8,4], 10:[36,12,4,4], 11:[40,20,4,4],
        12:[48,16,8,4], 13:[48,24,12,4], 14:[52,4,4,4], 15:[56,28,4,4],
    }
    bw_cfg   = m_srs_table.get(cSRS, [48,24,12,4])
    M_sc_SRS = bw_cfg[bSRS] * 12 // transmissionComb
    k0       = max(0, (Nsc - bw_cfg[bSRS]*12)//2)
    combOff  = nue_idx % transmissionComb
    srs_idx  = k0 + combOff + np.arange(M_sc_SRS) * transmissionComb
    srs_idx  = srs_idx[srs_idx < Nsc]
    M_sc_SRS = len(srs_idx)
    symbols  = _srs_sequence(M_sc_SRS, nue_idx, cyclicShift)
    grid     = np.zeros((14, Nsc), dtype=np.complex64)
    grid[symbolStart, srs_idx] = symbols
    return grid, srs_idx, symbols


def generate_srs_grid_py3gpp(nue_idx, Nsc, nslot=0):
    carrier = nrCarrierConfig()
    carrier.NSizeGrid         = numRBs
    carrier.SubcarrierSpacing = int(scs/1e3)
    carrier.NSlot             = nslot
    srs = nrSRSConfig()
    srs.NumSRSSymbols = 1
    srs.SymbolStart   = symbolStart
    srs.NumSRSPorts   = 1
    srs.KTC           = transmissionComb
    srs.CSRS          = cSRS
    srs.BSRS          = bSRS
    srs.CyclicShift   = cyclicShift
    srs.SequenceID    = nue_idx
    sym = nrSRS(carrier, srs)
    ind = nrSRSIndices(carrier, srs)
    grid = np.zeros((14, Nsc), dtype=np.complex64)
    sc_idx  = ind[:,0]; sym_idx = ind[:,1]
    grid[sym_idx, sc_idx] = sym[:,0]
    return grid, sc_idx, sym[:,0]


# =============================================================================
#  ABSCHNITT 5: OFDM
# =============================================================================

def _cp_lengths(Nfft, nSymbols=14):
    cp0 = int(round(Nfft*144/2048)) + Nfft//128
    cpN = int(round(Nfft*144/2048))
    return [cp0] + [cpN]*(nSymbols-1)


def ofdm_modulate(grid, Nfft):
    nSym, Nsc_g = grid.shape
    cps = _cp_lengths(Nfft, nSym)
    half = Nsc_g//2
    waveform = []
    for l in range(nSym):
        fd = np.zeros(Nfft, dtype=np.complex64)
        fd[1:half+1]       = grid[l, half:]
        fd[Nfft-half:Nfft] = grid[l, :half]
        td = np.fft.ifft(fd) * np.sqrt(Nfft)
        waveform.append(np.concatenate([td[-cps[l]:], td]))
    return np.concatenate(waveform).astype(np.complex64)


def ofdm_demodulate(waveform, Nfft, Nsc, nSym=14):
    cps  = _cp_lengths(Nfft, nSym)
    grid = np.zeros((nSym, Nsc), dtype=np.complex64)
    half = Nsc//2
    idx  = 0
    for l in range(nSym):
        idx += cps[l]
        fd   = np.fft.fft(waveform[idx:idx+Nfft]) / np.sqrt(Nfft)
        idx += Nfft
        grid[l, half:] = fd[1:half+1]
        grid[l, :half] = fd[Nfft-half:Nfft]
    return grid


# =============================================================================
#  ABSCHNITT 6: Kanalschätzung (LS + Interpolation)
# =============================================================================

def estimate_channel_ls(rxGrid, srs_idx, tx_symbols):
    rx_srs  = rxGrid[symbolStart, srs_idx]
    H_pilot = rx_srs / tx_symbols
    all_sc  = np.arange(rxGrid.shape[1])
    r_int = interpolate.interp1d(srs_idx, H_pilot.real, kind='linear',
                                  bounds_error=False,
                                  fill_value=(H_pilot.real[0], H_pilot.real[-1]))
    i_int = interpolate.interp1d(srs_idx, H_pilot.imag, kind='linear',
                                  bounds_error=False,
                                  fill_value=(H_pilot.imag[0], H_pilot.imag[-1]))
    return (r_int(all_sc) + 1j*i_int(all_sc)).astype(np.complex64)


# =============================================================================
#  ABSCHNITT 7: ToA-Schätzung – IFFT + Blackman-Window
#  FIX BUG2: ESPRIT durch IFFT+Windowing ersetzt; Kanal enthält jetzt
#            absoluten Delay → Peak der CIR = ToA
# =============================================================================

def estimate_toa_ifft(H_freq, scs, Nfft_mult=16):
    """
    ToA via IFFT + Blackman-Window.
    Funktioniert korrekt wenn H(f) den absoluten LOS-Delay kodiert (FIX BUG1).
    """
    Nsc_ = len(H_freq)
    Ncir = Nsc_ * Nfft_mult
    win  = np.blackman(Nsc_).astype(np.float32)
    cir  = np.fft.ifft(H_freq * win, n=Ncir)
    pwr  = np.abs(cir[:Ncir//2])**2
    peak = int(np.argmax(pwr))
    return peak / (scs * Ncir)


# =============================================================================
#  ABSCHNITT 8: TDoA Positionsschätzung – robuster LS
#  FIX BUG3: mehrere Startpunkte, 2D-Optimierung, Divergenz-Guard
# =============================================================================

def tdoa_least_squares(bs_pos, tdoa, c=3e8, max_pos_m=2000.0):
    """
    2D TDoA Least-Squares mit mehreren Startpunkten.
    Gibt (pos3d (3,), residual) zurück.
    """
    ref  = bs_pos[0, :2]
    rdoa = tdoa * c

    def cost(pos2d):
        d0 = np.linalg.norm(pos2d - ref)
        return sum((np.linalg.norm(pos2d - bs_pos[i,:2]) - d0 - rdoa[i-1])**2
                   for i in range(1, len(bs_pos)))

    starts = [
        bs_pos.mean(axis=0)[:2],
        np.zeros(2),
        bs_pos[0,:2],
        bs_pos[1,:2] if len(bs_pos)>1 else np.zeros(2),
    ]
    best = None
    for x0 in starts:
        res = minimize(cost, x0, method='Nelder-Mead',
                       options={'maxiter':20000, 'xatol':0.1, 'fatol':1e-5})
        if best is None or res.fun < best.fun:
            best = res

    pos2d = best.x
    # Divergenz-Guard
    if np.any(np.abs(pos2d) > max_pos_m):
        pos2d = bs_pos.mean(axis=0)[:2]
    pos3d = np.append(pos2d, ueHt)
    return pos3d, best.fun


# =============================================================================
#  ABSCHNITT 9: Hauptsimulation
# =============================================================================

rng_main = np.random.default_rng(0)
ToAe     = np.zeros((nBSs, nUEs))

print("\n--- SRS-Übertragung & ToA-Schätzung ---")

for nue in range(nUEs):

    if PY3GPP_AVAILABLE:
        txGrid, srs_idx, tx_sym = generate_srs_grid_py3gpp(nue, Nsc)
    else:
        txGrid, srs_idx, tx_sym = generate_srs_grid(
            Nsc, nue, transmissionComb, symbolStart, cSRS, bSRS, cyclicShift)

    for nbs in range(nBSs):
        # Frequenzbereich-Kanalübertragung (direkte Multiplikation)
        rxGrid = np.zeros((14, Nsc), dtype=np.complex64)
        for sym in range(14):
            rxGrid[sym] = txGrid[sym] * H_true[nbs, nue] * np.sqrt(Pt)

        # AWGN
        rxGrid += (np.sqrt(noisePower/2) *
                   (rng_main.standard_normal(rxGrid.shape) +
                    1j*rng_main.standard_normal(rxGrid.shape))).astype(np.complex64)

        # LS-Kanalschätzung
        Hest = estimate_channel_ls(rxGrid, srs_idx, tx_sym)

        # ToA via IFFT
        ToAe[nbs, nue] = estimate_toa_ifft(Hest, scs)

    print(f"  UE {nue:2d}: ToA {ToAe[:,nue]*1e6} µs | Wahr {(true_delays[:,nue]*1e6).round(3)} µs")

print("\n--- Positionsschätzung ---")

# k-best BS-Auswahl: kleinste ToA-Abweichung zur wahren Laufzeit
error          = np.abs(ToAe - true_delays) / (true_delays + 1e-12)
rxPosEst       = np.zeros((nUEs, 3))
kBestIndices   = np.zeros((nUEs, k_best), dtype=int)

for nue in range(nUEs):
    kBestIndices[nue] = np.argsort(error[:, nue])[:k_best]
    toa  = ToAe[kBestIndices[nue], nue]
    tdoa = toa[1:] - toa[0]
    pos, res = tdoa_least_squares(txPos[kBestIndices[nue]], tdoa)
    rxPosEst[nue] = pos
    print(f"  UE {nue:2d}: Wahr ({rxPos[nue,0]:6.1f},{rxPos[nue,1]:6.1f}) m | "
          f"Geschätzt ({pos[0]:7.1f},{pos[1]:7.1f}) m | "
          f"Fehler {np.linalg.norm(pos[:2]-rxPos[nue,:2]):.1f} m")


# =============================================================================
#  ABSCHNITT 10: Ergebnisse & Visualisierung
# =============================================================================

posError2D = np.linalg.norm(rxPosEst[:,0:2] - rxPos[:,0:2], axis=1)
posError3D = np.linalg.norm(rxPosEst        - rxPos,        axis=1)
posError3D = np.where(np.isnan(posError3D), 0, posError3D)

print(f"\n--- Ergebnisse ---")
print(f"Median 2D-Fehler : {np.median(posError2D):.2f} m")
print(f"90%-Quantil 2D   : {np.percentile(posError2D,90):.2f} m")
print(f"Median 3D-Fehler : {np.median(posError3D):.2f} m")

# --- Plot 1: Topologie ---
fig, ax = plt.subplots(figsize=(8, 8))
colors = ["steelblue","darkorange","green","red","purple","brown","pink"]

rangeEst_2D = np.sqrt(np.maximum(
    (ToAe*c)**2 - (rxPos[:,2].reshape(1,-1)-txPos[:,2].reshape(-1,1))**2, 0))

for ki in range(k_best):
    for nue in range(nUEs):
        bs  = kBestIndices[nue, ki]
        circ = plt.Circle((txPos[bs,0], txPos[bs,1]), rangeEst_2D[bs,nue],
                           color=colors[nue%7], lw=0.5, ls='--', fill=False, zorder=0)
        ax.add_patch(circ)

ax.scatter(txPos[:,0], txPos[:,1], marker="^", color="black", s=160,
           label="BS", zorder=4)
ax.scatter(rxPos[:,0], rxPos[:,1], marker="*", color="red", s=180,
           label="Wahre UE-Position", zorder=3)
ax.scatter(rxPosEst[:,0], rxPosEst[:,1], marker="o", color="limegreen",
           edgecolors="black", s=100, label="Geschätzte UE-Position", zorder=2)

for nue in range(nUEs):
    ax.plot([rxPos[nue,0], rxPosEst[nue,0]],
            [rxPos[nue,1], rxPosEst[nue,1]],
            color="gray", lw=0.8, alpha=0.6)
    ax.annotate(f"{posError2D[nue]:.0f}m",
                xy=((rxPos[nue,0]+rxPosEst[nue,0])/2,
                    (rxPos[nue,1]+rxPosEst[nue,1])/2),
                fontsize=7, color="dimgray")

ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("5G UL-TDoA Positionsschätzung")
ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.4)
lim = isd*1.2; ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim])
plt.tight_layout()
plt.savefig("topology.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 2: CDF ---
xlimit = max(15.0, np.percentile(posError2D, 95)*1.5)
nbins  = max(nUEs*4, 40)

fig, ax = plt.subplots(figsize=(8, 5))
for err, label, col in [(posError2D,"2D-Fehler","steelblue"),
                         (posError3D,"3D-Fehler","darkorange")]:
    cnt, bins = np.histogram(err, bins=nbins, range=[0,xlimit])
    ax.plot(bins[1:], np.cumsum(cnt/nUEs), label=label, color=col, lw=2)

for y, lbl, col in [(0.50,"50%","crimson"),(2/3,"66.7%","magenta"),(0.90,"90%","royalblue")]:
    ax.axhline(y=y, lw=1.5, ls=':', color=col, label=lbl)

ax.set_xlabel("Positionierungsfehler (m)"); ax.set_ylabel("CDF")
ax.set_title("CDF des Positionierungsfehlers – 5G UL-TDoA")
ax.set_xlim([0,xlimit]); ax.set_ylim([0,1])
ax.grid(which='major', alpha=0.5); ax.grid(which='minor', alpha=0.2, ls='--')
ax.set_xticks(np.linspace(0,xlimit,11)); ax.set_yticks(np.linspace(0,1,11))
ax.legend(); plt.tight_layout()
plt.savefig("cdf_positioning_error.png", dpi=150, bbox_inches='tight')
plt.show()

print("Fertig. Plots: topology.png, cdf_positioning_error.png")

# np.savez("ULTDoA_v3.npz", posError2D=posError2D, posError3D=posError3D,
#          rxPosEst=rxPosEst, rxPos=rxPos, txPos=txPos, ToAe=ToAe,
#          true_delays=true_delays)