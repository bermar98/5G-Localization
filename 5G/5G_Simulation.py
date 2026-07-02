# 5g_simulation/prs_positioning.py
import numpy as np
import matplotlib.pyplot as plt
from config import Config as cfg


class PRSPositioning:
    """
    Python-Implementierung des MATLAB 'NR Positioning Using PRS' Beispiels.
    OTDOA Positionierung mit PRS Signalen.
    """

    def __init__(self):
        np.random.seed(cfg.SEED)
        self.sample_rate = cfg.SCS * cfg.NFFT
        self.gnb_pos     = None
        self.ue_pos      = None
        self.cell_ids    = None
        self.prs_ids     = None

    # ── 1. Positionen generieren (h38901Scenario) ───────────────────────────
    def generiere_positionen(self):
        """Hexagonales gNB Layout + zufällige UE Position."""

        if cfg.GNB_POSITIONEN_MANUELL is not None:
            self.gnb_pos = np.array(cfg.GNB_POSITIONEN_MANUELL, dtype=float)
        else:
            d = cfg.INTERSITE_DIST
            h = 25.0  # gNB Höhe
            # 7-Zellen Hexagonal Layout
            alle = np.array([
                [0,          0,         h],
                [d,          0,         h],
                [d/2,        d*np.sqrt(3)/2, h],
                [-d/2,       d*np.sqrt(3)/2, h],
                [-d,         0,         h],
                [-d/2,      -d*np.sqrt(3)/2, h],
                [d/2,       -d*np.sqrt(3)/2, h],
            ])
            self.gnb_pos = alle[:cfg.NUM_GNB]

        # UE zufällig im inneren Bereich
        r   = cfg.INTERSITE_DIST * 0.3
        phi = np.random.uniform(0, 2*np.pi)
        rad = np.random.uniform(0, r)
        self.ue_pos = np.array([
            rad * np.cos(phi),
            rad * np.sin(phi),
            1.5
        ])

        # Cell IDs und PRS IDs
        self.cell_ids = np.random.choice(1008, cfg.NUM_GNB, replace=False)
        self.prs_ids  = np.random.choice(4096, cfg.NUM_GNB, replace=False)

        print("=== gNB und UE Positionen ===")
        for i in range(cfg.NUM_GNB):
            print(f"  gNB{i+1} (NCellID={self.cell_ids[i]}): "
                  f"x={self.gnb_pos[i,0]:.1f}m, "
                  f"y={self.gnb_pos[i,1]:.1f}m")
        print(f"  UE: x={self.ue_pos[0]:.2f}m, y={self.ue_pos[1]:.2f}m")

    # ── 2. PRS Signal generieren (nrPRS) ────────────────────────────────────
    def generiere_prs_sequenz(self, cell_id, prs_id, slot_offset):
        """
        Gold-Sequenz basiertes PRS Signal.
        Entspricht nrPRS() in MATLAB.
        """
        n = cfg.NUM_RB * 12

        # Gold-Sequenz Initialisierung (3GPP TS 38.211)
        c_init = (2**22 * np.floor(slot_offset / 14) +
                  2**10 * (slot_offset % 14) +
                  2**9  * cfg.COMB_SIZE +
                  prs_id) % (2**31)
        np.random.seed(int(c_init) % (2**31 - 1))

        # QPSK Symbole
        bits = np.random.randint(0, 2, 2*n)
        prs  = (1 - 2*bits[:n] + 1j*(1 - 2*bits[n:])) / np.sqrt(2)
        return prs

    # ── 3. Resource Grid + OFDM Modulation ──────────────────────────────────
    def generiere_waveform(self, gnb_idx):
        """
        PRS auf Resource Grid mappen und OFDM modulieren.
        Entspricht nrOFDMModulate(carrier, prsGrid + dataGrid).
        """
        slots_per_frame = 10
        total_slots     = cfg.N_FRAMES * slots_per_frame
        slot_offset     = gnb_idx * 2  # verschiedene Slots pro gNB

        waveform_parts = []

        for slot in range(total_slots):
            grid = np.zeros(cfg.NFFT, dtype=complex)

            if slot == slot_offset:
                # PRS in diesen Slot
                prs_seq = self.generiere_prs_sequenz(
                    self.cell_ids[gnb_idx],
                    self.prs_ids[gnb_idx],
                    slot
                )
                # Comb Mapping: jeder zweite Subcarrier
                sc_indices = np.arange(0, cfg.NUM_RB*12, cfg.COMB_SIZE)
                n_sc = min(len(sc_indices), len(prs_seq))
                grid[sc_indices[:n_sc]] = prs_seq[:n_sc]

            # IFFT → Zeitbereich (OFDM Symbol)
            symbol = np.fft.ifft(grid, n=cfg.NFFT) * np.sqrt(cfg.NFFT)
            waveform_parts.append(symbol)

        return np.concatenate(waveform_parts)

    # ── 4. Kanal anwenden (sampleDelay) ─────────────────────────────────────
    def wende_kanal_an(self, waveform, gnb_idx):
        """
        Propagationsverzögerung berechnen und anwenden.
        Entspricht rangeangle() + dsp.Delay() in MATLAB.
        """
        # Distanz gNB → UE
        delta   = self.gnb_pos[gnb_idx] - self.ue_pos
        distanz = np.sqrt(np.sum(delta**2))

        # Delay in Sekunden und Samples
        delay_s    = distanz / cfg.C
        delay_samp = delay_s * self.sample_rate

        # Ganzzahlige Verzögerung anwenden
        int_delay = int(np.floor(delay_samp))
        rx = np.concatenate([
            np.zeros(int_delay, dtype=complex),
            waveform
        ])

        # Pfadverlust (vereinfacht, freies Feld)
        path_loss_db = 20 * np.log10(4 * np.pi * distanz * cfg.FC / cfg.C)
        path_loss    = 10 ** (-path_loss_db / 20)
        rx           = rx * path_loss

        return rx, delay_samp, distanz

    # ── 5. Rauschen hinzufügen ───────────────────────────────────────────────
    def addiere_rauschen(self, rx_combined):
        """
        Thermisches Rauschen hinzufügen.
        Entspricht N0-Berechnung in MATLAB.
        """
        nf_lin = 10 ** (cfg.NOISE_FIGURE_DB / 10)
        t_eq   = cfg.RX_ANT_TEMP + 290 * (nf_lin - 1)
        n0     = np.sqrt(cfg.K_BOLTZ * self.sample_rate * t_eq / 2)
        noise  = n0 * (np.random.randn(len(rx_combined)) +
                      1j * np.random.randn(len(rx_combined)))
        return rx_combined + noise

    # ── 6. TOA Schätzung (nrTimingEstimate) ─────────────────────────────────
    def schaetze_toa(self, rx_waveform, gnb_idx, slot_offset):
        """
        TOA via Kreuzkorrelation mit PRS Referenz.
        Entspricht nrTimingEstimate() in MATLAB.
        """
        # Referenz PRS generieren
        prs_ref  = self.generiere_prs_sequenz(
            self.cell_ids[gnb_idx],
            self.prs_ids[gnb_idx],
            slot_offset
        )

        # Auf Grid mappen
        grid = np.zeros(cfg.NFFT, dtype=complex)
        sc_indices = np.arange(0, cfg.NUM_RB*12, cfg.COMB_SIZE)
        n_sc = min(len(sc_indices), len(prs_ref))
        grid[sc_indices[:n_sc]] = prs_ref[:n_sc]
        prs_td = np.fft.ifft(grid, n=cfg.NFFT) * np.sqrt(cfg.NFFT)

        # Kreuzkorrelation
        len_korr = cfg.NFFT * int(cfg.SCS / 15e3)
        rx_chunk = rx_waveform[:len_korr] if len(rx_waveform) >= len_korr else rx_waveform

        korr     = np.correlate(rx_chunk, prs_td, mode='full')
        korr_abs = np.abs(korr)

        # Peak = TOA
        peak_idx = np.argmax(korr_abs)
        max_korr = korr_abs[peak_idx]

        return peak_idx, max_korr, korr_abs

    # ── 7. RSTD Werte (getRSTDValues) ────────────────────────────────────────
    def berechne_rstd(self, toa_samples):
        """
        RSTD[i,j] = TOA[i] - TOA[j].
        Entspricht getRSTDValues() in MATLAB.
        """
        n    = len(toa_samples)
        rstd = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                rstd[i, j] = (toa_samples[i] - toa_samples[j]) / self.sample_rate
        return rstd

    # ── 8. Hyperbel berechnen (getRSTDCurve) ────────────────────────────────
    def berechne_hyperbel(self, gnb1_pos, gnb2_pos, rstd_dist):
        """
        Hyperbel aus RSTD Wert berechnen.
        Entspricht getRSTDCurve() in MATLAB — exakt gleiche Logik.
        """
        delta       = gnb1_pos[:2] - gnb2_pos[:2]
        phi         = np.arctan2(delta[1], delta[0])
        r           = np.sqrt(np.sum(delta**2))
        rd          = (r + rstd_dist) / 2

        a  = r/2 - rd
        c  = r/2
        b2 = c**2 - a**2

        if b2 < 0:
            return None, None

        b  = np.sqrt(b2)
        hk = (gnb1_pos[:2] + gnb2_pos[:2]) / 2
        mu = np.arange(-2, 2, 1e-3)

        x = (a*np.cosh(mu)*np.cos(phi) - b*np.sinh(mu)*np.sin(phi)) + hk[0]
        y = (a*np.cosh(mu)*np.sin(phi) + b*np.sinh(mu)*np.cos(phi)) + hk[1]

        return x, y

    # ── 9. UE Position schätzen (getEstimatedUEPosition) ────────────────────
    def schaetze_ue_position(self, curve_x, curve_y):
        """
        Schnittpunkt der Hyperbeln finden.
        Entspricht getEstimatedUEPosition() + findMinDistanceElements() in MATLAB.
        """
        xc_list = []
        yc_list = []

        num_curves = len(curve_x)

        for idx1 in range(num_curves - 1):
            for idx2 in range(idx1+1, num_curves):
                x1, y1 = curve_x[idx1], curve_y[idx1]
                x2, y2 = curve_x[idx2], curve_y[idx2]

                # Minimale Distanz zwischen Kurven (findMinDistanceElements)
                dist = np.sqrt(
                    (x1[:, None] - x2[None, :])**2 +
                    (y1[:, None] - y2[None, :])**2
                )
                min_val = np.min(dist)
                rows, cols = np.where(dist <= min_val + 5)

                if len(rows) == 0:
                    continue

                # Linearisierung um Schnittpunkt (wie in MATLAB)
                for k in range(min(2, len(rows))):
                    r, c = rows[k], cols[k]

                    # Punkte auf Kurve 1
                    x1a, y1a = x1[r], y1[r]
                    r2 = r+1 if r < len(x1)-1 else r-1
                    x1b, y1b = x1[r2], y1[r2]

                    # Punkte auf Kurve 2
                    x2a, y2a = x2[c], y2[c]
                    c2 = c+1 if c < len(x2)-1 else c-1
                    x2b, y2b = x2[c2], y2[c2]

                    # Schnittpunkt der Tangenten
                    if abs(x1b - x1a) < 1e-10 or abs(x2b - x2a) < 1e-10:
                        continue

                    a1 = (y1b - y1a) / (x1b - x1a)
                    b1 = y1a - a1 * x1a
                    a2 = (y2b - y2a) / (x2b - x2a)
                    b2 = y2a - a2 * x2a

                    if abs(a1 - a2) < 1e-10:
                        continue

                    xc = (b2 - b1) / (a1 - a2)
                    yc = a1 * xc + b1
                    xc_list.append(xc)
                    yc_list.append(yc)

        if not xc_list:
            return None

        return np.array([np.mean(xc_list), np.mean(yc_list), 0.0])

    # ── 10. Visualisierung ───────────────────────────────────────────────────
    def visualisieren(self, curve_x, curve_y, est_pos, ref_idx, det_gnbs, gnb_nums):

        wahre_pos = self.ue_pos
        fehler    = np.sqrt(
            (wahre_pos[0] - est_pos[0])**2 +
            (wahre_pos[1] - est_pos[1])**2
        )

        colors = plt.cm.tab10(np.linspace(0, 1, cfg.NUM_GNB))
        fig, ax = plt.subplots(figsize=(10, 8))

        # gNBs
        for i in range(cfg.NUM_GNB):
            label = f'gNB{i+1}'
            if i == ref_idx:
                label += ' (Referenz)'
            ax.plot(self.gnb_pos[i, 0], self.gnb_pos[i, 1],
                   marker='^', color=colors[i], markersize=12,
                   label=label, linestyle='None')
            ax.annotate(f'gNB{i+1}', self.gnb_pos[i, :2],
                       textcoords='offset points', xytext=(5, 5), fontsize=8)

        # Hyperbeln
        for i, (x, y) in enumerate(zip(curve_x, curve_y)):
            j, ref = gnb_nums[i]
            ax.plot(x, y, '--', linewidth=1.2, alpha=0.8,
                   label=f'Hyperbel gNB{ref+1}-gNB{j+1}')

        # Wahre Position
        ax.scatter(wahre_pos[0], wahre_pos[1], s=120, color='green',
                  zorder=5, label=f'Wahre UE Position ({wahre_pos[0]:.1f}, {wahre_pos[1]:.1f})')

        # Geschätzte Position
        ax.plot(est_pos[0], est_pos[1], '+', markersize=14,
               color='#D95319', markeredgewidth=3, zorder=5,
               label=f'Geschätzte Position ({est_pos[0]:.1f}, {est_pos[1]:.1f})')

        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_title(
            f'NR Positionierung mit PRS (OTDOA)\n'
            f'Fehler: {fehler:.2f}m | gNBs: {cfg.NUM_GNB} | '
            f'Frequenz: {cfg.FC/1e9:.1f} GHz'
        )
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('prs_otdoa_ergebnis.png', dpi=150)
        plt.show()

        print(f'\nWahre Position:      x={wahre_pos[0]:.2f}m, y={wahre_pos[1]:.2f}m')
        print(f'Geschätzte Position: x={est_pos[0]:.2f}m, y={est_pos[1]:.2f}m')
        print(f'Positionsfehler:     {fehler:.2f}m')
        return fehler

    # ── Main ─────────────────────────────────────────────────────────────────
    def run(self):
        print('=== NR Positionierung mit PRS (OTDOA) ===\n')

        # 1. Positionen
        self.generiere_positionen()

        # 2. Waveforms generieren und Kanal anwenden
        print('\n=== Waveform + Kanal ===')
        rx_combined = None
        delay_est   = np.zeros(cfg.NUM_GNB)
        radius      = np.zeros(cfg.NUM_GNB)

        for i in range(cfg.NUM_GNB):
            wf = self.generiere_waveform(i)
            rx, delay_samp, dist = self.wende_kanal_an(wf, i)
            delay_est[i] = delay_samp
            radius[i]    = dist

            if rx_combined is None:
                rx_combined = rx.copy()
            else:
                # Längen angleichen
                l = max(len(rx_combined), len(rx))
                a = np.zeros(l, dtype=complex)
                b = np.zeros(l, dtype=complex)
                a[:len(rx_combined)] = rx_combined
                b[:len(rx)]          = rx
                rx_combined = a + b

            print(f'  gNB{i+1}: Distanz={dist:.1f}m, '
                  f'Delay={delay_samp:.1f} Samples')

        # 3. Rauschen
        rx_combined = self.addiere_rauschen(rx_combined)

        # 4. TOA schätzen
        print('\n=== TOA Schätzung (Kreuzkorrelation) ===')
        toa_geschaetzt = np.zeros(cfg.NUM_GNB)
        max_korr       = np.zeros(cfg.NUM_GNB)
        korr_alle      = []

        for i in range(cfg.NUM_GNB):
            slot_offset = i * 2
            peak, mkorr, korr = self.schaetze_toa(rx_combined, i, slot_offset)
            toa_geschaetzt[i] = peak
            max_korr[i]       = mkorr
            korr_alle.append(korr)
            print(f'  gNB{i+1}: TOA_geschätzt={peak} Samples, '
                  f'TOA_wahr={delay_est[i]:.1f} Samples, '
                  f'Korr={mkorr:.4f}')

        # 5. Beste 3 gNBs auswählen
        cells_to_detect = min(3, cfg.NUM_GNB)
        det_gnbs        = np.argsort(max_korr)[::-1][:cells_to_detect]
        ref_idx         = det_gnbs[0]
        print(f'\nDetektierte gNBs: {[f"gNB{i+1}" for i in det_gnbs]}')
        print(f'Referenz gNB:     gNB{ref_idx+1}')

        # 6. RSTD berechnen
        print('\n=== RSTD Werte ===')
        rstd_vals = self.berechne_rstd(toa_geschaetzt)
        for j in det_gnbs[1:]:
            rstd_dist = rstd_vals[j, ref_idx] * cfg.C
            print(f'  RSTD gNB{j+1}-gNB{ref_idx+1}: '
                  f'{rstd_vals[j,ref_idx]*1e6:.3f}µs = {rstd_dist:.1f}m')

        # 7. Hyperbeln berechnen
        print('\n=== Hyperbeln ===')
        curve_x  = []
        curve_y  = []
        gnb_nums = []

        for j in det_gnbs[1:]:
            rstd_dist = rstd_vals[j, ref_idx] * cfg.C
            x, y = self.berechne_hyperbel(
                self.gnb_pos[j],
                self.gnb_pos[ref_idx],
                rstd_dist
            )
            if x is not None and np.all(np.isreal(x)):
                curve_x.append(x)
                curve_y.append(y)
                gnb_nums.append((j, ref_idx))
                print(f'  Hyperbel gNB{j+1}-gNB{ref_idx+1}: OK')
            else:
                print(f'  Hyperbel gNB{j+1}-gNB{ref_idx+1}: Fehler (imaginär) '
                      f'→ gNB Paar wird ignoriert')

        if len(curve_x) < 2:
            print('\nZu wenig Hyperbeln — erhöhe NUM_GNB in config.py '
                  'oder ändere SEED!')
            return None

        # 8. Position schätzen
        print('\n=== UE Position schätzen ===')
        est_pos = self.schaetze_ue_position(curve_x, curve_y)

        if est_pos is None:
            print('Positionsschätzung fehlgeschlagen!')
            return None

        # 9. Visualisieren
        fehler = self.visualisieren(curve_x, curve_y, est_pos, ref_idx,
                                   det_gnbs, gnb_nums)

        print(f'\nEstimated UE Position       : [{est_pos[0]:.4f} {est_pos[1]:.4f}]')
        print(f'UE Position Estimation Error: {fehler:.4f} meters')
        return fehler


if __name__ == '__main__':
    sim = PRSPositioning()
    sim.run()