# =============================================================================
#  positioning/tdoa_ls.py
#  Hyperbel-basierte OTDOA Positionsschätzung
#  Analog zu: getRSTDCurve() + getEstimatedUEPosition() aus Matlab-Beispiel
# =============================================================================

import numpy as np
from scipy.optimize import minimize
import config


def getRSTDCurve(gnb1: np.ndarray, gnb2: np.ndarray,
                 rstd_distance: float,
                 mu_range: np.ndarray = None) -> tuple:
    """
    Berechnet Hyperbel-Kurve für ein RSTD-Wertepaar.
    Direkter Python-Port der Matlab-Funktion getRSTDCurve().

    Matlab-Code:
        delta = gNB1 - gNB2;
        [phi,r] = cart2pol(delta(1),delta(2));
        rd = (r+rstd)/2;
        a = (r/2)-rd;  c = r/2;  b = sqrt(c^2-a^2);
        hk = (gNB1+gNB2)/2;
        x = a*cosh(mu)*cos(phi) - b*sinh(mu)*sin(phi) + hk(1)
        y = a*cosh(mu)*sin(phi) + b*sinh(mu)*cos(phi) + hk(2)

    Parameters
    ----------
    gnb1, gnb2    : np.ndarray shape (3,) – gNB-Positionen [m]
    rstd_distance : float – RSTD × c [m] (Entfernungsdifferenz)
    mu_range      : Hyperbel-Parameter (default: -2 bis 2, Schritt 0.001)

    Returns
    -------
    x, y : np.ndarray – Koordinaten der Hyperbel
    """
    if mu_range is None:
        mu_range = np.arange(-2, 2, 1e-3)

    delta = gnb1[:2] - gnb2[:2]
    phi   = np.arctan2(delta[1], delta[0])   # cart2pol
    r     = np.linalg.norm(delta)             # Abstand zwischen gNBs

    rd = (r + rstd_distance) / 2
    a  = r/2 - rd
    c  = r/2
    b2 = c**2 - a**2

    if b2 <= 0:
        return None, None    # Ungültige Geometrie

    b  = np.sqrt(b2)
    hk = (gnb1[:2] + gnb2[:2]) / 2   # Mittelpunkt

    x = a*np.cosh(mu_range)*np.cos(phi) - b*np.sinh(mu_range)*np.sin(phi) + hk[0]
    y = a*np.cosh(mu_range)*np.sin(phi) + b*np.sinh(mu_range)*np.cos(phi) + hk[1]

    # Nur reelle Werte zurückgeben (analog zu Matlab: isreal(x) && isreal(y))
    if not (np.all(np.isreal(x)) and np.all(np.isreal(y))):
        return None, None

    return x.real, y.real


def getEstimatedUEPosition(curve_x: list, curve_y: list) -> np.ndarray:
    """
    Schätzt UE-Position aus Schnittpunkten der Hyperbeln.
    Analog zu: estPos = getEstimatedUEPosition(curveX, curveY)

    Matlab-Ansatz:
    1. Finde nächste Punkte zwischen je zwei Hyperbeln
    2. Linearisiere um diese Punkte (Tangenten)
    3. Berechne Schnittpunkt der Tangenten
    4. Mittele alle Schnittpunkte

    Parameters
    ----------
    curve_x, curve_y : list of np.ndarray – Hyperbel-Koordinaten

    Returns
    -------
    est_pos : np.ndarray, shape (3,) – [x, y, 0]
    """
    num_curves = len(curve_x)
    intersections_x = []
    intersections_y = []

    for i in range(num_curves - 1):
        for j in range(i+1, num_curves):
            x1, y1 = curve_x[i], curve_y[i]
            x2, y2 = curve_x[j], curve_y[j]

            # Nächste Punkte zwischen den zwei Hyperbeln finden
            # (analog zu findMinDistanceElements)
            dist = np.sqrt(
                (x1[:, None] - x2[None, :])**2 +
                (y1[:, None] - y2[None, :])**2
            )
            min_dist = np.min(dist)
            row, col = np.unravel_index(np.argmin(dist), dist.shape)

            # Tangente an Hyperbel 1 um Punkt row
            r1 = min(row+1, len(x1)-1)
            x1a, y1a = x1[row], y1[row]
            x1b, y1b = x1[r1],  y1[r1]

            # Tangente an Hyperbel 2 um Punkt col
            c1 = min(col+1, len(x2)-1)
            x2a, y2a = x2[col], y2[col]
            x2b, y2b = x2[c1],  y2[c1]

            # Schnittpunkt der zwei Tangenten (Linearisierung)
            dx1 = x1b - x1a
            dx2 = x2b - x2a

            if abs(dx1) < 1e-10 or abs(dx2) < 1e-10:
                continue

            a1 = (y1b - y1a) / dx1
            b1 = y1a - a1 * x1a
            a2 = (y2b - y2a) / dx2
            b2 = y2a - a2 * x2a

            if abs(a1 - a2) < 1e-10:
                continue    # Parallel – kein Schnittpunkt

            xc = (b2 - b1) / (a1 - a2)
            yc = a1 * xc + b1

            intersections_x.append(xc)
            intersections_y.append(yc)

    if not intersections_x:
        # Fallback: Least-Squares wenn keine Schnittpunkte gefunden
        return np.array([0.0, 0.0, 0.0])

    # Mittelwert aller Schnittpunkte (analog zu Matlab: mean(xC), mean(yC))
    return np.array([np.mean(intersections_x),
                     np.mean(intersections_y),
                     0.0])


def estimate_position_otdoa(gnb_positions: np.ndarray,
                             rstd_matrix:   np.ndarray,
                             ref_gnb_idx:   int = 0,
                             neighbor_idxs: list = None) -> tuple:
    """
    Vollständige OTDOA-Positionsschätzung.

    Analog zum Matlab-Hauptcode:
        for jj = detectedgNBs(1)       % Referenz-gNB
            for ii = detectedgNBs(2:end)  % Nachbar-gNBs
                rstd = rstdVals(ii,jj) * speedOfLight
                [x,y] = getRSTDCurve(gNBPos{txi}, gNBPos{txj}, rstd)
                ...
        estimatedPos = getEstimatedUEPosition(curveX, curveY)

    Parameters
    ----------
    gnb_positions : np.ndarray, shape (n_bs, 3)
    rstd_matrix   : np.ndarray, shape (n_bs, n_bs) – RSTD in Sekunden
    ref_gnb_idx   : Index des Referenz-gNB (erster detektierter)
    neighbor_idxs : Indizes der Nachbar-gNBs

    Returns
    -------
    est_pos   : np.ndarray, shape (3,)
    curve_x   : list – Hyperbel x-Koordinaten (für Plot)
    curve_y   : list – Hyperbel y-Koordinaten
    gnb_pairs : list – (ref, neighbor) Paare
    """
    if neighbor_idxs is None:
        neighbor_idxs = [i for i in range(len(gnb_positions)) if i != ref_gnb_idx]

    curve_x   = []
    curve_y   = []
    gnb_pairs = []

    for nb_idx in neighbor_idxs:
        rstd_dist = rstd_matrix[nb_idx, ref_gnb_idx] * config.C

        x, y = getRSTDCurve(
            gnb_positions[nb_idx],
            gnb_positions[ref_gnb_idx],
            rstd_dist
        )

        if x is not None and y is not None:
            curve_x.append(x)
            curve_y.append(y)
            gnb_pairs.append((ref_gnb_idx, nb_idx))
        else:
            print(f"  [Warnung] Hyperbel für gNB{ref_gnb_idx}–gNB{nb_idx} "
                  f"ungültig (RSTD={rstd_dist:.1f}m) – übersprungen.")

    if len(curve_x) < 2:
        print(f"  [Fehler] Nur {len(curve_x)} Hyperbel(n) – "
              f"mind. 2 benötigt für 2D-Positionierung.")
        return np.zeros(3), curve_x, curve_y, gnb_pairs

    est_pos = getEstimatedUEPosition(curve_x, curve_y)
    return est_pos, curve_x, curve_y, gnb_pairs
