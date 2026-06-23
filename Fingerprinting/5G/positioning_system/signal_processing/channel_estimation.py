# =============================================================================
#  signal_processing/channel_estimation.py
#  LS-Kanalschätzung für PRS (mehrere Symbole, staggered)
# =============================================================================

import numpy as np
from scipy import interpolate
import config


def estimate_channel_ls_prs(
    rx_grid:    np.ndarray,
    prs_indices: dict,
    tx_symbols:  dict,
) -> np.ndarray:
    """
    LS-Kanalschätzung aus PRS-Piloten über alle PRS-Symbole.

    Bei PRS mit mehreren Symbolen wird die Schätzung über alle
    Symbole gemittelt → höheres effektives SNR.

    Für jedes Symbol l:
        H_pilot[k] = Y[l,k] / X[l,k]   (LS auf Pilot-Subcarriern)

    Dann Interpolation auf alle Nsc Subcarrier und Mittelung.

    Parameters
    ----------
    rx_grid     : np.ndarray, shape (14, Nsc)
    prs_indices : dict {symbol_idx: subcarrier_indices}
    tx_symbols  : dict {symbol_idx: gesendete PRS-Symbole}

    Returns
    -------
    H_est : np.ndarray, shape (Nsc,), dtype complex64
    """
    Nsc      = rx_grid.shape[1]
    all_sc   = np.arange(Nsc)
    H_accum  = np.zeros(Nsc, dtype=np.complex128)
    n_sym    = len(prs_indices)

    for sym_idx, k_idx in prs_indices.items():
        rx_sym  = rx_grid[sym_idx, k_idx]
        tx_sym  = tx_symbols[sym_idx]

        # LS-Schätzung auf Pilot-Positionen
        H_pilot = rx_sym / tx_sym

        # Lineare Interpolation auf alle Subcarrier
        r_int = interpolate.interp1d(
            k_idx, H_pilot.real, kind='linear',
            bounds_error=False,
            fill_value=(H_pilot.real[0], H_pilot.real[-1])
        )
        i_int = interpolate.interp1d(
            k_idx, H_pilot.imag, kind='linear',
            bounds_error=False,
            fill_value=(H_pilot.imag[0], H_pilot.imag[-1])
        )
        H_accum += r_int(all_sc) + 1j * i_int(all_sc)

    return (H_accum / n_sym).astype(np.complex64)
