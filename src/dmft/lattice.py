"""Bethe lattice local Green's function and h-ghost sector correlators.

H_lat is quadratic: lattice d-levels + h-ghost levels.
The Bethe lattice self-consistency gives an analytic local GF:

    G_loc(iw) = (zeta - sqrt(zeta^2 - D^2)) / (D^2 / 2)

where zeta = iw + mu - eps_d - Sigma(iw) and D = 2t is the half-bandwidth.
Equivalently: G_loc = 2 / (zeta + sqrt(zeta^2 - D^2)) (numerically stable form).

The h-ghost sector GFs follow from Schur complement relations.
"""

import numpy as np
from .matsubara import (
    fermi_function,
    matsubara_sum_numerical,
    matsubara_sum_convergence,
    matsubara_sum_pair_numerical,
    matsubara_sum_pair_convergence,
)


def bethe_local_gf(iw: np.ndarray, mu: float, eps_d: float,
                   sigma: np.ndarray, t: float) -> np.ndarray:
    """Local Green's function on the Bethe lattice.

    Uses the numerically stable form:
        G_loc = 2 / (zeta + sqrt(zeta^2 - D^2))

    where zeta = iw + mu - eps_d - sigma, D = 2t.

    Parameters
    ----------
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    mu, eps_d : float
        Chemical potential and impurity level.
    sigma : array, shape (N,)
        Self-energy on the Matsubara grid.
    t : float
        Hopping parameter (half-bandwidth D = 2t).

    Returns
    -------
    array, shape (N,)
        Local Green's function G_loc(iw_n).
    """
    D = 2.0 * t
    zeta = iw + mu - eps_d - sigma
    disc = zeta**2 - D**2
    sq = np.sqrt(disc)

    # Use the stable form G = 2 / (zeta + sqrt(zeta^2 - D^2))
    G = 2.0 / (zeta + sq)

    # Verify causality: Im G < 0 for Im(iw) > 0
    # If violated, flip to the other branch
    mask = G.imag > 0
    if np.any(mask):
        G[mask] = (zeta[mask] - sq[mask]) / (2.0 * t**2)

    return G


def bethe_self_consistency(G_loc: np.ndarray, t: float) -> np.ndarray:
    """Bethe lattice self-consistency: cavity hybridization.

    Delta^{cav}(iw) = t^2 * G_loc(iw)

    Parameters
    ----------
    G_loc : array, shape (N,)
        Local Green's function.
    t : float
        Hopping parameter.

    Returns
    -------
    array, shape (N,)
        The new hybridization function.
    """
    return t**2 * G_loc


def lattice_h_sector_gf(iw: np.ndarray, G_dd_lat: np.ndarray,
                         W: np.ndarray, eta: np.ndarray) -> dict:
    """Off-diagonal and diagonal h-ghost sector GFs on the lattice.

    From Schur complement (integrating out d from H_lat):

    G_lat^{h_l,d}(iw) = W_l / (iw - eta_l) * G_dd_lat(iw)

    G_lat^{h_l,h_l}(iw) = 1/(iw - eta_l) + |W_l|^2/(iw - eta_l)^2 * G_dd_lat(iw)

    Parameters
    ----------
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    G_dd_lat : array, shape (N,)
        Local lattice d-d Green's function.
    W : array, shape (M_h,)
        Ghost hybridization amplitudes.
    eta : array, shape (M_h,)
        Ghost level energies.

    Returns
    -------
    dict with:
        'hd': array, shape (M_h, N) — G^{h_l, d}
        'hh': array, shape (M_h, N) — G^{h_l, h_l}
    """
    M_h = len(eta)
    N = len(iw)

    # (iw - eta_l)^{-1}, shape (M_h, N)
    denom = 1.0 / (iw[None, :] - eta[:, None])

    # G^{hd}_l(iw) = W_l * denom_l * G_dd(iw)
    G_hd = W[:, None] * denom * G_dd_lat[None, :]

    # G^{hh}_l(iw) = denom_l + |W_l|^2 * denom_l^2 * G_dd(iw)
    G_hh = denom + (np.abs(W)**2)[:, None] * denom**2 * G_dd_lat[None, :]

    return {'hd': G_hd, 'hh': G_hh}


def lattice_correlators(iw: np.ndarray, G_dd_lat: np.ndarray,
                        W: np.ndarray, eta: np.ndarray,
                        beta: float,
                        return_diagnostics: bool = False,
                        diagnostic_n_values: np.ndarray = None) -> dict:
    """Equal-time lattice correlators from Matsubara sums.

    Computes:
        <h_l^dag h_l>_lat  from G_lat^{h_l, h_l}
        <d^dag h_l>_lat    from G_lat^{h_l, d}  (note: <d^dag h_l> = G_{h_l,d})

    For the h-ghost GFs, we decompose into pole contributions that
    can be summed exactly, plus a remainder involving G_dd_lat.

    Parameters
    ----------
    iw, G_dd_lat, W, eta, beta : as above.

    Returns
    -------
    dict with:
        'hh': array, shape (M_h,) — <h_l^dag h_l>_lat
        'dh': array, shape (M_h,) — <d^dag h_l>_lat
        optional 'diagnostics': convergence curves vs n_matsubara
    """
    gf = lattice_h_sector_gf(iw, G_dd_lat, W, eta)
    M_h = len(eta)

    hh_corr = np.zeros(M_h)
    dh_corr = np.zeros(M_h, dtype=complex)

    diagnostics = None
    if return_diagnostics:
        n_values = np.asarray(
            matsubara_sum_convergence(
                np.zeros_like(iw), beta, n_values=diagnostic_n_values
            )['n_matsubara']
        )
        diagnostics = {
            'n_matsubara': n_values,
            'hh': np.zeros((M_h, len(n_values))),
            'dh': np.zeros((M_h, len(n_values)), dtype=complex),
        }

    for l in range(M_h):
        # <h_l^dag h_l> = (1/beta) sum_n G^{hh}_l(iw_n) over all n
        # G^{hh}_l = 1/(iw - eta_l) + |W_l|^2/(iw - eta_l)^2 * G_dd

        # First term: 1/(iw - eta_l) -> occupancy sum with convergence factor = f(eta_l)
        term1 = fermi_function(np.array([eta[l]]), beta)[0]

        # Second term: dynamic part summed numerically with robust tail handling.
        F_hh = np.abs(W[l])**2 * G_dd_lat / (iw - eta[l])**2
        term2 = matsubara_sum_numerical(F_hh, beta).real

        hh_corr[l] = term1 + term2

        # <d^dag h_l> = (1/beta) sum_n G^{hd}_l(iw_n)
        # G^{hd}_l = W_l / (iw - eta_l) * G_dd  (for our matrix convention).
        F_dh = W[l] * G_dd_lat / (iw - eta[l])              # G_{h,d}
        F_hd = np.conj(W[l]) * G_dd_lat / (iw - eta[l])     # G_{d,h}
        dh_corr[l] = matsubara_sum_pair_numerical(
            F_dh, F_hd, beta, tail_c2_ab=W[l], tail_c2_ba=np.conj(W[l])
        )

        if return_diagnostics:
            hh_seq = matsubara_sum_convergence(
                F_hh, beta, n_values=diagnostic_n_values
            )['sum']
            dh_seq = matsubara_sum_pair_convergence(
                F_dh, F_hd, beta,
                tail_c2_ab=W[l], tail_c2_ba=np.conj(W[l]),
                n_values=diagnostic_n_values
            )['sum']
            diagnostics['hh'][l] = term1 + hh_seq.real
            diagnostics['dh'][l] = dh_seq

    out = {'hh': np.real_if_close(hh_corr), 'dh': np.real_if_close(dh_corr)}
    if diagnostics is not None:
        out['diagnostics'] = diagnostics
    return out
