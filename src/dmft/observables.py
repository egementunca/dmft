"""Physical observables: quasiparticle weight, spectral function, free energy."""

import numpy as np
from .greens_function import hybridization, self_energy_poles
from .matsubara import fermi_function


def quasiparticle_weight(Sigma_iw: np.ndarray, wn: np.ndarray) -> float:
    """Quasiparticle weight from lowest Matsubara frequency.

    Z = [1 - Im Sigma(iw_0) / w_0]^{-1}

    This is a Matsubara proxy — accurate in the Fermi liquid regime.
    """
    w0 = wn[0]
    dSigma = np.imag(Sigma_iw[0]) / w0
    Z = 1.0 / (1.0 - dSigma)
    return max(0.0, min(Z, 1.0))


def spectral_function(omega: np.ndarray, mu: float, eps_d: float,
                       Sigma_omega: np.ndarray, Delta_omega: np.ndarray,
                       eta: float = 0.0) -> np.ndarray:
    """Spectral function A(omega) = -(1/pi) Im G(omega + i*eta).

    Parameters
    ----------
    omega : array
        Real frequency grid.
    mu, eps_d : float
        Chemical potential and impurity level.
    Sigma_omega : array
        Self-energy on real axis (omega + i*eta).
    Delta_omega : array
        Hybridization on real axis.
    eta : float
        Additional broadening (set to 0 if already included in Sigma/Delta).
    """
    z = omega + 1j * eta
    G = 1.0 / (z + mu - eps_d - Delta_omega - Sigma_omega)
    return -G.imag / np.pi


def spectral_function_from_poles(omega: np.ndarray, mu: float, eps_d: float,
                                  V: np.ndarray, eps: np.ndarray,
                                  W: np.ndarray, eta_poles: np.ndarray,
                                  sigma_inf: float,
                                  broadening: float = 0.02) -> np.ndarray:
    """Spectral function from pole representations (trivial analytic continuation).

    Major advantage of the pole representation: just replace iw_n -> omega + i*eta.

    Parameters
    ----------
    omega : array
        Real frequency grid.
    mu, eps_d : float
    V, eps : arrays
        Bath pole parameters.
    W, eta_poles : arrays
        Ghost pole parameters.
    sigma_inf : float
    broadening : float
        Lorentzian broadening (eta in omega + i*eta).
    """
    z = omega + 1j * broadening  # shape (N_omega,)

    # Analytic continuation of hybridization
    Delta_real = np.sum(np.abs(V)**2 / (z[:, None] - eps[None, :]), axis=1)

    # Analytic continuation of self-energy
    Sigma_real = sigma_inf + np.sum(
        np.abs(W)**2 / (z[:, None] - eta_poles[None, :]), axis=1
    )

    G = 1.0 / (z + mu - eps_d - Delta_real - Sigma_real)
    return -G.imag / np.pi


def spectral_function_bethe(omega: np.ndarray, mu: float, eps_d: float,
                             Sigma_omega: np.ndarray, t: float) -> np.ndarray:
    """Spectral function on the Bethe lattice from the local GF.

    Uses the same analytic formula as bethe_local_gf but on the real axis.
    """
    D = 2.0 * t
    zeta = omega + mu - eps_d - Sigma_omega
    disc = zeta**2 - D**2
    sq = np.sqrt(disc.astype(complex))
    G = 2.0 / (zeta + sq)

    # Fix branch
    mask = G.imag > 0
    if np.any(mask):
        G[mask] = (zeta[mask] - sq[mask]) / (2.0 * t**2)

    return -G.imag / np.pi


def impurity_g_correlators(iw: np.ndarray, G_imp: np.ndarray,
                            V: np.ndarray, eps: np.ndarray,
                            beta: float) -> dict:
    """Compute impurity bath (g-sector) correlators via Matsubara sums.

    Uses Schur complement relations for the interacting impurity model:
        G_imp^{g_l,g_l}(iw) = 1/(iw - eps_l) + |V_l|^2/(iw - eps_l)^2 * G_imp(iw)
        G_imp^{g_l,d}(iw) = V_l/(iw - eps_l) * G_imp(iw)

    Then equal-time correlators from Matsubara sums.

    Parameters
    ----------
    iw : array, shape (N,)
        Positive Matsubara frequencies (1j * w_n).
    G_imp : array, shape (N,)
        Impurity Green's function on positive Matsubara frequencies.
    V, eps : arrays, shape (M_g,)
        Bath hybridization amplitudes and energies.
    beta : float
        Inverse temperature.

    Returns
    -------
    dict with:
        'gg': array (M_g,) — <g_l^dag g_l>_imp
        'dg': array (M_g,) — <d^dag g_l>_imp
    """
    M_g = len(eps)
    gg_corr = np.zeros(M_g)
    dg_corr = np.zeros(M_g)

    for l in range(M_g):
        # <g_l^dag g_l>: first term 1/(iw - eps_l)
        # The occupancy sum (with convergence factor e^{iw 0+}) gives f(eps_l)
        term1 = fermi_function(np.array([eps[l]]), beta)[0]

        # Second term: |V_l|^2 * G_imp / (iw - eps_l)^2
        F_gg = np.abs(V[l])**2 * G_imp / (iw - eps[l])**2
        term2 = (1.0 / beta) * 2.0 * np.sum(F_gg).real

        gg_corr[l] = term1 + term2

        # <d^dag g_l>: V_l^* * G_imp / (iw - eps_l)
        F_dg = np.conj(V[l]) * G_imp / (iw - eps[l])
        dg_corr[l] = (1.0 / beta) * 2.0 * np.sum(F_dg).real

    return {'gg': gg_corr, 'dg': dg_corr}
