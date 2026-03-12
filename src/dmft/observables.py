"""Physical observables: quasiparticle weight, spectral function, free energy."""

import numpy as np
import warnings
from .greens_function import hybridization, self_energy_poles
from .matsubara import (
    fermi_function,
    matsubara_sum_numerical,
    matsubara_sum_convergence,
    matsubara_sum_pair_numerical,
    matsubara_sum_pair_convergence,
)


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
                            beta: float,
                            return_diagnostics: bool = False,
                            diagnostic_n_values: np.ndarray = None) -> dict:
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
        optional 'diagnostics': convergence curves vs n_matsubara
    """
    M_g = len(eps)
    gg_corr = np.zeros(M_g)
    dg_corr = np.zeros(M_g, dtype=complex)

    diagnostics = None
    if return_diagnostics:
        n_values = np.asarray(
            matsubara_sum_convergence(
                np.zeros_like(iw), beta, n_values=diagnostic_n_values
            )['n_matsubara']
        )
        diagnostics = {
            'n_matsubara': n_values,
            'gg': np.zeros((M_g, len(n_values))),
            'dg': np.zeros((M_g, len(n_values)), dtype=complex),
        }

    for l in range(M_g):
        # <g_l^dag g_l>: first term 1/(iw - eps_l)
        # The occupancy sum (with convergence factor e^{iw 0+}) gives f(eps_l)
        term1 = fermi_function(np.array([eps[l]]), beta)[0]

        # Second term: |V_l|^2 * G_imp / (iw - eps_l)^2
        F_gg = np.abs(V[l])**2 * G_imp / (iw - eps[l])**2
        term2 = matsubara_sum_numerical(F_gg, beta).real

        gg_corr[l] = term1 + term2

        # <d^dag g_l>: sum_n G_{g_l,d}(iw_n), and G_{g_l,d}=V_l/(iw-eps_l)*G_imp
        F_dg = V[l] * G_imp / (iw - eps[l])              # G_{g,d}
        F_gd = np.conj(V[l]) * G_imp / (iw - eps[l])     # G_{d,g}
        dg_corr[l] = matsubara_sum_pair_numerical(
            F_dg, F_gd, beta, tail_c2_ab=V[l], tail_c2_ba=np.conj(V[l])
        )

        if return_diagnostics:
            gg_seq = matsubara_sum_convergence(
                F_gg, beta, n_values=diagnostic_n_values
            )['sum']
            dg_seq = matsubara_sum_pair_convergence(
                F_dg, F_gd, beta,
                tail_c2_ab=V[l], tail_c2_ba=np.conj(V[l]),
                n_values=diagnostic_n_values
            )['sum']
            diagnostics['gg'][l] = term1 + gg_seq.real
            diagnostics['dg'][l] = dg_seq

    out = {'gg': np.real_if_close(gg_corr), 'dg': np.real_if_close(dg_corr)}
    if diagnostics is not None:
        out['diagnostics'] = diagnostics
    return out


def check_pole_sigma_consistency(poles, Sigma: np.ndarray, iw: np.ndarray,
                                 tol: float = 0.1) -> float:
    """Check consistency between pole self-energy and full Matsubara self-energy.

    Parameters
    ----------
    poles : PoleParams-like
        Object with attributes `W`, `eta`, and `sigma_inf`.
    Sigma : array, shape (N,)
        Reference self-energy on the Matsubara grid.
    iw : array, shape (N,)
        Matsubara frequencies (1j*w_n) matching `Sigma`.
    tol : float
        Warning threshold on max absolute mismatch.

    Returns
    -------
    float
        max_n |Sigma_poles(iw_n) - Sigma(iw_n)|
    """
    Sigma_poles = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)
    max_diff = float(np.max(np.abs(np.asarray(Sigma_poles) - np.asarray(Sigma))))
    if max_diff > tol:
        warnings.warn(
            f"Pole Sigma differs from loop Sigma by {max_diff:.4f} (tol={tol:.4f})",
            RuntimeWarning,
            stacklevel=2,
        )
    return max_diff
