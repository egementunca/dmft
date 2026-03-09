"""Iterated Perturbation Theory (IPT) solver for the Anderson impurity model.

IPT at half-filling on the Bethe lattice:
    Sigma(iw) = U/2 + U^2 * (1/beta) sum_tau e^{iw tau} G0_tilde(tau)^3

where G0_tilde^{-1}(iw) = G_0^{-1}(iw) - U/2 enforces particle-hole symmetry.

References:
    Georges et al., Rev. Mod. Phys. 68, 13 (1996), Eq. (157).
"""

import numpy as np
from .base import ImpuritySolver
from ..greens_function import hybridization


class IPTSolver(ImpuritySolver):
    """IPT solver — fast, decent at half-filling."""

    def solve(self, iw, mu, eps_d, U, V, eps, beta, sigma_inf):
        n_w = len(iw)

        # Build hybridization
        delta = hybridization(iw, V, eps)

        # Weiss field: G_0^{-1} = iw + mu - eps_d - Delta
        G0_inv = iw + mu - eps_d - delta

        # Particle-hole shifted Weiss field for half-filling IPT
        # G0_tilde^{-1} = G_0^{-1} - U/2
        G0_tilde = 1.0 / (G0_inv - U / 2.0)

        # Transform to imaginary time via Fourier
        n_tau = 4 * n_w  # oversample for accuracy
        tau = np.linspace(0, beta, n_tau, endpoint=False)
        dtau = beta / n_tau

        G0_tilde_tau = _matsubara_to_tau(G0_tilde, iw, tau, beta)

        # Second-order self-energy in tau
        Sigma2_tau = U**2 * G0_tilde_tau**3

        # Transform back to Matsubara
        Sigma2_iw = _tau_to_matsubara(Sigma2_tau, tau, iw, beta)

        # Full self-energy
        Sigma_iw = U / 2.0 + Sigma2_iw

        # Impurity Green's function
        G_imp = 1.0 / (G0_inv - Sigma_iw)

        # At half-filling: n = 0.5 by symmetry
        n_imp = 0.5

        return {
            'G_imp': G_imp,
            'Sigma_imp': Sigma_iw,
            'n_imp': n_imp,
            'n_double': _estimate_double_occ(U, Sigma_iw, G_imp, beta),
        }


def _matsubara_to_tau(G_iw, iw, tau, beta):
    """Fourier transform from Matsubara to imaginary time with tail subtraction.

    G(tau) = (1/beta) sum_n e^{-iw_n tau} G(iw_n)  [sum over ALL n]

    Using G(-iw) = G(iw)* and e^{-i(-w)tau} = e^{iwt}:
        G(tau) = (2/beta) Re[sum_{n>=0} e^{-iw_n tau} G(iw_n)]

    Tail subtraction: G ~ c1/iw at high freq.
    Analytic: (1/beta) sum_n e^{-iw_n tau} * c1/(iw_n) = -c1/2 for 0 < tau < beta
    (this is the free-particle Green's function at half-filling).
    """
    wn = np.imag(iw)  # real frequencies

    # High-frequency tail: G ~ 1/iw, c1 = 1
    c1 = 1.0
    G_sub = G_iw - c1 / iw

    # Numerical sum of subtracted part (decays as 1/w^2)
    # phase[n, t] = exp(-i w_n tau_t)
    phase = np.exp(-1j * wn[:, None] * tau[None, :])  # (n_w, n_tau)
    G_tau = (2.0 / beta) * np.real(G_sub[:, None] * phase).sum(axis=0)

    # Add back analytic tail: -c1/2 for 0 < tau < beta
    G_tau += -c1 / 2.0

    return G_tau


def _tau_to_matsubara(F_tau, tau, iw, beta):
    """Fourier transform from imaginary time to Matsubara.

    F(iw_n) = integral_0^beta dtau e^{iw_n tau} F(tau)

    Discretized with trapezoidal rule.
    """
    wn = np.imag(iw)
    dtau = tau[1] - tau[0]

    # phase[t, n] = exp(i w_n tau_t)
    phase = np.exp(1j * wn[None, :] * tau[:, None])  # (n_tau, n_w)
    F_iw = dtau * np.sum(F_tau[:, None] * phase, axis=0)

    return F_iw


def _estimate_double_occ(U, Sigma_iw, G_imp, beta):
    """Estimate double occupancy from Galitskii-Migdal formula.

    <n_up n_down> = (1/(U*beta)) sum_n [Sigma(iw_n) - U/2] * G(iw_n)
    (at half-filling).
    """
    if abs(U) < 1e-10:
        return 0.25  # non-interacting half-filling
    Sigma_dynamic = Sigma_iw - U / 2.0
    # Sum over all n: use 2*Re[sum_{n>=0}]
    val = (2.0 / beta) * np.sum(Sigma_dynamic * G_imp).real
    return val / U
