"""Pole-expansion representations of hybridization and self-energy.

Bath hybridization (g-levels):
    Delta(iw) = sum_l |V_l|^2 / (iw - eps_l)

Self-energy (h-ghosts):
    Sigma(iw) = sigma_inf + sum_l |W_l|^2 / (iw - eta_l)

Impurity Green's function (Dyson equation):
    G(iw) = 1 / (iw + mu - eps_d - Delta(iw) - Sigma(iw))
"""

import numpy as np


def hybridization(iw: np.ndarray, V: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Compute bath hybridization function Delta(iw).

    Delta(iw_n) = sum_l |V_l|^2 / (iw_n - eps_l)

    Parameters
    ----------
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    V : array, shape (M_g,)
        Bath hybridization amplitudes.
    eps : array, shape (M_g,)
        Bath level energies.

    Returns
    -------
    array, shape (N,)
    """
    # Broadcasting: iw[:, None] - eps[None, :]  -> (N, M_g)
    return np.sum(np.abs(V)**2 / (iw[:, None] - eps[None, :]), axis=1)


def self_energy_poles(iw: np.ndarray, W: np.ndarray, eta: np.ndarray,
                      sigma_inf: float) -> np.ndarray:
    """Compute pole-expanded self-energy Sigma(iw).

    Sigma(iw_n) = sigma_inf + sum_l |W_l|^2 / (iw_n - eta_l)

    Parameters
    ----------
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    W : array, shape (M_h,)
        Ghost hybridization amplitudes.
    eta : array, shape (M_h,)
        Ghost level energies.
    sigma_inf : float
        Static (high-frequency) self-energy.

    Returns
    -------
    array, shape (N,)
    """
    dynamic = np.sum(np.abs(W)**2 / (iw[:, None] - eta[None, :]), axis=1)
    return sigma_inf + dynamic


def greens_function_impurity(iw: np.ndarray, mu: float, eps_d: float,
                              delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Impurity Green's function from Dyson equation.

    G(iw) = 1 / (iw + mu - eps_d - Delta(iw) - Sigma(iw))

    Parameters
    ----------
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    mu : float
        Chemical potential.
    eps_d : float
        Impurity level energy.
    delta : array, shape (N,)
        Hybridization function.
    sigma : array, shape (N,)
        Self-energy.

    Returns
    -------
    array, shape (N,)
    """
    return 1.0 / (iw + mu - eps_d - delta - sigma)


def weiss_field_inverse(iw: np.ndarray, mu: float, eps_d: float,
                         delta: np.ndarray) -> np.ndarray:
    """Inverse Weiss field (bare impurity propagator).

    G_0^{-1}(iw) = iw + mu - eps_d - Delta(iw)
    """
    return iw + mu - eps_d - delta
