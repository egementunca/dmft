"""Matsubara frequency grid, Fermi function, and summation utilities.

Key identity exploited throughout:
    (1/beta) sum_n 1/(iw_n - x) = -f(x)
where f(x) = 1/(exp(beta*x) + 1) is the Fermi function.

For pole-form Green's functions, this gives EXACT Matsubara sums
with no truncation error.
"""

import numpy as np


def matsubara_frequencies(n_max: int, beta: float) -> np.ndarray:
    """Fermionic Matsubara frequencies w_n = (2n+1)*pi/beta.

    Parameters
    ----------
    n_max : int
        Number of positive frequencies (n = 0, 1, ..., n_max-1).
    beta : float
        Inverse temperature.

    Returns
    -------
    np.ndarray
        Real array of frequencies, shape (n_max,).
    """
    n = np.arange(n_max)
    return (2 * n + 1) * np.pi / beta


def fermi_function(x: np.ndarray, beta: float) -> np.ndarray:
    """Fermi-Dirac distribution f(x) = 1/(exp(beta*x) + 1).

    Handles overflow for large |beta*x|.
    """
    x = np.asarray(x, dtype=float)
    bx = beta * x
    result = np.empty_like(bx)
    large_pos = bx > 500
    large_neg = bx < -500
    normal = ~(large_pos | large_neg)
    result[large_pos] = 0.0
    result[large_neg] = 1.0
    result[normal] = 1.0 / (np.exp(bx[normal]) + 1.0)
    return result


def pole_matsubara_sum(residues: np.ndarray, poles: np.ndarray,
                       beta: float) -> complex:
    """Exact Matsubara sum for a pole-form function.

    For G(iw_n) = sum_l residues[l] / (iw_n - poles[l]),
    the full Matsubara sum (positive + negative frequencies) is:

        (1/beta) sum_{all n} G(iw_n) = -sum_l residues[l] * f(poles[l])

    This is the equal-time correlator <b^dag a> when G = G_ab.

    Parameters
    ----------
    residues : array, shape (M,)
        Residues at each pole (e.g., |V_l|^2 for hybridization).
    poles : array, shape (M,)
        Pole positions (e.g., eps_l for bath levels).
    beta : float
        Inverse temperature.

    Returns
    -------
    complex
        The Matsubara sum value.
    """
    residues = np.asarray(residues)
    poles = np.asarray(poles, dtype=float)
    return -np.sum(residues * fermi_function(poles, beta))


def matsubara_sum_numerical(G_positive: np.ndarray, beta: float,
                            tail_c1: complex = 0.0,
                            tail_c2: complex = 0.0) -> complex:
    """Numerical Matsubara sum with tail subtraction.

    Computes (1/beta) sum_{all n} G(iw_n) given G at positive frequencies only.
    Uses G(-iw_n) = G(iw_n)* for real Hamiltonians.

    High-frequency tail: G(iw_n) ~ c1/(iw_n) + c2/(iw_n)^2 + ...

    Analytically known sums:
        (1/beta) sum_{all n} 1/(iw_n) = 0  (imaginary, cancels between +/- n)
        (1/beta) sum_{all n} 1/(iw_n)^2 = -beta/4

    Parameters
    ----------
    G_positive : array, shape (N,)
        Green's function at positive Matsubara frequencies.
    beta : float
        Inverse temperature.
    tail_c1, tail_c2 : complex
        First two high-frequency tail coefficients.

    Returns
    -------
    complex
        The Matsubara sum value.
    """
    n_w = len(G_positive)
    wn = matsubara_frequencies(n_w, beta)
    iw = 1j * wn

    # Subtract tail
    G_subtracted = G_positive - tail_c1 / iw - tail_c2 / iw**2

    # Numerical sum over positive frequencies, doubled for negative
    # G(-iw) = G(iw)* so contribution is 2*Re[G(iw)] for real part
    # and the imaginary parts cancel for the c1 tail (purely imaginary).
    numerical_sum = (1.0 / beta) * 2.0 * np.sum(G_subtracted).real

    # Add back analytic tail contributions
    #
    # c1 term: (1/beta) sum_{all n} e^{iw_n 0+} c1/(iw_n) = -c1/2
    #   The convergence factor e^{iw_n 0+} is ESSENTIAL here.
    #   Without it, the sum vanishes by antisymmetry. With it, contour
    #   integration gives -c1/2. This is the Fermi function at x=0.
    #
    # c2 term: (1/beta) sum_{all n} c2/(iw_n)^2 = c2 * (-beta/4)
    #   This sum converges absolutely, so e^{iw_n 0+} is irrelevant.
    analytic_c1 = -tail_c1 / 2.0
    analytic_c2 = tail_c2 * (-beta / 4.0)

    return numerical_sum + analytic_c1 + analytic_c2


def matsubara_sum_full(G_positive: np.ndarray, beta: float) -> complex:
    """Simple Matsubara sum without tail subtraction.

    (1/beta) sum_{all n} G(iw_n) using G(-iw) = G(iw)*

    Only use for functions that decay faster than 1/w^2.
    """
    return (1.0 / beta) * 2.0 * np.sum(G_positive).real
