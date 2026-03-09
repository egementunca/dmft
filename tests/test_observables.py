"""Tests for observables: Z, spectral function, bath correlators."""

import numpy as np
import pytest
from dmft.matsubara import matsubara_frequencies
from dmft.solvers.ed import EDSolver
from dmft.observables import (
    quasiparticle_weight,
    impurity_g_correlators,
)


def test_z_noninteracting():
    """Z = 1 for non-interacting system (Sigma = 0)."""
    wn = matsubara_frequencies(256, 50.0)
    Sigma = np.zeros(256, dtype=complex)
    Z = quasiparticle_weight(Sigma, wn)
    assert abs(Z - 1.0) < 1e-12


def test_z_bounded():
    """Z should be clamped to [0, 1]."""
    wn = matsubara_frequencies(256, 50.0)
    # Large imaginary self-energy -> small Z
    Sigma = -1j * 5.0 * np.ones(256)
    Z = quasiparticle_weight(Sigma, wn)
    assert 0.0 <= Z <= 1.0


def test_impurity_g_correlators_vs_ed():
    """impurity_g_correlators matches ED exact bath correlators."""
    beta = 50.0
    n_w = 2048  # Need many frequencies for convergence of off-diagonal sums
    iw = 1j * matsubara_frequencies(n_w, beta)

    V = np.array([0.3, 0.5])
    eps = np.array([-0.4, 0.4])
    U = 2.0
    mu = U / 2

    solver = EDSolver()
    result = solver.solve(iw, mu, 0.0, U, V, eps, beta, mu)

    # Matsubara-sum correlators
    corr = impurity_g_correlators(iw, result['G_imp'], V, eps, beta)

    # ED exact correlators — gg converges faster, dg needs more frequencies
    np.testing.assert_allclose(corr['gg'], result['bath_gg'], atol=1e-4)
    np.testing.assert_allclose(corr['dg'], result['bath_dg'], atol=1e-3)


def test_ed_bath_correlators_occupancy_bounds():
    """ED bath occupancies should be in [0, 1]."""
    beta = 50.0
    iw = 1j * matsubara_frequencies(256, beta)
    V = np.array([0.4, 0.4])
    eps = np.array([-0.5, 0.5])

    solver = EDSolver()
    result = solver.solve(iw, 1.0, 0.0, 2.0, V, eps, beta, 1.0)

    assert np.all(result['bath_gg'] >= -0.01)
    assert np.all(result['bath_gg'] <= 1.01)
