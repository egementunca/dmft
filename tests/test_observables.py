"""Tests for observables: Z, spectral function, bath correlators."""

import numpy as np
import pytest
from dmft.matsubara import matsubara_frequencies
from dmft.solvers.ed import EDSolver
from dmft.observables import (
    quasiparticle_weight,
    impurity_g_correlators,
    spectral_function_from_poles,
    check_pole_sigma_consistency,
)
from dmft.greens_function import self_energy_poles
from dmft.config import PoleParams


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


def test_impurity_g_correlator_diagnostics_converge():
    """Diagnostics should show improved dg accuracy as n_matsubara increases."""
    beta = 50.0
    n_w = 2048
    iw = 1j * matsubara_frequencies(n_w, beta)

    V = np.array([0.3, 0.5])
    eps = np.array([-0.4, 0.4])
    U = 2.0
    mu = U / 2

    solver = EDSolver()
    result = solver.solve(iw, mu, 0.0, U, V, eps, beta, 0.0)

    diag_n = np.array([64, 128, 256, 512, 1024, 2048])
    corr = impurity_g_correlators(
        iw, result['G_imp'], V, eps, beta,
        return_diagnostics=True, diagnostic_n_values=diag_n,
    )
    dg_seq = corr['diagnostics']['dg']

    err_small = np.max(np.abs(dg_seq[:, 0] - result['bath_dg']))
    err_large = np.max(np.abs(dg_seq[:, -1] - result['bath_dg']))
    assert err_large < err_small


def test_impurity_g_correlators_complex_hybridization():
    """Complex V should produce consistent complex <d^dag g> values."""
    beta = 50.0
    n_w = 2048
    iw = 1j * matsubara_frequencies(n_w, beta)

    V = np.array([0.3 + 0.2j, -0.25 + 0.15j])
    eps = np.array([-0.4, 0.4])
    U = 2.0
    mu = U / 2

    solver = EDSolver()
    result = solver.solve(iw, mu, 0.0, U, V, eps, beta, 0.0)
    corr = impurity_g_correlators(iw, result['G_imp'], V, eps, beta)

    np.testing.assert_allclose(corr['gg'], result['bath_gg'], atol=2e-4)
    np.testing.assert_allclose(corr['dg'], result['bath_dg'], atol=2e-3)


def test_spectral_function_sum_rule():
    """Pole-based spectral function should satisfy the unit spectral-weight sum rule."""
    omega = np.linspace(-20.0, 20.0, 8001)
    A = spectral_function_from_poles(
        omega,
        mu=0.0,
        eps_d=0.0,
        V=np.array([0.4, 0.4]),
        eps=np.array([-0.5, 0.5]),
        W=np.array([0.0]),
        eta_poles=np.array([0.0]),
        sigma_inf=0.0,
        broadening=0.02,
    )
    weight = np.trapezoid(A, omega)
    assert abs(weight - 1.0) < 0.05


def test_check_pole_sigma_consistency():
    """Consistency helper should return max error and warn above tolerance."""
    iw = 1j * matsubara_frequencies(256, 50.0)
    poles = PoleParams(
        eps=np.array([-0.5, 0.5]),
        V=np.array([0.3, 0.3]),
        eta=np.array([-0.4, 0.4]),
        W=np.array([0.25, 0.25]),
        sigma_inf=1.0,
    )
    sigma_ref = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)
    err_ok = check_pole_sigma_consistency(poles, sigma_ref, iw, tol=1e-6)
    assert err_ok < 1e-12

    sigma_bad = sigma_ref + 0.2
    with pytest.warns(RuntimeWarning):
        err_bad = check_pole_sigma_consistency(poles, sigma_bad, iw, tol=0.1)
    assert err_bad > 0.1
