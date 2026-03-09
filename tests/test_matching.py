"""Tests for pole fitting and correlator matching."""

import numpy as np
import pytest
from dmft.matsubara import matsubara_frequencies
from dmft.greens_function import hybridization, self_energy_poles
from dmft.matching import (
    fit_hybridization_poles,
    fit_self_energy_poles,
    match_h_correlators,
    match_g_correlators,
)
from dmft.gateway import gateway_correlators


@pytest.fixture
def iw():
    return 1j * matsubara_frequencies(512, 50.0)


# ---------------------------------------------------------------------------
# Pole fitting tests
# ---------------------------------------------------------------------------

def test_fit_hybridization_roundtrip(iw):
    """Fit poles to a known pole-form hybridization and recover parameters."""
    V_true = np.array([0.3, 0.5])
    eps_true = np.array([-0.4, 0.4])
    delta = hybridization(iw, V_true, eps_true)

    V_fit, eps_fit = fit_hybridization_poles(delta, iw, 2, symmetric=False)

    # Check reconstructed Delta matches
    delta_fit = hybridization(iw, V_fit, eps_fit)
    np.testing.assert_allclose(delta_fit, delta, atol=1e-8)


def test_fit_hybridization_symmetric(iw):
    """Symmetric fit recovers PH-symmetric poles."""
    V_true = np.array([0.4, 0.4])
    eps_true = np.array([-0.5, 0.5])
    delta = hybridization(iw, V_true, eps_true)

    V_fit, eps_fit = fit_hybridization_poles(delta, iw, 2, symmetric=True)

    delta_fit = hybridization(iw, V_fit, eps_fit)
    np.testing.assert_allclose(delta_fit, delta, atol=1e-8)
    # Symmetry: eps should be +-symmetric
    assert abs(eps_fit[0] + eps_fit[1]) < 1e-6


def test_fit_self_energy_roundtrip(iw):
    """Fit poles to a known pole-form self-energy."""
    # Use equal W for symmetric mode (constrains W_pair equal for +/- eta)
    W_true = np.array([0.3, 0.3])
    eta_true = np.array([-0.5, 0.5])
    sigma_inf = 1.0
    sigma = self_energy_poles(iw, W_true, eta_true, sigma_inf)

    W_fit, eta_fit = fit_self_energy_poles(sigma, iw, sigma_inf, 2,
                                            symmetric=True)

    sigma_fit = self_energy_poles(iw, W_fit, eta_fit, sigma_inf)
    np.testing.assert_allclose(sigma_fit, sigma, atol=1e-8)


# ---------------------------------------------------------------------------
# Correlator matching tests
# ---------------------------------------------------------------------------

def test_match_h_correlators_roundtrip():
    """Match h-correlators with known gateway parameters recovers bath poles."""
    beta = 50.0
    mu, eps_d, sigma_inf = 1.0, 0.0, 1.0
    V_true = np.array([0.3, 0.5])
    eps_true = np.array([-0.4, 0.4])
    W = np.array([0.2, 0.4])
    eta = np.array([-0.5, 0.5])

    # Generate target correlators from the true parameters
    corr = gateway_correlators(mu, eps_d, sigma_inf,
                                V_true, eps_true, W, eta, beta)
    target_hh = corr['hh']
    target_dh = corr['dh']

    # Recover bath poles from matching
    V_fit, eps_fit = match_h_correlators(
        target_hh, target_dh, mu, eps_d, sigma_inf,
        W, eta, M_g=2, beta=beta, symmetric=False)

    # Verify by checking that correlators match
    corr_fit = gateway_correlators(mu, eps_d, sigma_inf,
                                    V_fit, eps_fit, W, eta, beta)
    np.testing.assert_allclose(corr_fit['hh'], target_hh, atol=1e-8)
    np.testing.assert_allclose(corr_fit['dh'], target_dh, atol=1e-8)


def test_match_g_correlators_roundtrip():
    """Match g-correlators with known gateway parameters recovers ghost poles."""
    beta = 50.0
    mu, eps_d, sigma_inf = 1.0, 0.0, 1.0
    V = np.array([0.3, 0.5])
    eps = np.array([-0.4, 0.4])
    W_true = np.array([0.2, 0.4])
    eta_true = np.array([-0.5, 0.5])

    # Generate target correlators from the true parameters
    corr = gateway_correlators(mu, eps_d, sigma_inf,
                                V, eps, W_true, eta_true, beta)
    target_gg = corr['gg']
    target_dg = corr['dg']

    # Recover ghost poles from matching
    W_fit, eta_fit = match_g_correlators(
        target_gg, target_dg, mu, eps_d, sigma_inf,
        V, eps, M_h=2, beta=beta, symmetric=False)

    # Verify by checking that correlators match
    corr_fit = gateway_correlators(mu, eps_d, sigma_inf,
                                    V, eps, W_fit, eta_fit, beta)
    np.testing.assert_allclose(corr_fit['gg'], target_gg, atol=1e-8)
    np.testing.assert_allclose(corr_fit['dg'], target_dg, atol=1e-8)


def test_match_h_symmetric():
    """Symmetric h-correlator matching preserves PH symmetry."""
    beta = 50.0
    mu, eps_d, sigma_inf = 1.0, 0.0, 1.0
    V_true = np.array([0.4, 0.4])
    eps_true = np.array([-0.5, 0.5])
    W = np.array([0.3, 0.3])
    eta = np.array([-0.6, 0.6])

    corr = gateway_correlators(mu, eps_d, sigma_inf,
                                V_true, eps_true, W, eta, beta)

    V_fit, eps_fit = match_h_correlators(
        corr['hh'], corr['dh'], mu, eps_d, sigma_inf,
        W, eta, M_g=2, beta=beta, symmetric=True)

    # Symmetry check
    assert abs(eps_fit[0] + eps_fit[1]) < 1e-6


def test_match_g_symmetric():
    """Symmetric g-correlator matching preserves PH symmetry."""
    beta = 50.0
    mu, eps_d, sigma_inf = 1.0, 0.0, 1.0
    V = np.array([0.4, 0.4])
    eps = np.array([-0.5, 0.5])
    W_true = np.array([0.3, 0.3])
    eta_true = np.array([-0.6, 0.6])

    corr = gateway_correlators(mu, eps_d, sigma_inf,
                                V, eps, W_true, eta_true, beta)

    W_fit, eta_fit = match_g_correlators(
        corr['gg'], corr['dg'], mu, eps_d, sigma_inf,
        V, eps, M_h=2, beta=beta, symmetric=True)

    assert abs(eta_fit[0] + eta_fit[1]) < 1e-6
