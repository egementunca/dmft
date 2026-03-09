"""Tests for Green's function pole expansions."""

import numpy as np
import pytest
from dmft.greens_function import (
    hybridization,
    self_energy_poles,
    greens_function_impurity,
    weiss_field_inverse,
)
from dmft.matsubara import matsubara_frequencies


@pytest.fixture
def iw():
    beta = 50.0
    wn = matsubara_frequencies(100, beta)
    return 1j * wn


def test_hybridization_single_pole(iw):
    """Delta(iw) = |V|^2 / (iw - eps) for single bath level."""
    V = np.array([0.5])
    eps = np.array([0.3])
    delta = hybridization(iw, V, eps)
    expected = 0.25 / (iw - 0.3)
    np.testing.assert_allclose(delta, expected, atol=1e-14)


def test_hybridization_multiple_poles(iw):
    V = np.array([0.3, 0.5])
    eps = np.array([-0.5, 0.5])
    delta = hybridization(iw, V, eps)
    expected = 0.09 / (iw + 0.5) + 0.25 / (iw - 0.5)
    np.testing.assert_allclose(delta, expected, atol=1e-14)


def test_hybridization_high_freq(iw):
    """Delta(iw) ~ sum|V|^2 / iw at large frequency."""
    V = np.array([0.3, 0.5])
    eps = np.array([-0.5, 0.5])
    delta = hybridization(iw, V, eps)

    # High-frequency tail
    sum_V2 = np.sum(np.abs(V)**2)
    tail = sum_V2 / iw
    # Check last few frequencies (rtol loose since only 100 freqs)
    np.testing.assert_allclose(delta[-10:], tail[-10:], rtol=3e-2)


def test_self_energy_poles_structure(iw):
    W = np.array([0.4])
    eta = np.array([0.2])
    sigma_inf = 1.0

    sigma = self_energy_poles(iw, W, eta, sigma_inf)

    # At large frequency, Sigma -> sigma_inf
    np.testing.assert_allclose(sigma[-5:].real, sigma_inf, rtol=1e-2)


def test_dyson_equation(iw):
    """G = 1/(G_0^{-1} - Sigma) = 1/(iw + mu - eps_d - Delta - Sigma)."""
    mu = 1.0
    eps_d = 0.0
    V = np.array([0.3])
    eps = np.array([0.0])
    W = np.array([0.2])
    eta = np.array([0.1])
    sigma_inf = 1.0

    delta = hybridization(iw, V, eps)
    sigma = self_energy_poles(iw, W, eta, sigma_inf)
    G = greens_function_impurity(iw, mu, eps_d, delta, sigma)

    G_inv_expected = iw + mu - eps_d - delta - sigma
    np.testing.assert_allclose(1.0 / G, G_inv_expected, atol=1e-12)


def test_causality(iw):
    """Im G(iw) < 0 for positive Matsubara frequencies (causal)."""
    mu = 1.0
    eps_d = 0.0
    V = np.array([0.5])
    eps = np.array([0.0])

    delta = hybridization(iw, V, eps)
    sigma = np.zeros_like(iw)
    G = greens_function_impurity(iw, mu, eps_d, delta, sigma)

    assert np.all(G.imag < 0), "Green's function should be causal"
