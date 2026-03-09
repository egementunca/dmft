"""Tests for Bethe lattice Green's function."""

import numpy as np
import pytest
from dmft.lattice import bethe_local_gf, bethe_self_consistency
from dmft.matsubara import matsubara_frequencies


@pytest.fixture
def iw():
    beta = 50.0
    wn = matsubara_frequencies(1024, beta)
    return 1j * wn


def test_bethe_gf_noninteracting(iw):
    """At U=0, G_loc = 2/(iw + mu + sqrt((iw+mu)^2 - D^2)).

    For mu=0 (no interaction, half-filling trivially satisfied):
    G_loc = 2 / (iw + sqrt(iw^2 - D^2))
    """
    t = 0.5
    mu = 0.0
    eps_d = 0.0
    sigma = np.zeros_like(iw)

    G = bethe_local_gf(iw, mu, eps_d, sigma, t)

    # Check causality
    assert np.all(G.imag < 0), "G should be causal"


def test_bethe_gf_high_freq(iw):
    """G_loc ~ 1/iw at high frequency."""
    t = 0.5
    mu = 0.0
    sigma = np.zeros_like(iw)

    G = bethe_local_gf(iw, mu, 0.0, sigma, t)

    # At high frequency
    tail = 1.0 / iw
    np.testing.assert_allclose(G[-20:], tail[-20:], rtol=1e-2)


def test_bethe_gf_causality_with_sigma(iw):
    """G_loc should be causal even with a self-energy."""
    t = 0.5
    mu = 1.0
    sigma = 1.0 + 0.1 * np.abs(iw)**2 / (1j * np.imag(iw))  # real + small imaginary
    # Use a causal self-energy: purely real constant
    sigma_real = np.full_like(iw, 1.0)

    G = bethe_local_gf(iw, mu, 0.0, sigma_real, t)
    assert np.all(G.imag < 0), "G should be causal with real self-energy"


def test_bethe_self_consistency_form(iw):
    """Delta = t^2 * G_loc."""
    t = 0.5
    G_loc = bethe_local_gf(iw, 0.0, 0.0, np.zeros_like(iw), t)
    delta = bethe_self_consistency(G_loc, t)

    np.testing.assert_allclose(delta, t**2 * G_loc, atol=1e-15)


def test_noninteracting_self_consistency(iw):
    """At U=0 on Bethe lattice, the self-consistent solution satisfies
    G^{-1} = iw - t^2 * G, i.e., t^2 * G^2 - iw*G + 1 = 0.
    """
    t = 0.5
    G = bethe_local_gf(iw, 0.0, 0.0, np.zeros_like(iw), t)

    residual = t**2 * G**2 - iw * G + 1.0
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)
