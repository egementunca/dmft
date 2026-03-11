"""Tests for the gateway (H_imp^(0)) quadratic model."""

import numpy as np
import pytest
from dmft.gateway import (
    gateway_greens_functions,
    gateway_correlators,
    gateway_correlators_from_matsubara,
    gateway_onebody_matrix,
)
from dmft.greens_function import hybridization, self_energy_poles
from dmft.matsubara import matsubara_frequencies


@pytest.fixture
def params():
    return {
        'mu': 1.0,
        'eps_d': 0.0,
        'sigma_inf': 1.0,
        'V': np.array([0.3, 0.5]),
        'eps': np.array([-0.4, 0.4]),
        'W': np.array([0.2, 0.4]),
        'eta': np.array([-0.5, 0.5]),
        'beta': 50.0,
        'n_w': 1024,
    }


@pytest.fixture
def iw(params):
    wn = matsubara_frequencies(params['n_w'], params['beta'])
    return 1j * wn


def test_gateway_gdd_equals_dyson(iw, params):
    """G_dd^(0) = 1/(iw + mu - eps_d - sigma_inf - Delta - Sigma_h)."""
    p = params
    gf = gateway_greens_functions(iw, p['mu'], p['eps_d'], p['sigma_inf'],
                                   p['V'], p['eps'], p['W'], p['eta'])

    delta = hybridization(iw, p['V'], p['eps'])
    sigma_h = self_energy_poles(iw, p['W'], p['eta'], 0.0)  # no sigma_inf here
    expected = 1.0 / (iw + p['mu'] - p['eps_d'] - p['sigma_inf'] - delta - sigma_h)

    np.testing.assert_allclose(gf['dd'], expected, atol=1e-12)


def test_gateway_vs_direct_inverse(iw, params):
    """Gateway GF blocks match direct inversion of full matrix."""
    p = params
    M_g = len(p['eps'])
    M_h = len(p['eta'])
    dim = 1 + M_g + M_h

    # Build full inverse resolvent at one frequency
    w = iw[10]
    G_inv = np.zeros((dim, dim), dtype=complex)
    G_inv[0, 0] = w + p['mu'] - p['eps_d'] - p['sigma_inf']

    for l in range(M_g):
        G_inv[0, 1 + l] = -np.conj(p['V'][l])
        G_inv[1 + l, 0] = -p['V'][l]
        G_inv[1 + l, 1 + l] = w - p['eps'][l]

    for l in range(M_h):
        G_inv[0, 1 + M_g + l] = -np.conj(p['W'][l])
        G_inv[1 + M_g + l, 0] = -p['W'][l]
        G_inv[1 + M_g + l, 1 + M_g + l] = w - p['eta'][l]

    G_full = np.linalg.inv(G_inv)

    # Compare with our routine (need single-frequency version)
    iw_single = np.array([w])
    gf = gateway_greens_functions(iw_single, p['mu'], p['eps_d'], p['sigma_inf'],
                                   p['V'], p['eps'], p['W'], p['eta'])

    np.testing.assert_allclose(gf['dd'][0], G_full[0, 0], atol=1e-12)

    for l in range(M_g):
        np.testing.assert_allclose(gf['gd'][l, 0], G_full[1 + l, 0], atol=1e-12)
        np.testing.assert_allclose(gf['gg'][l, 0], G_full[1 + l, 1 + l], atol=1e-12)

    for l in range(M_h):
        np.testing.assert_allclose(gf['hd'][l, 0], G_full[1 + M_g + l, 0], atol=1e-12)
        np.testing.assert_allclose(gf['hh'][l, 0], G_full[1 + M_g + l, 1 + M_g + l], atol=1e-12)


def test_gateway_gdd_causality(iw, params):
    """Im G_dd < 0 for causal Green's function."""
    p = params
    gf = gateway_greens_functions(iw, p['mu'], p['eps_d'], p['sigma_inf'],
                                   p['V'], p['eps'], p['W'], p['eta'])
    assert np.all(gf['dd'].imag < 0)


def test_gateway_correlators_run(params):
    """Gateway correlators should be real and between 0 and 1 (quadratic model)."""
    p = params
    corr = gateway_correlators(p['mu'], p['eps_d'], p['sigma_inf'],
                                p['V'], p['eps'], p['W'], p['eta'], p['beta'])

    assert corr['hh'].shape == (len(p['eta']),)
    assert corr['gg'].shape == (len(p['eps']),)
    assert np.all(np.isfinite(corr['hh']))
    assert np.all(np.isfinite(corr['gg']))
    # For a quadratic model, all occupancies must be in [0, 1]
    assert np.all(corr['hh'] >= -1e-12) and np.all(corr['hh'] <= 1.0 + 1e-12)
    assert np.all(corr['gg'] >= -1e-12) and np.all(corr['gg'] <= 1.0 + 1e-12)


def test_gateway_correlators_vs_direct_diag(params):
    """Gateway correlators match direct diagonalization of K."""
    p = params
    from dmft.matsubara import fermi_function
    K = gateway_onebody_matrix(p['mu'], p['eps_d'], p['sigma_inf'],
                                p['V'], p['eps'], p['W'], p['eta'])
    eigvals, U = np.linalg.eigh(K)
    f_e = fermi_function(eigvals, p['beta'])
    f_matrix = U @ np.diag(f_e) @ U.conj().T

    corr = gateway_correlators(p['mu'], p['eps_d'], p['sigma_inf'],
                                p['V'], p['eps'], p['W'], p['eta'], p['beta'])

    # Check <g_l^dag g_l> = f_matrix[1+l, 1+l]
    M_g = len(p['eps'])
    for l in range(M_g):
        np.testing.assert_allclose(corr['gg'][l], f_matrix[1 + l, 1 + l], atol=1e-12)

    # Check <h_l^dag h_l> = f_matrix[1+M_g+l, 1+M_g+l]
    M_h = len(p['eta'])
    for l in range(M_h):
        np.testing.assert_allclose(corr['hh'][l], f_matrix[1 + M_g + l, 1 + M_g + l], atol=1e-12)

    # Off-diagonal orientation: <d^dag g_l> = [f(K)]_{g_l,d}
    for l in range(M_g):
        np.testing.assert_allclose(corr['dg'][l], f_matrix[1 + l, 0], atol=1e-12)

    # Off-diagonal orientation: <d^dag h_l> = [f(K)]_{h_l,d}
    for l in range(M_h):
        np.testing.assert_allclose(corr['dh'][l], f_matrix[1 + M_g + l, 0], atol=1e-12)


def test_gateway_correlators_complex_hermitian():
    """gateway_correlators handles complex Hermitian K correctly."""
    beta = 40.0
    mu, eps_d, sigma_inf = 1.2, 0.1, 0.4
    V = np.array([0.35 + 0.2j, -0.28 + 0.1j])
    eps = np.array([-0.3, 0.45])
    W = np.array([0.22 - 0.17j, 0.18 + 0.09j])
    eta = np.array([-0.55, 0.62])

    K = gateway_onebody_matrix(mu, eps_d, sigma_inf, V, eps, W, eta)
    eigvals, U = np.linalg.eigh(K)
    from dmft.matsubara import fermi_function
    f_matrix = U @ np.diag(fermi_function(eigvals, beta)) @ U.conj().T

    corr = gateway_correlators(mu, eps_d, sigma_inf, V, eps, W, eta, beta)
    M_g = len(eps)
    M_h = len(eta)

    for l in range(M_g):
        np.testing.assert_allclose(corr['gg'][l], f_matrix[1 + l, 1 + l], atol=1e-12)
        np.testing.assert_allclose(corr['dg'][l], f_matrix[1 + l, 0], atol=1e-12)
    for l in range(M_h):
        np.testing.assert_allclose(corr['hh'][l], f_matrix[1 + M_g + l, 1 + M_g + l], atol=1e-12)
        np.testing.assert_allclose(corr['dh'][l], f_matrix[1 + M_g + l, 0], atol=1e-12)


def test_gateway_correlators_diag_vs_matsubara(iw, params):
    """Diagonalization and Matsubara routes agree for gateway correlators."""
    p = params
    corr_diag = gateway_correlators(
        p['mu'], p['eps_d'], p['sigma_inf'], p['V'], p['eps'], p['W'], p['eta'], p['beta']
    )
    corr_matsu = gateway_correlators_from_matsubara(
        iw, p['mu'], p['eps_d'], p['sigma_inf'], p['V'], p['eps'], p['W'], p['eta'], p['beta']
    )

    np.testing.assert_allclose(corr_matsu['gg'], corr_diag['gg'], atol=3e-4)
    np.testing.assert_allclose(corr_matsu['hh'], corr_diag['hh'], atol=3e-4)
    np.testing.assert_allclose(corr_matsu['dg'], corr_diag['dg'], atol=3e-4)
    np.testing.assert_allclose(corr_matsu['dh'], corr_diag['dh'], atol=3e-4)


def test_gateway_single_bath_level_matches_2x2_schur():
    """Single-level gateway reduces to the textbook 2x2 Schur formula."""
    beta = 50.0
    iw = 1j * matsubara_frequencies(256, beta)
    mu, eps_d, sigma_inf = 1.0, 0.0, 0.3
    V = np.array([0.4 + 0.1j])
    eps = np.array([-0.25])
    W = np.array([])
    eta = np.array([])

    gf = gateway_greens_functions(iw, mu, eps_d, sigma_inf, V, eps, W, eta)
    expected = 1.0 / (iw + mu - eps_d - sigma_inf - np.abs(V[0])**2 / (iw - eps[0]))
    np.testing.assert_allclose(gf['dd'], expected, atol=1e-12)
