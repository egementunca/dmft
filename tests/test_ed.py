"""Tests for the Exact Diagonalization impurity solver."""

import numpy as np
import warnings
import pytest
from dmft.matsubara import matsubara_frequencies
from dmft.solvers.ed import EDSolver
from dmft.greens_function import hybridization

warnings.filterwarnings('ignore')


@pytest.fixture
def iw():
    return 1j * matsubara_frequencies(256, 50.0)


@pytest.fixture
def solver():
    return EDSolver()


def test_ed_noninteracting(iw, solver):
    """At U=0, ED gives the exact non-interacting Green's function."""
    V = np.array([0.3, 0.5])
    eps = np.array([-0.4, 0.4])
    result = solver.solve(iw, 0.0, 0.0, 0.0, V, eps, 50.0, 0.0)

    delta = hybridization(iw, V, eps)
    G_exact = 1.0 / (iw - delta)
    np.testing.assert_allclose(result['G_imp'], G_exact, atol=1e-12)


def test_ed_atomic_limit(iw, solver):
    """At V=0 (atomic limit), G = 0.5/(iw+U/2) + 0.5/(iw-U/2)."""
    U = 2.0
    V = np.array([0.0])
    eps = np.array([0.0])
    result = solver.solve(iw, U / 2, 0.0, U, V, eps, 50.0, U / 2)

    G_atom = 0.5 / (iw + U / 2) + 0.5 / (iw - U / 2)
    np.testing.assert_allclose(result['G_imp'], G_atom, atol=1e-12)
    assert abs(result['n_imp'] - 0.5) < 1e-10


def test_ed_half_filling(iw, solver):
    """PH-symmetric setup gives n = 0.5 and Sigma_inf = U/2."""
    U = 2.0
    V = np.array([0.4, 0.4])
    eps = np.array([-0.4, 0.4])
    result = solver.solve(iw, U / 2, 0.0, U, V, eps, 50.0, U / 2)

    assert abs(result['n_imp'] - 0.5) < 1e-8
    # High-frequency self-energy should be U/2
    np.testing.assert_allclose(result['Sigma_imp'][-1].real, U / 2, atol=0.01)


def test_ed_causality(iw, solver):
    """Im G(iw) < 0 for positive Matsubara frequencies."""
    V = np.array([0.4, 0.4])
    eps = np.array([-0.4, 0.4])
    result = solver.solve(iw, 1.0, 0.0, 2.0, V, eps, 50.0, 1.0)

    assert np.all(result['G_imp'].imag < 0), "G should be causal"


def test_ed_noninteracting_complex_v(iw, solver):
    """ED supports complex bath hybridization amplitudes."""
    V = np.array([0.3 + 0.2j, -0.25 + 0.1j])
    eps = np.array([-0.4, 0.4])
    result = solver.solve(iw, 0.0, 0.0, 0.0, V, eps, 50.0, 0.0)

    delta = hybridization(iw, V, eps)
    G_exact = 1.0 / (iw - delta)
    np.testing.assert_allclose(result['G_imp'], G_exact, atol=1e-12)
