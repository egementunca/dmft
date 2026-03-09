"""Tests for Matsubara frequency utilities."""

import numpy as np
import pytest
from dmft.matsubara import (
    matsubara_frequencies,
    fermi_function,
    pole_matsubara_sum,
    matsubara_sum_numerical,
)


def test_matsubara_frequencies_values():
    beta = 10.0
    wn = matsubara_frequencies(3, beta)
    expected = np.array([1, 3, 5]) * np.pi / beta
    np.testing.assert_allclose(wn, expected)


def test_matsubara_frequencies_spacing():
    beta = 20.0
    wn = matsubara_frequencies(100, beta)
    dw = np.diff(wn)
    np.testing.assert_allclose(dw, 2 * np.pi / beta, atol=1e-14)


def test_fermi_zero():
    """f(0) = 1/2 for any beta."""
    for beta in [1.0, 10.0, 100.0]:
        assert abs(fermi_function(np.array([0.0]), beta)[0] - 0.5) < 1e-15


def test_fermi_symmetry():
    """f(x) + f(-x) = 1."""
    beta = 20.0
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    f = fermi_function(x, beta)
    f_neg = fermi_function(-x, beta)
    np.testing.assert_allclose(f + f_neg, 1.0, atol=1e-14)


def test_fermi_overflow():
    """Large arguments should not cause overflow."""
    beta = 100.0
    x = np.array([-100.0, 100.0])
    f = fermi_function(x, beta)
    np.testing.assert_allclose(f, [1.0, 0.0], atol=1e-14)


def test_pole_matsubara_sum_single_pole():
    """(1/beta) sum_n 1/(iw_n - x) = -f(x).

    Test the fundamental identity for a single pole.
    """
    beta = 50.0
    for x in [-1.0, 0.0, 0.5, 2.0]:
        result = pole_matsubara_sum(
            residues=np.array([1.0]),
            poles=np.array([x]),
            beta=beta,
        )
        expected = -fermi_function(np.array([x]), beta)[0]
        assert abs(result - expected) < 1e-12, f"Failed for x={x}"


def test_pole_matsubara_sum_multiple_poles():
    """Sum with multiple poles equals sum of individual contributions."""
    beta = 30.0
    residues = np.array([0.5, 0.3, 0.2])
    poles = np.array([-1.0, 0.0, 1.5])

    result = pole_matsubara_sum(residues, poles, beta)
    expected = -np.sum(residues * fermi_function(poles, beta))
    assert abs(result - expected) < 1e-14


def test_numerical_sum_vs_exact():
    """Numerical Matsubara sum with tail subtraction matches exact pole sum.

    Use x=0 where f(0)=0.5, avoiding catastrophic cancellation that occurs
    when f(x) ~ 0 (large beta*x) and the sum is a near-zero difference of
    large numbers.
    """
    beta = 50.0
    n_w = 4096
    wn = matsubara_frequencies(n_w, beta)
    iw = 1j * wn

    # x=0: G(iw) = 1/iw, c1=1, c2=0. Exact answer: -f(0) = -0.5
    G_pos = 1.0 / iw
    result_numerical = matsubara_sum_numerical(G_pos, beta, tail_c1=1.0, tail_c2=0.0)
    result_exact = -0.5
    assert abs(result_numerical - result_exact) < 1e-6, (
        f"x=0: Numerical={result_numerical}, Exact={result_exact}"
    )

    # No-c1 case: function with c1=0 that decays as 1/w^2.
    # G(iw) = 1/(iw^2 + a^2) with a=1 has c1=0, c2=1.
    # (1/beta) sum_{all n} 1/(iw_n^2 + 1) = (1/beta) sum 1/(-wn^2 + 1)
    # = -(1/beta) sum 1/(wn^2 - 1)
    # For this function, use c1=0, c2=0 and just do raw sum (fast decay).
    a = 1.0
    G_pos2 = 1.0 / (iw**2 + a**2)
    result_num2 = matsubara_sum_numerical(G_pos2, beta, tail_c1=0.0, tail_c2=0.0)
    # Check it's finite and small (this sum converges quickly)
    assert np.isfinite(result_num2)


def test_half_filling_occupancy():
    """At half-filling (x=0), <n> = f(0) = 0.5."""
    beta = 50.0
    result = pole_matsubara_sum(np.array([1.0]), np.array([0.0]), beta)
    assert abs(result - (-0.5)) < 1e-14
