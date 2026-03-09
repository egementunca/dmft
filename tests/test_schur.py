"""Tests for Schur complement routines."""

import numpy as np
import pytest
from dmft.schur import schur_complement, schur_complement_diag, block_greens_functions


def test_schur_2x2():
    """2x2 Schur complement: (A - B*C/D) for scalars."""
    A = 5.0
    B = np.array([2.0])
    C = np.array([3.0])
    D = np.array([[4.0]])
    S = schur_complement(np.array([[A]]), B.reshape(1, 1), C.reshape(1, 1), D)
    expected = A - B[0] * C[0] / D[0, 0]
    np.testing.assert_allclose(S[0, 0], expected)


def test_schur_diag_vs_general():
    """Diagonal-D Schur complement matches general version."""
    M = 4
    A = 3.0 + 1j
    B = np.random.randn(M) + 1j * np.random.randn(M)
    C = np.random.randn(M) + 1j * np.random.randn(M)
    D_diag = np.random.randn(M) + 1j * np.random.randn(M)

    S_diag = schur_complement_diag(A, B, C, D_diag)
    S_gen = schur_complement(
        np.array([[A]]),
        B.reshape(1, M),
        C.reshape(M, 1),
        np.diag(D_diag),
    )
    np.testing.assert_allclose(S_diag, S_gen[0, 0], atol=1e-12)


def test_schur_gives_pole_sum():
    """Schur complement of (iw + mu - H) gives hybridization pole sum.

    For H = [[eps_d, V1, V2], [V1, e1, 0], [V2, 0, e2]]:
        G_dd^{-1} = iw + mu - eps_d - |V1|^2/(iw - e1) - |V2|^2/(iw - e2)
    """
    iw = 1j * 3.14
    mu = 1.0
    eps_d = 0.0
    V = np.array([0.3, 0.5])
    e = np.array([-0.4, 0.6])

    A = iw + mu - eps_d
    B = -np.conj(V)  # off-diag row of (iw + mu - H)
    C = -V           # off-diag col
    D_diag = iw * np.ones(2) - e  # diagonal: iw - e_l (note: mu is in A, not D)

    S = schur_complement_diag(A, B, C, D_diag)
    expected = iw + mu - eps_d - np.sum(np.abs(V)**2 / (iw * np.ones(2) - e))
    np.testing.assert_allclose(S, expected, atol=1e-14)


def test_schur_vs_direct_inverse():
    """Compare Schur complement G_dd with direct matrix inversion.

    Build the full (1 + M) x (1 + M) inverse resolvent, invert it directly,
    and compare the (0,0) block with 1/S from Schur complement.
    """
    M = 3
    iw = 1j * 1.57
    mu = 0.5
    eps_d = 0.1
    V = np.array([0.2, 0.4, 0.3])
    e = np.array([-0.5, 0.0, 0.7])

    # Build full matrix (iw + mu - H)
    dim = 1 + M
    G_inv = np.zeros((dim, dim), dtype=complex)
    G_inv[0, 0] = iw + mu - eps_d
    for l in range(M):
        G_inv[0, l + 1] = -np.conj(V[l])
        G_inv[l + 1, 0] = -V[l]
        G_inv[l + 1, l + 1] = iw - e[l]

    # Direct inversion
    G_full = np.linalg.inv(G_inv)
    G_dd_direct = G_full[0, 0]

    # Schur complement
    A = iw + mu - eps_d
    B = -np.conj(V)
    C = -V
    D_diag = iw * np.ones(M) - e
    S = schur_complement_diag(A, B, C, D_diag)
    G_dd_schur = 1.0 / S

    np.testing.assert_allclose(G_dd_schur, G_dd_direct, atol=1e-12)


def test_block_greens_functions_vs_direct():
    """All Green's function blocks from Schur match direct inversion."""
    M = 3
    iw = 1j * 2.5
    mu = 0.5
    eps_d = 0.0
    V = np.array([0.3, 0.2, 0.5])
    e = np.array([-0.6, 0.0, 0.8])

    # Build full inverse resolvent
    dim = 1 + M
    G_inv = np.zeros((dim, dim), dtype=complex)
    G_inv[0, 0] = iw + mu - eps_d
    for l in range(M):
        G_inv[0, l + 1] = -np.conj(V[l])
        G_inv[l + 1, 0] = -V[l]
        G_inv[l + 1, l + 1] = iw - e[l]

    G_full = np.linalg.inv(G_inv)

    # Block GFs via our routine
    A = iw + mu - eps_d
    B = -np.conj(V)
    C = -V
    D_diag = iw * np.ones(M) - e
    blocks = block_greens_functions(A, B, C, D_diag)

    np.testing.assert_allclose(blocks['dd'], G_full[0, 0], atol=1e-12)
    for l in range(M):
        np.testing.assert_allclose(blocks['dl'][l], G_full[0, l + 1], atol=1e-12)
        np.testing.assert_allclose(blocks['ld'][l], G_full[l + 1, 0], atol=1e-12)
        np.testing.assert_allclose(blocks['ll'][l], G_full[l + 1, l + 1], atol=1e-12)
