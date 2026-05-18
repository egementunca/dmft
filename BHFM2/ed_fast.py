#!/usr/bin/env python3
"""Numba-accelerated observables and Hamiltonian builder for ED.

Key routines:
  - build_H_sector_fast: builds dense H matrix for a (N_up, N_dn) sector
  - expect_cdag_c_fast: vectorized <psi| c_a^dag c_b + c_b^dag c_a |psi>
                        or just <psi| c_a^dag c_b |psi> (single orbital pair)
  - expect_n_fast: vectorized occupation
  - expect_double_fast: vectorized double occupancy

All take basis as int64 arrays. Compatible with the build_sector output from
ghost_dmft_bond.py (same encoding: (up<<N_orb) | dn).
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _popcount(x):
    """Hamming weight of a 64-bit int."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=True)
def _sign(state, target):
    """+1 if even number of set bits in `state` below position `target`, else -1."""
    mask = (1 << target) - 1
    if _popcount(state & mask) % 2 == 0:
        return 1.0
    return -1.0


@njit(cache=True)
def build_H_sector_fast(h1, U, U_orbs, N_orb, basis, idx_keys, idx_vals):
    """Dense Hamiltonian on a sector.

    h1: (N_orb, N_orb) one-body matrix (real).
    U: Hubbard U.
    U_orbs: int array of orbitals with Hubbard U.
    basis: int64 array of basis states (encoded (up<<N_orb)|dn).
    idx_keys, idx_vals: for lookup basis-value -> index. See helper below.

    Returns dense (dim, dim) real array.
    """
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=np.float64)
    Nmask = (1 << N_orb) - 1

    for i in range(dim):
        s = basis[i]
        up = (s >> N_orb) & Nmask
        dn = s & Nmask

        # One-body
        for a in range(N_orb):
            for b in range(N_orb):
                t_ab = h1[a, b]
                if t_ab == 0.0:
                    continue
                # spin up
                bits = up
                if bits & (1 << b):
                    if a == b:
                        H[i, i] += t_ab
                    elif not (bits & (1 << a)):
                        sgn = _sign(bits, b)
                        nb = bits ^ (1 << b)
                        sgn2 = _sign(nb, a)
                        new_bits = nb | (1 << a)
                        new_s = (new_bits << N_orb) | dn
                        j = _lookup(new_s, idx_keys, idx_vals)
                        if j >= 0:
                            H[j, i] += t_ab * sgn * sgn2
                # spin down
                bits = dn
                if bits & (1 << b):
                    if a == b:
                        H[i, i] += t_ab
                    elif not (bits & (1 << a)):
                        sgn = _sign(bits, b)
                        nb = bits ^ (1 << b)
                        sgn2 = _sign(nb, a)
                        new_bits = nb | (1 << a)
                        new_s = (up << N_orb) | new_bits
                        j = _lookup(new_s, idx_keys, idx_vals)
                        if j >= 0:
                            H[j, i] += t_ab * sgn * sgn2

    # Hubbard U
    for ci in range(len(U_orbs)):
        c = U_orbs[ci]
        cmask = 1 << c
        for i in range(dim):
            s = basis[i]
            up = (s >> N_orb) & Nmask
            dn = s & Nmask
            if (up & cmask) and (dn & cmask):
                H[i, i] += U

    # Symmetrize
    for i in range(dim):
        for j in range(i+1, dim):
            avg = 0.5 * (H[i, j] + H[j, i])
            H[i, j] = avg
            H[j, i] = avg

    return H


@njit(cache=True)
def _lookup(key, keys, vals):
    """Binary search: return vals[i] where keys[i] == key, or -1."""
    lo = 0
    hi = len(keys) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if keys[mid] == key:
            return vals[mid]
        elif keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def make_lookup(basis):
    """Build sorted (keys, vals) arrays for lookup. basis is an int64 array."""
    order = np.argsort(basis)
    keys = basis[order].astype(np.int64)
    vals = order.astype(np.int64)
    return keys, vals


@njit(cache=True, parallel=False)
def expect_n_fast(orb, psi, basis, N_orb):
    """<psi| n_{orb,up} + n_{orb,dn} |psi>."""
    Nmask = (1 << N_orb) - 1
    mask = 1 << orb
    val = 0.0
    for i in range(len(basis)):
        s = basis[i]
        up = (s >> N_orb) & Nmask
        dn = s & Nmask
        cnt = 0
        if up & mask:
            cnt += 1
        if dn & mask:
            cnt += 1
        if cnt > 0:
            val += cnt * psi[i] * psi[i]
    return val


@njit(cache=True)
def expect_double_fast(c, psi, basis, N_orb):
    """<psi| n_{c,up} n_{c,dn} |psi>."""
    Nmask = (1 << N_orb) - 1
    cmask = 1 << c
    val = 0.0
    for i in range(len(basis)):
        s = basis[i]
        up = (s >> N_orb) & Nmask
        dn = s & Nmask
        if (up & cmask) and (dn & cmask):
            val += psi[i] * psi[i]
    return val


@njit(cache=True)
def expect_cdag_c_fast(a, b, psi, basis, idx_keys, idx_vals, N_orb):
    """Sum over both spins of <psi| c_a^dag c_b |psi>.
    Real-valued for real psi. Returns 0 if a==b (use expect_n_fast for diagonal)."""
    if a == b:
        return expect_n_fast(a, psi, basis, N_orb)

    Nmask = (1 << N_orb) - 1
    a_mask = 1 << a
    b_mask = 1 << b
    val = 0.0
    for i in range(len(basis)):
        s = basis[i]
        up = (s >> N_orb) & Nmask
        dn = s & Nmask
        # spin up: c_{a,up}^dag c_{b,up}
        bits = up
        if (bits & b_mask) and not (bits & a_mask):
            sgn = _sign(bits, b)
            nb = bits ^ b_mask
            sgn2 = _sign(nb, a)
            new_bits = nb | a_mask
            new_s = (new_bits << N_orb) | dn
            j = _lookup(new_s, idx_keys, idx_vals)
            if j >= 0:
                val += psi[j] * psi[i] * sgn * sgn2
        # spin down
        bits = dn
        if (bits & b_mask) and not (bits & a_mask):
            sgn = _sign(bits, b)
            nb = bits ^ b_mask
            sgn2 = _sign(nb, a)
            new_bits = nb | a_mask
            new_s = (up << N_orb) | new_bits
            j = _lookup(new_s, idx_keys, idx_vals)
            if j >= 0:
                val += psi[j] * psi[i] * sgn * sgn2
    return val


# ==========================================================
# Test vs dense reference
# ==========================================================

if __name__ == '__main__':
    import time
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ghost_dmft_bond import (build_sector, build_H_sector,
                                  expect_n_orb_sector, expect_double_orb,
                                  expect_cdag_c)

    N_orb = 6   # same as imp2 at M=1, Mb=1 (d1, d2, g1, g2, gbp, gbm)
    U = 1.3
    np.random.seed(42)
    h1 = np.random.randn(N_orb, N_orb)
    h1 = 0.5 * (h1 + h1.T)

    # Sector (3, 3)
    basis_np, idx_dict = build_sector(N_orb, 3, 3)
    basis = basis_np.astype(np.int64)
    idx_keys, idx_vals = make_lookup(basis)
    dim = len(basis)
    print(f'Sector (3,3): dim = {dim}')

    # Build H via both paths
    t0 = time.time()
    H_ref = build_H_sector(h1, U, [0, 1], N_orb, basis_np, idx_dict)
    H_ref = 0.5*(H_ref + H_ref.T)
    t_ref = time.time() - t0

    U_orbs_np = np.array([0, 1], dtype=np.int64)

    # Warm up numba
    _ = build_H_sector_fast(h1, U, U_orbs_np, N_orb, basis, idx_keys, idx_vals)

    t0 = time.time()
    H_fast = build_H_sector_fast(h1, U, U_orbs_np, N_orb, basis, idx_keys, idx_vals)
    t_fast = time.time() - t0

    print(f'build_H_sector:      dense={t_ref*1000:.1f}ms, fast={t_fast*1000:.1f}ms, '
          f'speedup={t_ref/t_fast:.1f}x')
    print(f'  match? {np.allclose(H_ref, H_fast)}')

    # Eigen
    from numpy.linalg import eigh
    evals, evecs = eigh(H_fast)
    psi = evecs[:, 0]   # ground state

    # Test expect_n
    for orb in range(N_orb):
        n_ref = expect_n_orb_sector(orb, psi, basis_np, N_orb)
        # warm up
        _ = expect_n_fast(orb, psi, basis, N_orb)
        n_fast = expect_n_fast(orb, psi, basis, N_orb)
        assert abs(n_ref - n_fast) < 1e-12, f'orb {orb}: {n_ref} vs {n_fast}'
    print('  expect_n: agrees')

    # Test expect_cdag_c (a != b)
    for a in [0, 2]:
        for b in [1, 3, 4, 5]:
            C_ref = expect_cdag_c(a, b, psi, basis_np, idx_dict, N_orb)
            _ = expect_cdag_c_fast(a, b, psi, basis, idx_keys, idx_vals, N_orb)
            C_fast = expect_cdag_c_fast(a, b, psi, basis, idx_keys, idx_vals, N_orb)
            assert abs(C_ref - C_fast) < 1e-12, f'{a},{b}: {C_ref} vs {C_fast}'
    print('  expect_cdag_c: agrees')

    # Benchmark: one pair, many calls
    t0 = time.time()
    for _ in range(10):
        C_ref = expect_cdag_c(0, 1, psi, basis_np, idx_dict, N_orb)
    t_ref_cc = (time.time() - t0) / 10

    t0 = time.time()
    for _ in range(100):
        C_fast = expect_cdag_c_fast(0, 1, psi, basis, idx_keys, idx_vals, N_orb)
    t_fast_cc = (time.time() - t0) / 100

    print(f'expect_cdag_c one call: dense={t_ref_cc*1000:.2f}ms, fast={t_fast_cc*1000:.2f}ms, '
          f'speedup={t_ref_cc/t_fast_cc:.0f}x')
