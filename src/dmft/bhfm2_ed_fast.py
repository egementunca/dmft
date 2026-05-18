"""BHFM2 minimal ED helper kernels for sector Hamiltonians and observables.

Parity-first internal port of `BHFM2/ed_fast.py`.

Notes
-----
- Uses Numba when available.
- Falls back to pure-Python execution if Numba is unavailable.
- Basis encoding matches BHFM2: state = (up_bits << N_orb) | dn_bits.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    def njit(*_args, **_kwargs):
        def deco(fn):
            return fn
        return deco


@njit(cache=True)
def _popcount(x):
    """Hamming weight of an integer."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=True)
def _sign(state, target):
    """+1 if even number of set bits below `target`, else -1."""
    mask = (1 << target) - 1
    if _popcount(state & mask) % 2 == 0:
        return 1.0
    return -1.0


@njit(cache=True)
def _lookup(key, keys, vals):
    """Binary search: return vals[i] where keys[i] == key, or -1."""
    lo = 0
    hi = len(keys) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if keys[mid] == key:
            return vals[mid]
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def make_lookup(basis):
    """Build sorted (keys, vals) arrays for basis-value -> row lookup."""
    order = np.argsort(basis)
    keys = basis[order].astype(np.int64)
    vals = order.astype(np.int64)
    return keys, vals


@njit(cache=True)
def build_H_sector_fast(h1, U, U_orbs, N_orb, basis, idx_keys, idx_vals):
    """Dense Hamiltonian on a fixed (N_up, N_dn) sector."""
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=np.float64)
    Nmask = (1 << N_orb) - 1

    for i in range(dim):
        s = basis[i]
        up = (s >> N_orb) & Nmask
        dn = s & Nmask

        # One-body part
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

    # Symmetrize for numerical robustness
    for i in range(dim):
        for j in range(i + 1, dim):
            avg = 0.5 * (H[i, j] + H[j, i])
            H[i, j] = avg
            H[j, i] = avg

    return H


@njit(cache=True)
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
        if cnt:
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
    """Sum over both spins of <psi| c_a^dag c_b |psi>."""
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

        # spin up
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
