#!/usr/bin/env python3
"""Sparse-matrix ED for larger impurity sectors.

Builds sparse H in COO format (rows, cols, data lists) via the same
bit-manipulation logic as ed_fast.build_H_sector_fast, but storing only
nonzero entries. Then uses scipy.sparse.linalg.eigsh (ARPACK Lanczos)
to get the lowest k eigenvalues/vectors.

For imp2 at M=2, Mb=1 (minimal, N_orb=7): largest sector = 1225 dim.
For imp2 at M=2, Mb=1 (anti-bond, N_orb=8): largest sector = 4900 dim.
Dense eigh on 1225: ~0.3s; on 4900: ~7s.
Sparse eigsh with k=30: ~0.05s and ~0.3s respectively.

We keep only states with Boltzmann weight exp(-beta*(E-E0)) > tol (e.g. 1e-10)
which limits how many eigenstates we actually need.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh as dense_eigh

from ed_fast import _popcount, _sign, _lookup, make_lookup


@njit(cache=True)
def build_H_sparse_coo(h1, U, U_orbs, N_orb, basis, idx_keys, idx_vals):
    """Build H in COO format: returns (rows, cols, data) arrays.

    Preserves symmetry; off-diagonal entries are produced as-is (not symmetrized).
    The sparse matrix constructor will combine duplicate entries via .sum_duplicates().
    """
    dim = len(basis)
    Nmask = (1 << N_orb) - 1

    # Pre-count a max nnz for allocation
    # Each basis state can hybridize to at most 2 * N_orb * N_orb other states
    # (for each (a,b) one-body term, for each spin). Plus diag for U.
    # Use conservative upper bound:
    max_nnz = dim * (4 * N_orb * N_orb + 1)
    rows = np.empty(max_nnz, dtype=np.int64)
    cols = np.empty(max_nnz, dtype=np.int64)
    data = np.empty(max_nnz, dtype=np.float64)
    k = 0

    for i in range(dim):
        s = basis[i]
        up = (s >> N_orb) & Nmask
        dn = s & Nmask

        for a in range(N_orb):
            for b in range(N_orb):
                t_ab = h1[a, b]
                if t_ab == 0.0:
                    continue
                # spin up
                bits = up
                if bits & (1 << b):
                    if a == b:
                        rows[k] = i; cols[k] = i; data[k] = t_ab
                        k += 1
                    elif not (bits & (1 << a)):
                        sgn = _sign(bits, b)
                        nb = bits ^ (1 << b)
                        sgn2 = _sign(nb, a)
                        new_bits = nb | (1 << a)
                        new_s = (new_bits << N_orb) | dn
                        j = _lookup(new_s, idx_keys, idx_vals)
                        if j >= 0:
                            rows[k] = j; cols[k] = i; data[k] = t_ab * sgn * sgn2
                            k += 1
                # spin down
                bits = dn
                if bits & (1 << b):
                    if a == b:
                        rows[k] = i; cols[k] = i; data[k] = t_ab
                        k += 1
                    elif not (bits & (1 << a)):
                        sgn = _sign(bits, b)
                        nb = bits ^ (1 << b)
                        sgn2 = _sign(nb, a)
                        new_bits = nb | (1 << a)
                        new_s = (up << N_orb) | new_bits
                        j = _lookup(new_s, idx_keys, idx_vals)
                        if j >= 0:
                            rows[k] = j; cols[k] = i; data[k] = t_ab * sgn * sgn2
                            k += 1

    # Hubbard U (diagonal)
    for ci in range(len(U_orbs)):
        c = U_orbs[ci]
        cmask = 1 << c
        for i in range(dim):
            s = basis[i]
            up = (s >> N_orb) & Nmask
            dn = s & Nmask
            if (up & cmask) and (dn & cmask):
                rows[k] = i; cols[k] = i; data[k] = U
                k += 1

    return rows[:k], cols[:k], data[:k]


def build_H_sparse(h1, U, U_orbs, N_orb, basis, idx_keys, idx_vals):
    """Wrapper that returns a scipy CSR sparse Hermitian matrix.

    Symmetrizes by averaging (H + H.T)/2 so that eigsh treats it as Hermitian.
    """
    dim = len(basis)
    rows, cols, data = build_H_sparse_coo(h1, U, U_orbs, N_orb, basis, idx_keys, idx_vals)
    H = coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    H.sum_duplicates()
    # Symmetrize
    H = 0.5 * (H + H.T)
    return H


def solve_sector_lanczos(H, k_req, dim_threshold=50, tol=1e-12):
    """Solve for lowest-k eigenstates of sparse H.

    For small sectors (dim <= dim_threshold), fall back to dense eigh since
    Lanczos has overhead.

    Returns (eigvals, eigvecs).
    eigvecs has shape (dim, n_returned); n_returned = min(k_req, dim-1) for Lanczos
    or dim for dense.
    """
    dim = H.shape[0]
    if dim <= dim_threshold:
        # Dense eigh
        Hd = H.toarray()
        evals, evecs = dense_eigh(Hd)
        return evals, evecs

    k_eff = min(k_req, dim - 2)
    if k_eff < 1:
        Hd = H.toarray()
        return dense_eigh(Hd)

    try:
        # ARPACK Lanczos for lowest k_eff eigenvalues
        evals, evecs = eigsh(H, k=k_eff, which='SA', tol=tol, maxiter=2000)
        # Sort ascending
        order = np.argsort(evals)
        return evals[order], evecs[:, order]
    except Exception:
        # Fall back to dense
        Hd = H.toarray()
        return dense_eigh(Hd)


def adaptive_k(dim, beta, extra=10, max_k=None):
    """Estimate needed k (number of lowest eigenstates) for finite-T observables.

    Rough heuristic: we need all states within energy window ~1/beta of the
    ground state (plus some buffer). We typically have dim/10 states in that
    window for intermediate cluster sizes.

    Default: min(50, dim/4) which is usually safe at T~1.
    """
    k = max(extra, int(dim / 10))
    k = min(k, 80)
    if max_k is not None:
        k = min(k, max_k)
    return k


if __name__ == '__main__':
    # Quick test: small Hubbard dimer
    from ghost_dmft_bond import build_sector
    N_orb = 2
    h1 = np.array([[0.0, -0.5], [-0.5, 0.0]])
    U_orbs = np.array([0, 1], dtype=np.int64)
    U = 1.3
    # N_up=1, N_dn=1 sector (half filling)
    basis_np, _ = build_sector(N_orb, 1, 1)
    basis = basis_np.astype(np.int64)
    keys, vals = make_lookup(basis)
    H_sp = build_H_sparse(h1, U, U_orbs, N_orb, basis, keys, vals)
    print('Sparse H:')
    print(H_sp.toarray())
    evals, evecs = solve_sector_lanczos(H_sp, k_req=4)
    print('Eigenvalues:', evals)
    # Dense check
    from ed_fast import build_H_sector_fast
    Hd = build_H_sector_fast(h1, U, U_orbs, N_orb, basis, keys, vals)
    evals_d, _ = dense_eigh(Hd)
    print('Dense check:', evals_d)
