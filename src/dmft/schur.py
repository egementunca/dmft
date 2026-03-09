"""Schur complement routines for block matrix inversion.

For block matrix M = [[A, B], [C, D]] with D invertible:
    (M^{-1})_{11} = (A - B D^{-1} C)^{-1}  (Schur complement of D)

Physics: "integrating out" the D-sector gives an effective inverse
propagator for the A-sector.

In our application, D is always diagonal (uncoupled bath/ghost levels),
so D^{-1} is trivially element-wise reciprocal.
"""

import numpy as np


def schur_complement(A, B, C, D):
    """General Schur complement S = A - B @ D^{-1} @ C.

    Parameters
    ----------
    A : ndarray, shape (p, p)
    B : ndarray, shape (p, q)
    C : ndarray, shape (q, p)
    D : ndarray, shape (q, q)

    Returns
    -------
    ndarray, shape (p, p)
        The Schur complement of D in the block matrix.
    """
    D_inv = np.linalg.inv(D)
    return A - B @ D_inv @ C


def schur_complement_diag(A, B, C, D_diag):
    """Schur complement when D is diagonal (our standard case).

    S = A - sum_l B[l] * C[l] / D_diag[l]

    Parameters
    ----------
    A : scalar or ndarray
        The (1,1) block.
    B : ndarray, shape (M,)
        Row coupling vector (d -> bath/ghost).
    C : ndarray, shape (M,)
        Column coupling vector (bath/ghost -> d).
    D_diag : ndarray, shape (M,)
        Diagonal entries of the D block.

    Returns
    -------
    scalar or ndarray
        The Schur complement value.
    """
    return A - np.sum(B * C / D_diag)


def block_greens_functions(A, B, C, D_diag):
    """Compute all Green's function blocks from the inverse resolvent.

    Given the inverse Green's function in block form:
        G^{-1} = [[A, -B], [-C, D]]
    where the minus signs on B, C are conventional (hybridization enters
    as -V in the off-diagonal blocks of iw + mu - H).

    Actually, for our Hamiltonians:
        (iw + mu - H) = [[iw + mu - eps_d - sigma_inf, -V^*], [-V, iw - eps]]
    Wait — the inverse resolvent is (iw + mu - H), and the Green's function
    blocks follow from inverting this. Let's use the Schur complement directly.

    For inverse resolvent M = [[A, B_od], [C_od, D_diag]], the full inverse is:
        G_dd = S^{-1}   where S = A - B_od @ diag(1/D_diag) @ C_od
        G_dl = -S^{-1} * B_od[l] / D_diag[l]
        G_ld = -C_od[l] / D_diag[l] * S^{-1}
        G_ll = 1/D_diag[l] + C_od[l] * B_od[l] / D_diag[l]^2 * S^{-1}

    Parameters
    ----------
    A : complex scalar
        Diagonal element of the d-orbital block: iw + mu - eps_d - sigma_inf.
    B : ndarray, shape (M,)
        Off-diagonal row: typically -V_l^* (conjugate of hybridization).
    C : ndarray, shape (M,)
        Off-diagonal column: typically -V_l.
    D_diag : ndarray, shape (M,)
        Diagonal of the bath/ghost block: iw - eps_l or iw - eta_l.

    Returns
    -------
    dict with keys:
        'dd': complex scalar — G_{dd}
        'dl': ndarray, shape (M,) — G_{d,l}
        'ld': ndarray, shape (M,) — G_{l,d}
        'll': ndarray, shape (M,) — diagonal G_{l,l}
    """
    D_inv = 1.0 / D_diag
    S = A - np.sum(B * C * D_inv)
    S_inv = 1.0 / S

    G_dd = S_inv
    G_dl = -S_inv * B * D_inv
    G_ld = -D_inv * C * S_inv
    G_ll = D_inv + (D_inv * C) * S_inv * (B * D_inv)

    return {'dd': G_dd, 'dl': G_dl, 'ld': G_ld, 'll': G_ll}
