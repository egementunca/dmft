"""Dimer ghost-DMFT lattice k-sum (vectorized).

Square lattice with dimer unit cell (sites A, B).
Orbital layout per k-point: dA=0, dB=1, hA_0..hA_{M-1}, hB_0..hB_{M-1}.
Matrix size per k: 2 + 2M.
"""

import numpy as np


def _fermi_lat(e, beta):
    x = beta * np.asarray(e, dtype=float)
    out = np.empty_like(x)
    out[x > 500] = 0.0
    out[x < -500] = 1.0
    m = (x >= -500) & (x <= 500)
    out[m] = 1.0 / (np.exp(x[m]) + 1.0)
    return out


def dimer_square_lattice_kgrid(t_d, nk=20):
    """Square lattice k-grid.

    Returns
    -------
    eps_k : array (nk²,) — dispersion -2t_d(cos kx + cos ky)
    wk    : array (nk²,) — uniform BZ weights (sum to 1)
    """
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    eps_k = (-2.0 * t_d * (np.cos(kx) + np.cos(ky))).ravel()
    wk = np.ones(eps_k.size) / eps_k.size
    return eps_k, wk


def dimer_lattice_obs(beta, mu, Sigma_inf, t_b, M,
                      eta_h, W_h, t_h, hop, eps_k, wk):
    """Vectorized BZ sum for dimer lattice correlators.

    d-level = Sigma_inf - mu on both sites (= 0 at half-filling).
    dA gets additional eps_k dispersion.

    Parameters
    ----------
    beta, mu, Sigma_inf : float
    t_b : float — intra-dimer hopping
    M : int — ghost families per site
    eta_h, W_h : arrays (M,)
    t_h : array (M,) — inter-site h-ghost hopping (used if hop)
    hop : bool
    eps_k, wk : lattice arrays

    Returns
    -------
    dict with n_hA, d_hA, hhop (arrays of shape M), n_dimer_lat (float)
    """
    dA = 0; dB = 1
    hA = [2 + m for m in range(M)]
    hB = [2 + M + m for m in range(M)]
    sz = 2 + 2 * M
    Nk = len(eps_k)
    dlev = Sigma_inf - mu

    H = np.zeros((Nk, sz, sz))
    H[:, dA, dA] = eps_k + dlev
    H[:, dB, dB] = dlev
    H[:, dA, dB] = H[:, dB, dA] = -t_b
    for m in range(M):
        H[:, hA[m], hA[m]] = eta_h[m]
        H[:, hB[m], hB[m]] = eta_h[m]
        H[:, dA, hA[m]] = H[:, hA[m], dA] = W_h[m]
        H[:, dB, hB[m]] = H[:, hB[m], dB] = W_h[m]
        if hop:
            H[:, hA[m], hB[m]] = H[:, hB[m], hA[m]] = -t_h[m]

    e, U = np.linalg.eigh(H)
    f = _fermi_lat(e, beta)
    rho = np.einsum('kin,kn,kjn->kij', U, f, U)

    n_dimer_lat = 2.0 * float(np.dot(wk, rho[:, dA, dA] + rho[:, dB, dB]))

    n_hA = np.zeros(M); d_hA = np.zeros(M); hhop = np.zeros(M)
    for m in range(M):
        n_hA[m] = np.dot(wk, rho[:, hA[m], hA[m]])
        d_hA[m] = np.dot(wk, rho[:, dA, hA[m]])
        hhop[m] = np.dot(wk, rho[:, hA[m], hB[m]])

    return dict(n_hA=n_hA, d_hA=d_hA, hhop=hhop, n_dimer_lat=n_dimer_lat)
