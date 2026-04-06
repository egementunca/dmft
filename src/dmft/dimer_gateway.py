"""Dimer ghost-DMFT gateway (quadratic, one-body diagonalization).

Orbital layout: dA=0, dB=1,
  gA_0..gA_{M-1} = 2..M+1,   gB_0..gB_{M-1} = M+2..2M+1,
  hA_0..hA_{M-1} = 2M+2..3M+1, hB_0..hB_{M-1} = 3M+2..4M+1
Matrix size: 2 + 4M.
"""

import numpy as np
from .dimer_lattice import _fermi_lat


def dimer_gateway_obs(beta, mu, t_b, M,
                      eps_g, V_g, t_g, eta_h, W_h, t_h, hop,
                      Sigma_inf=0.0):
    """Dimer gateway correlators via one-body diagonalization.

    d-level = Sigma_inf - mu (= 0 at half-filling).

    Returns
    -------
    dict with n_gA, d_gA, ghop, n_hA, d_hA, hhop (all arrays of shape M)
    """
    dA = 0; dB = 1
    gA = [2 + m for m in range(M)]
    gB = [2 + M + m for m in range(M)]
    hA = [2 + 2 * M + m for m in range(M)]
    hB = [2 + 3 * M + m for m in range(M)]
    sz = 2 + 4 * M

    dlev = Sigma_inf - mu
    H = np.zeros((sz, sz))
    H[dA, dA] = dlev
    H[dB, dB] = dlev
    H[dA, dB] = H[dB, dA] = -t_b
    for m in range(M):
        H[gA[m], gA[m]] = eps_g[m]; H[gB[m], gB[m]] = eps_g[m]
        H[dA, gA[m]] = H[gA[m], dA] = V_g[m]
        H[dB, gB[m]] = H[gB[m], dB] = V_g[m]
        H[hA[m], hA[m]] = eta_h[m]; H[hB[m], hB[m]] = eta_h[m]
        H[dA, hA[m]] = H[hA[m], dA] = W_h[m]
        H[dB, hB[m]] = H[hB[m], dB] = W_h[m]
        if hop:
            H[gA[m], gB[m]] = H[gB[m], gA[m]] = -t_g[m]
            H[hA[m], hB[m]] = H[hB[m], hA[m]] = -t_h[m]

    e, U = np.linalg.eigh(H)
    rho = (U * _fermi_lat(e, beta)[None, :]) @ U.T

    return dict(
        n_gA=np.array([rho[gA[m], gA[m]] for m in range(M)]),
        d_gA=np.array([rho[dA, gA[m]] for m in range(M)]),
        ghop=np.array([rho[gA[m], gB[m]] for m in range(M)]),
        n_hA=np.array([rho[hA[m], hA[m]] for m in range(M)]),
        d_hA=np.array([rho[dA, hA[m]] for m in range(M)]),
        hhop=np.array([rho[hA[m], hB[m]] for m in range(M)]),
    )
