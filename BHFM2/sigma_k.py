#!/usr/bin/env python3
"""Extract momentum-resolved self-energy Sigma(k, iwn) from ghost params.

Pair-manifold form:
    Sigma(k, iwn) = Sigma_infty + Sigma_loc(iwn) + gamma_k * Sigma_nn(iwn)

where gamma_k = cos(kx) + cos(ky).
"""
from __future__ import annotations
import numpy as np
import os


def sigma_k_iwn(k, iwn, params):
    """Compute Sigma(k, iwn) from ghost params (dict from results.pkl).

    k: (kx, ky) tuple or array of shape (N, 2)
    iwn: complex array of shape (M,)
    params: dict with 'eta', 'W', 'eta_b', 'B_h' arrays and 'Sigma_inf' scalar

    Returns Sigma of shape (N, M).
    """
    if isinstance(params, dict):
        eta = np.asarray(params['eta'])
        W = np.asarray(params['W'])
        eta_b = np.asarray(params['eta_b'])
        B_h = np.asarray(params['B_h'])
        Sigma_inf = params.get('Sigma_inf', 0.0)
    else:
        eta = params.eta
        W = params.W
        eta_b = params.eta_b
        B_h = params.B_h
        Sigma_inf = getattr(params, 'Sigma_inf', 0.0)

    k = np.atleast_2d(k)
    iwn = np.atleast_1d(iwn)
    kx, ky = k[:, 0], k[:, 1]
    gamma_k = np.cos(kx) + np.cos(ky)

    Sigma_loc = np.zeros(len(iwn), dtype=complex)
    for a in range(len(eta)):
        Sigma_loc += W[a]**2 / (iwn - eta[a])
    Sigma_loc = Sigma_loc + Sigma_inf

    Sigma_bond = np.zeros((len(k), len(iwn)), dtype=complex)
    for b in range(len(eta_b)):
        bond_pole = B_h[b]**2 / (iwn - eta_b[b])
        Sigma_bond += 0.5 * (2 + gamma_k)[:, None] * bond_pole[None, :]

    return Sigma_loc[None, :] + Sigma_bond


def sigma_decompose(iwn, params):
    """Return (Sigma_loc, Sigma_nn) such that Sigma(k,iwn) = Sigma_loc + gamma_k Sigma_nn."""
    if isinstance(params, dict):
        eta = np.asarray(params['eta'])
        W = np.asarray(params['W'])
        eta_b = np.asarray(params['eta_b'])
        B_h = np.asarray(params['B_h'])
        Sigma_inf = params.get('Sigma_inf', 0.0)
    else:
        eta = params.eta; W = params.W
        eta_b = params.eta_b; B_h = params.B_h
        Sigma_inf = getattr(params, 'Sigma_inf', 0.0)

    iwn = np.atleast_1d(iwn)
    Sigma_loc_eff = np.full(len(iwn), Sigma_inf, dtype=complex)
    for a in range(len(eta)):
        Sigma_loc_eff += W[a]**2 / (iwn - eta[a])
    for b in range(len(eta_b)):
        Sigma_loc_eff += B_h[b]**2 / (iwn - eta_b[b])

    Sigma_nn_eff = np.zeros(len(iwn), dtype=complex)
    for b in range(len(eta_b)):
        Sigma_nn_eff += 0.5 * B_h[b]**2 / (iwn - eta_b[b])

    return Sigma_loc_eff, Sigma_nn_eff


def iwn_grid(beta, n_max):
    n = np.arange(n_max)
    return 1j * (2*n + 1) * np.pi / beta


def quasiparticle_weight_k(k, params, beta, n_max=2):
    iwn = iwn_grid(beta, n_max=2)
    Sigma = sigma_k_iwn(k, iwn, params)
    w0 = iwn[0].imag
    ImS = Sigma[0, 0].imag
    return 1.0 / (1.0 + abs(ImS) / w0)


def fermi_surface_k_points(nk=40):
    """Arc from (pi, 0) to (0, pi), passing through (pi/2, pi/2) at midpoint."""
    ts = np.linspace(0, 1, nk)
    kx = np.pi * (1 - ts)
    ky = np.pi * ts
    return np.stack([kx, ky], axis=1)


if __name__ == '__main__':
    import pickle, sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(_HERE, 'doping_M2_U1.25.pkl')
    path = sys.argv[1] if len(sys.argv) > 1 else default
    with open(path, 'rb') as f:
        results = pickle.load(f)
    print(f'Loaded {len(results)} results from {path}')
    for r in results:
        beta = r['beta']
        iwn = iwn_grid(beta, 32)
        k_AN = np.array([[np.pi, 0.0]])
        k_N = np.array([[np.pi/2, np.pi/2]])
        k_M = np.array([[3*np.pi/4, np.pi/4]])
        S_AN = sigma_k_iwn(k_AN, iwn, r)[0, 0]
        S_N = sigma_k_iwn(k_N, iwn, r)[0, 0]
        S_M = sigma_k_iwn(k_M, iwn, r)[0, 0]
        print(f'T={r["T"]}  n={r["n_target"]}: '
              f'ImSigma(AN,w0)={S_AN.imag:+.4f}  '
              f'ImSigma(N,w0)={S_N.imag:+.4f}  '
              f'ImSigma(mid,w0)={S_M.imag:+.4f}')
        print(f'  Z_AN={quasiparticle_weight_k(k_AN, r, beta):.4f}  '
              f'Z_N={quasiparticle_weight_k(k_N, r, beta):.4f}  '
              f'Z_mid={quasiparticle_weight_k(k_M, r, beta):.4f}')
