#!/usr/bin/env python3
"""Plot Ferrero-style pseudogap/Fermi arc diagnostics.

Usage (from the directory containing this script):
    python plot_fermi_arcs.py                        # uses default pkl name
    python plot_fermi_arcs.py path/to/results.pkl
"""
import sys
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sigma_k import (sigma_k_iwn, iwn_grid, quasiparticle_weight_k,
                      fermi_surface_k_points)


def plot_all(path=None, outdir=None):
    if path is None:
        path = os.path.join(_HERE, 'doping_M2_U1.25.pkl')
    if outdir is None:
        outdir = _HERE
    with open(path,'rb') as f: results = pickle.load(f)
    print(f'Loaded {len(results)} points from {path}')

    results.sort(key=lambda r: (r['n_target'], -r['T']))
    fillings = sorted(set(r['n_target'] for r in results))

    # Plot 1: Im Sigma vs T at AN and N, per filling
    fig, axes = plt.subplots(1, len(fillings), figsize=(5*len(fillings), 5),
                              sharey=True, constrained_layout=True)
    if len(fillings) == 1: axes = [axes]
    for ax, n in zip(axes, fillings):
        rs = [r for r in results if r['n_target'] == n]
        rs.sort(key=lambda r: r['T'])
        Ts = [r['T'] for r in rs]
        ImS_AN = []; ImS_N = []
        for r in rs:
            iwn = iwn_grid(r['beta'], 2)
            S_AN = sigma_k_iwn(np.array([[np.pi, 0.0]]), iwn, r)[0, 0]
            S_N = sigma_k_iwn(np.array([[np.pi/2, np.pi/2]]), iwn, r)[0, 0]
            ImS_AN.append(S_AN.imag)
            ImS_N.append(S_N.imag)
        ax.plot(Ts, ImS_AN, 'o-', color='tab:red', label='Antinode (π,0)')
        ax.plot(Ts, ImS_N, 's-', color='tab:blue', label='Node (π/2,π/2)')
        ax.set_xlabel('T'); ax.set_ylabel('Im Σ(k, iω_0)')
        ax.set_title(f'n = {n}')
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f'Imaginary self-energy at lowest Matsubara freq (U={results[0]["U"]})')
    plt.savefig(os.path.join(outdir, 'ferrero_ImSigma_vs_T.png'), dpi=140)
    print(f'Saved ferrero_ImSigma_vs_T.png')

    # Plot 2: Z(k) along BZ arc
    fig, ax = plt.subplots(figsize=(10, 6))
    kpath = fermi_surface_k_points(40)
    ts = np.linspace(0, 1, len(kpath))
    for n in fillings:
        rs = [r for r in results if r['n_target'] == n]
        if not rs: continue
        rs.sort(key=lambda r: r['T'])
        r = rs[0]
        Z_k = np.array([quasiparticle_weight_k(np.atleast_2d(k), r, r['beta']) for k in kpath])
        ax.plot(ts, Z_k, 'o-', label=f'n={n}, T={r["T"]}')
    ax.axvline(0.5, ls=':', color='gray', label='Node')
    ax.set_xlabel('k along FS (0=antinode, 0.5=node, 1=antinode)')
    ax.set_ylabel('Z(k)')
    ax.set_title(f'Quasiparticle weight along Fermi surface (lowest T, U={results[0]["U"]})')
    ax.grid(alpha=0.3); ax.legend()
    plt.savefig(os.path.join(outdir, 'ferrero_Z_vs_k.png'), dpi=140)
    print(f'Saved ferrero_Z_vs_k.png')

    # Plot 3: Sigma(k, iwn) vs iwn
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    r = min(results, key=lambda r: (r['T'], r['n_target']))
    iwn = iwn_grid(r['beta'], 32)
    wns = iwn.imag
    for k_lbl, k_vec in [('Antinode (π,0)', [np.pi,0]),
                          ('Nodal (π/2,π/2)', [np.pi/2, np.pi/2]),
                          ('Mid (3π/4, π/4)', [3*np.pi/4, np.pi/4])]:
        S = sigma_k_iwn(np.array([k_vec]), iwn, r)[0]
        axes[0].plot(wns, S.imag, 'o-', label=k_lbl)
        axes[1].plot(wns, S.real, 'o-', label=k_lbl)
    axes[0].set_xlabel('ω_n'); axes[0].set_ylabel('Im Σ(k, iω_n)')
    axes[1].set_xlabel('ω_n'); axes[1].set_ylabel('Re Σ(k, iω_n)')
    axes[0].set_xlim(0, max(wns))
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f'Self-energy at T={r["T"]}, n={r["n_target"]}, U={r["U"]}')
    plt.savefig(os.path.join(outdir, 'ferrero_Sigma_vs_wn.png'), dpi=140)
    print(f'Saved ferrero_Sigma_vs_wn.png')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    plot_all(path)
