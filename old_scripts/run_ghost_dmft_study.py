#!/usr/bin/env python3
"""
run_ghost_dmft_study.py
========================
Master scan script for dimer ghost-DMFT study.
Runs both half-filling and finite-doping sweeps for a given M.
Identical parameter sets for M=1 and M=2 -- results directly comparable.

Full scan:
  U  = 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
  n  = 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.2, 1.0  (n_dimer, half-fill=2)
  T  = 4.0 -> 0.03,  24 points
  nk = 32

Extracts: Mott transition at half-filling, finite-doping phase boundary
          n_c(U), Fermi surface Z(k) evolution.

Usage:
  python run_ghost_dmft_study.py --M 1          # Mac, ~2-4 hours
  python run_ghost_dmft_study.py --M 2          # cluster, ~16x longer
  python run_ghost_dmft_study.py --M 1 --quick  # test ~10 min

Output (saved after every (U,n)):
  study_M{M}_halffill_U{U}.npy
  study_M{M}_doped_U{U}_n{n}.npy
  study_M{M}_summary.npy
"""

import numpy as np
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument('--M',       type=int,   default=1)
p.add_argument('--hop',     action='store_true', default=True)
p.add_argument('--quick',   action='store_true')
p.add_argument('--U_list',  type=float, nargs='+', default=None)
p.add_argument('--n_list',  type=float, nargs='+', default=None)
p.add_argument('--nk',      type=int,   default=None)
p.add_argument('--nT',      type=int,   default=None)
p.add_argument('--T_min',   type=float, default=None)
p.add_argument('--mix',     type=float, default=0.5)
p.add_argument('--tol',     type=float, default=1e-8)
p.add_argument('--maxiter', type=int,   default=600)
args = p.parse_args()

M   = args.M
hop = args.hop

# =============================================================================
# Parameters -- same for M=1 and M=2
# =============================================================================
if args.quick:
    U_list = args.U_list or [2.0, 3.0, 4.0]
    n_list = args.n_list or [2.0, 1.8, 1.6, 1.4]
    nk     = args.nk    or 20
    nT     = args.nT    or 8
    T_min  = args.T_min or 0.10
else:
    U_list = args.U_list or [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    n_list = args.n_list or [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.2, 1.0]
    nk     = args.nk    or 32
    nT     = args.nT    or 24
    T_min  = args.T_min or 0.03

T_max   = 4.0
t_d     = 0.5
t_b     = 0.3
mix     = args.mix
tol     = args.tol
maxiter = args.maxiter

# Import engines
from dimer_ghost_dmft       import run_sweep      as run_halffill
from dimer_ghost_dmft_doped import run_sweep_doped as run_doped

# =============================================================================
# Z(k) from converged M=1 parameters
# =============================================================================
def Z_scalar(W_h, eta_h):
    W = W_h[0]; e = eta_h[0]
    lam0 = (e - np.sqrt(e**2 + 4*W**2)) / 2.0
    return float(1.0 / (1.0 + W**2 / (lam0 - e)**2))

def Zk_map(W_h, eta_h, nk_map=64):
    if len(W_h) > 1:
        return None, None
    W = W_h[0]; e = eta_h[0]
    k = np.linspace(-np.pi, np.pi, nk_map, endpoint=False)
    kx,ky = np.meshgrid(k, k, indexing='ij')
    eps_k = -2.0*t_d*(np.cos(kx)+np.cos(ky))
    a = (eps_k+e)/2.; b = np.sqrt(((eps_k-e)/2.)**2+W**2)
    lam0 = a - b
    return 1./(1.+W**2/(lam0-e)**2), k

# =============================================================================
# Summary storage
# =============================================================================
summary = dict(
    M=M, U_list=U_list, n_list=n_list, T_min=T_min,
    D_lowT   = np.full((len(U_list), len(n_list)), np.nan),
    Z_lowT   = np.full((len(U_list), len(n_list)), np.nan),
    mu_lowT  = np.full((len(U_list), len(n_list)), np.nan),
    Si_lowT  = np.full((len(U_list), len(n_list)), np.nan),
    dp_lowT  = np.full((len(U_list), len(n_list)), np.nan),
    T_actual = np.full((len(U_list), len(n_list)), np.nan),
)

print(f'\n{"="*65}')
print(f'Dimer ghost-DMFT study  M={M}  hop={hop}')
print(f'U: {U_list}')
print(f'n: {n_list}  (dimer electrons, half-fill=2)')
print(f'T: {T_max} -> {T_min}  nT={nT}  nk={nk}')
print(f'{"="*65}')

# =============================================================================
# Main scan
# =============================================================================
for iU, U in enumerate(U_list):
    print(f'\n{"="*65}')
    print(f'U = {U}')
    print(f'{"="*65}')

    x_seed_doped = None

    for in_, n in enumerate(n_list):
        t0 = time.time()

        if abs(n - 2.0) < 1e-10:
            print(f'\n  n=2.0 (half-filling) U={U}')

            if M == 1:
                x0 = np.array([0.01, 0.20, 0.01, 0.01, -0.01, 0.20])
            elif M == 2:
                x0 = np.array([-0.30, 0.30,
                                 0.20, 0.20,
                                 0.01, 0.01,
                                 0.01, 0.01,
                                -0.30, 0.30,
                                 0.20, 0.20])
            else:
                sp = np.linspace(-0.4, 0.4, M)
                x0 = np.concatenate([sp, np.full(M,0.2), np.full(M,0.01),
                                      np.full(M,0.01), sp, np.full(M,0.2)])

            res = run_halffill(Uval=U, t_d=t_d, t_b=t_b, M=M, hop=hop,
                               nk=nk, nT=nT, T_max=T_max, T_min=T_min,
                               mix=mix, tol=tol, maxiter=maxiter)

            fname = f'study_M{M}_halffill_U{U:.1f}.npy'
            np.save(fname, np.array(res, dtype=object))

            rl = res[-1]
            Z  = Z_scalar(rl['W_h'], rl['eta_h'])
            mu_val = U/2.0;  Si_val = U/2.0

            x_seed_doped = np.concatenate([rl['x'], [U/2., U/2.]])

        else:
            print(f'\n  n={n:.2f}  U={U}')

            x0_doped = x_seed_doped if x_seed_doped is not None else None
            if x0_doped is None:
                if M == 1:
                    x0_doped = np.array([0.01,0.20,0.01,0.01,-0.01,0.20,
                                         U/2., U/2.])
                else:
                    sp = np.linspace(-0.4,0.4,M)
                    x0_doped = np.concatenate([sp,np.full(M,0.2),
                                               np.full(M,0.01),np.full(M,0.01),
                                               sp,np.full(M,0.2),[U/2.,U/2.]])

            res = run_doped(n_target=n, Uval=U, t_d=t_d, t_b=t_b,
                            M=M, hop=hop, nk=nk, nT=nT,
                            T_max=T_max, T_min=T_min,
                            mix=mix, tol=tol, maxiter=maxiter,
                            x0=x0_doped)

            fname = f'study_M{M}_doped_U{U:.1f}_n{n:.2f}.npy'
            np.save(fname, np.array(res, dtype=object))

            rl = res[-1]
            Z  = Z_scalar(rl['W_h'], rl['eta_h'])
            mu_val = rl['mu'];  Si_val = rl['Sigma_inf']

            x_seed_doped = rl['x'].copy()

        summary['D_lowT'] [iU, in_] = rl['docc']
        summary['Z_lowT'] [iU, in_] = Z
        summary['mu_lowT'][iU, in_] = mu_val
        summary['Si_lowT'][iU, in_] = Si_val
        summary['dp_lowT'][iU, in_] = rl['dp']
        summary['T_actual'][iU, in_]= rl['T']

        np.save(f'study_M{M}_summary.npy', summary)

        print(f'  --> SAVED {fname}')
        print(f'      D={rl["docc"]:.5f}  Z={Z:.4f}'
              f'  mu={mu_val:.4f}  Sinf={Si_val:.4f}'
              f'  dp={rl["dp"]:.1e}  t={time.time()-t0:.1f}s')

print(f'\n{"="*65}')
print(f'COMPLETE. Summary saved: study_M{M}_summary.npy')
print(f'  D(U,n) and Z(U,n) at T={T_min} for phase diagram.')
print(f'{"="*65}')
