#!/usr/bin/env python3
"""Finite-doping sweep for Ferrero et al comparison.

Parameters:
  U/t = 2.5 -> U = 1.25 (with t = 0.5)
  Fillings n = 0.85, 0.90, 0.95 (hole doping 15%, 10%, 5%)
  T = 0.05 to 0.5

All paths relative to script directory.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pickle, time
import numpy as np
from dataclasses import replace
from scipy.optimize import least_squares
from solve_min import (ModelParamsMin, CodecMin, init_min, SParMin,
                        residual_min, make_bounds_min,
                        imp1_obs, imp2_obs)


# ============ PARAMS ============
U = 1.25
t = 0.5
M, Mb = 2, 1
FILLINGS = [0.95, 0.90, 0.85]
TEMPS    = [0.5, 0.3, 0.2, 0.1, 0.05]

RESULTS = os.path.join(_HERE, 'doping_M2_U1.25.pkl')
PARAMS = os.path.join(_HERE, 'doping_M2_U1.25_params.pkl')

mp0 = ModelParamsMin(U=U, t=t, eps_d=0.0, z=0.5,
                     Sigma_inf=U/2, Nk=16, n_moments=8,
                     filling_target=1.0)
codec = CodecMin(M, Mb)


def make_bounds_doping(M, Mb, E=5.0, C=3.0, mu_lo=-5.0, mu_hi=5.0):
    fields = [
        ('eta', M, -E, E), ('W', M, 1e-8, C),
        ('eta_b', Mb, -E, E), ('B_h', Mb, 1e-8, C),
        ('eps1', M, -E, E), ('V1', M, 1e-8, C),
        ('eta1', M, -E, E), ('W1', M, 1e-8, C),
        ('eps2', M, -E, E), ('V2', M, 1e-8, C),
        ('eta2', M, -E, E), ('W2', M, 1e-8, C),
        ('eta_b2', Mb, -E, E), ('B_h2', Mb, 1e-8, C),
        ('eps_b', Mb, -E, E), ('B_g', Mb, 1e-8, C),
    ]
    lo = []; hi = []
    for _, n, l, h in fields:
        lo.extend([l]*n); hi.extend([h]*n)
    lo.append(mu_lo); hi.append(mu_hi)
    return np.array(lo), np.array(hi)


def residual_doping_iter(p, mp_base, M, Mb, filling_target):
    z = mp_base.z
    imp1 = imp1_obs(p, mp_base, M)
    imp2 = imp2_obs(p, mp_base, M, Mb)
    n_avg = (1-z)*imp1['dens'] + z*imp2['dens_per_site']
    Sigma_inf_iter = mp_base.U * n_avg / 2.0
    mp = replace(mp_base, Sigma_inf=Sigma_inf_iter, filling_target=filling_target)
    return residual_min(p, mp, M, Mb)


def load_all():
    results = []; params = {}
    if os.path.exists(RESULTS):
        with open(RESULTS,'rb') as f: results = pickle.load(f)
    if os.path.exists(PARAMS):
        with open(PARAMS,'rb') as f: params = pickle.load(f)
    return results, params


def save_all(results, params):
    with open(RESULTS,'wb') as f: pickle.dump(results, f)
    with open(PARAMS,'wb') as f: pickle.dump(params, f)


def solve_point(T, n_target, p_init, target_res=1e-3,
                 max_chunks=10, max_nfev_per_chunk=15):
    mp_base = replace(mp0, beta=1.0/T, filling_target=n_target,
                       Sigma_inf=U*n_target/2.0)
    lo, hi = make_bounds_doping(M, Mb)
    x = codec.pack(p_init); x = np.clip(x, lo+1e-8, hi-1e-8)

    ckpt = os.path.join(_HERE, f'M2_doped_T{T}_n{n_target}_ckpt.pkl')
    if os.path.exists(ckpt):
        with open(ckpt,'rb') as f: ck = pickle.load(f)
        if len(ck['x']) == len(x):
            x = ck['x']
            print(f'    resumed ckpt: ||r||={ck["r"]:.3e}')

    best = [np.inf, x.copy()]
    r0 = residual_doping_iter(codec.unpack(x), mp_base, M, Mb, n_target)
    best[0] = np.linalg.norm(r0); best[1] = x.copy()
    total_nfev = 0

    def fn(x_):
        r = residual_doping_iter(codec.unpack(x_), mp_base, M, Mb, n_target)
        rn = np.linalg.norm(r)
        if rn < best[0]:
            best[0] = rn; best[1] = x_.copy()
            with open(ckpt,'wb') as fp:
                pickle.dump({'x':x_.copy(),'r':rn},fp)
        return r

    if best[0] < target_res:
        print(f'    already converged: ||r||={best[0]:.3e}')
    else:
        for chunk in range(max_chunks):
            tc = time.time()
            try:
                sol = least_squares(fn, best[1], method='trf', bounds=(lo, hi),
                                    ftol=1e-11, xtol=1e-11,
                                    max_nfev=max_nfev_per_chunk, verbose=0)
            except Exception as e:
                print(f'    chunk {chunk}: exception {e}')
                break
            total_nfev += sol.nfev
            print(f'    chunk {chunk}: nfev={sol.nfev}  best={best[0]:.3e}  '
                  f'wall={time.time()-tc:.0f}s', flush=True)
            if best[0] < target_res:
                break

    p = codec.unpack(best[1])
    imp1 = imp1_obs(p, mp_base, M)
    imp2 = imp2_obs(p, mp_base, M, Mb)
    z = mp_base.z
    D_lat = (1-z)*imp1['double_occ'] + z*imp2['double_occ_per_site']
    n_avg = (1-z)*imp1['dens'] + z*imp2['dens_per_site']
    Sigma_inf = U * n_avg / 2.0
    info = dict(T=T, U=U, n_target=n_target, beta=mp_base.beta,
                 resnorm=best[0], nfev=total_nfev,
                 n_avg=n_avg, n_imp1=imp1['dens'], n_imp2=imp2['dens_per_site'],
                 D_imp1=imp1['double_occ'], D_imp2=imp2['double_occ_per_site'],
                 D_lat=D_lat, mu=p.mu, Sigma_inf=Sigma_inf,
                 eta=p.eta.tolist(), W=p.W.tolist(),
                 eta_b=p.eta_b.tolist(), B_h=p.B_h.tolist(),
                 eps1=p.eps1.tolist(), V1=p.V1.tolist(),
                 eta1=p.eta1.tolist(), W1=p.W1.tolist(),
                 eps2=p.eps2.tolist(), V2=p.V2.tolist(),
                 eta2=p.eta2.tolist(), W2=p.W2.tolist(),
                 eta_b2=p.eta_b2.tolist(), B_h2=p.B_h2.tolist(),
                 eps_b=p.eps_b.tolist(), B_g=p.B_g.tolist())
    return p, info


def get_warm_start(T_target, n_target, params):
    if not params:
        return None
    same_n = [(T, n, p) for (T, n), p in params.items() if n == n_target]
    if same_n:
        same_n.sort(key=lambda x: abs(x[0] - T_target))
        return same_n[0][2]
    best = min(params.items(),
               key=lambda kv: abs(kv[0][0] - T_target) + 5*abs(kv[0][1] - n_target))
    return best[1]


def main():
    results, params = load_all()
    done = set((round(r['T'], 3), round(r['n_target'], 3)) for r in results)

    # Seed from half-filling M=2 T=1 if available
    half_ckpt = os.path.join(_HERE, 'M2_min_T1_best.pkl')
    if os.path.exists(half_ckpt) and not params:
        with open(half_ckpt,'rb') as f: ck = pickle.load(f)
        params[(1.0, 1.0)] = codec.unpack(ck['x'])
        print(f'Seeded warm start from half-filling T=1 ckpt (||r||={ck["r"]:.3e})')
        save_all(results, params)

    for n in FILLINGS:
        for T in TEMPS:
            key = (round(T, 3), round(n, 3))
            if key in done:
                print(f'n={n} T={T}: already done, skipping')
                continue
            print(f'\n===== n={n}, T={T} =====', flush=True)
            p_init = get_warm_start(T, n, params)
            if p_init is None:
                p_init = init_min(M, Mb, W0=0.3, V0=0.3, B0=0.1, base_mu=U/2)
                p_init.eta = np.array([-0.5, 0.5])
                p_init.eps1 = np.array([-0.5, 0.5])
                p_init.eps2 = np.array([-0.5, 0.5])
                p_init.eta1 = np.array([-0.5, 0.5])
                p_init.eta2 = np.array([-0.5, 0.5])
                print(f'  fresh init')
            else:
                src_T_n = [k for k, v in params.items() if v is p_init]
                if src_T_n:
                    print(f'  warm start from T,n = {src_T_n[0]}')
            t_start = time.time()
            p, info = solve_point(T, n, p_init)
            print(f'  done: ||r||={info["resnorm"]:.3e} D_lat={info["D_lat"]:.5f} '
                  f'n_avg={info["n_avg"]:.5f} mu={info["mu"]:+.4f} '
                  f'Sigma_inf={info["Sigma_inf"]:+.4f} '
                  f'wall={time.time()-t_start:.0f}s')
            results.append(info)
            params[(T, n)] = p
            save_all(results, params)


if __name__ == '__main__':
    main()
