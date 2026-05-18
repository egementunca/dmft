#!/usr/bin/env python3
"""M=2 T-sweep runner for minimal (no anti-bond) at U=1.3, half filling.
All file paths relative to this script's directory."""
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

U = 1.3
M, Mb = 2, 1
RESULTS = os.path.join(_HERE, 'Tsweep_M2_U1.3.pkl')
PARAMS = os.path.join(_HERE, 'Tsweep_M2_U1.3_params.pkl')
mp0 = ModelParamsMin(U=U, t=0.5, eps_d=0.0, z=0.5,
                     Sigma_inf=U/2, Nk=16, n_moments=8,
                     filling_target=1.0)
codec = CodecMin(M, Mb)
lo, hi = make_bounds_min(M, Mb)


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


def solve_T(T, p_init, target_res=1e-3, max_chunks=15, max_nfev_per_chunk=15):
    mp = replace(mp0, beta=1.0/T)
    x = codec.pack(p_init); x = np.clip(x, lo+1e-8, hi-1e-8)
    ckpt = os.path.join(_HERE, f'M2_T{T}_ckpt.pkl')
    if os.path.exists(ckpt):
        with open(ckpt,'rb') as f: ck = pickle.load(f)
        if ck['r'] < np.linalg.norm(residual_min(codec.unpack(x), mp, M, Mb)):
            x = ck['x']
            print(f'    resumed from ckpt: ||r||={ck["r"]:.3e}')
    best = [np.inf, x.copy()]
    r0 = residual_min(codec.unpack(x), mp, M, Mb)
    best[0] = np.linalg.norm(r0); best[1] = x.copy()
    total_nfev = 0
    def fn(x_):
        r = residual_min(codec.unpack(x_), mp, M, Mb)
        rn = np.linalg.norm(r)
        if rn < best[0]:
            best[0] = rn; best[1] = x_.copy()
            with open(ckpt,'wb') as fp:
                pickle.dump({'x':x_.copy(),'r':rn},fp)
        return r
    if best[0] < target_res:
        print(f'    already below target_res: ||r||={best[0]:.3e}')
        max_chunks = 0

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
        print(f'    chunk {chunk}: nfev={sol.nfev}  best={best[0]:.3e}  wall={time.time()-tc:.0f}s', flush=True)
        if best[0] < target_res:
            break

    p = codec.unpack(best[1])
    imp1 = imp1_obs(p, mp, M)
    imp2 = imp2_obs(p, mp, M, Mb)
    z = mp.z
    D_lat = (1-z)*imp1['double_occ'] + z*imp2['double_occ_per_site']
    n_avg = (1-z)*imp1['dens'] + z*imp2['dens_per_site']
    info = dict(T=T, U=U, beta=mp.beta, resnorm=best[0], nfev=total_nfev,
                 n_avg=n_avg, n_imp1=imp1['dens'], n_imp2=imp2['dens_per_site'],
                 D_imp1=imp1['double_occ'], D_imp2=imp2['double_occ_per_site'],
                 D_lat=D_lat, mu=p.mu,
                 eta=p.eta.tolist(), W=p.W.tolist(),
                 eta_b=p.eta_b.tolist(), B_h=p.B_h.tolist(),
                 eps1=p.eps1.tolist(), V1=p.V1.tolist(),
                 eta1=p.eta1.tolist(), W1=p.W1.tolist(),
                 eps2=p.eps2.tolist(), V2=p.V2.tolist(),
                 eta2=p.eta2.tolist(), W2=p.W2.tolist(),
                 eta_b2=p.eta_b2.tolist(), B_h2=p.B_h2.tolist(),
                 eps_b=p.eps_b.tolist(), B_g=p.B_g.tolist())
    return p, info


def main():
    T_list = [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
    results, params = load_all()
    done_T = set(round(r['T'], 3) for r in results)

    prev_ckpt = os.path.join(_HERE, 'M2_min_T1_best.pkl')
    if os.path.exists(prev_ckpt) and 1.0 not in params:
        with open(prev_ckpt,'rb') as f: ck = pickle.load(f)
        params[1.0] = codec.unpack(ck['x'])
        save_all(results, params)

    for T in T_list:
        if round(T, 3) in done_T:
            print(f'T={T}: already done, skipping')
            continue
        done = sorted(params.keys())
        if done:
            nearest = min(done, key=lambda x: abs(x - T))
            p_init = params[nearest]
            src = f'warm start from T={nearest}'
        else:
            p_init = init_min(M, Mb, W0=0.3, V0=0.3, B0=0.1, base_mu=U/2)
            p_init.eta = np.array([-0.5, 0.5])
            p_init.eps1 = np.array([-0.5, 0.5])
            p_init.eps2 = np.array([-0.5, 0.5])
            p_init.eta1 = np.array([-0.5, 0.5])
            p_init.eta2 = np.array([-0.5, 0.5])
            src = 'fresh init'
        print(f'\n===== T={T} ({src}) =====', flush=True)
        t0 = time.time()
        p, info = solve_T(T, p_init)
        print(f'  T={T} done: ||r||={info["resnorm"]:.3e} D_lat={info["D_lat"]:.5f} '
              f'n_avg={info["n_avg"]:.6f} nfev={info["nfev"]} wall={time.time()-t0:.0f}s')
        results.append(info)
        params[T] = p
        save_all(results, params)

if __name__ == '__main__':
    main()
