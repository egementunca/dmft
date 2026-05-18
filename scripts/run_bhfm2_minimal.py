#!/usr/bin/env python3
"""Run BHFM2-minimal parity studies from the internal dmft package.

Modes:
- sweep: half-filling M2 temperature sweep (prof default list)
- doping: finite-doping sweep used for Ferrero-style comparisons
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.optimize import least_squares

# Allow running directly from repo root without editable install.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dmft.bhfm2_minimal import (
    CodecMin,
    ModelParamsMin,
    init_min,
    imp1_obs,
    imp2_obs,
    make_bounds_min,
    residual_min,
)


def _parse_floats(text: str):
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",")]


def _default_seed(M, Mb, U):
    p = init_min(M, Mb, W0=0.3, V0=0.3, B0=0.1, base_mu=U / 2.0)
    if M >= 2:
        vals = np.array([-0.5, 0.5], dtype=float)
        p.eta[:2] = vals
        p.eps1[:2] = vals
        p.eps2[:2] = vals
        p.eta1[:2] = vals
        p.eta2[:2] = vals
    return p


def _load_pickle(path: Path, default):
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return default


def _save_pickle(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f)


def _solve_sweep_point(
    T: float,
    p_init,
    mp0: ModelParamsMin,
    codec: CodecMin,
    lo: np.ndarray,
    hi: np.ndarray,
    M: int,
    Mb: int,
    ckpt_dir: Path,
    target_res: float,
    max_chunks: int,
    max_nfev_per_chunk: int,
):
    mp = replace(mp0, beta=1.0 / T)
    x = codec.pack(p_init)
    x = np.clip(x, lo + 1e-8, hi - 1e-8)
    ckpt = ckpt_dir / f"M{M}_T{T:g}_ckpt.pkl"
    if ckpt.exists():
        ck = _load_pickle(ckpt, None)
        if ck is not None and ck["r"] < np.linalg.norm(residual_min(codec.unpack(x), mp, M, Mb)):
            x = ck["x"]
            print(f"    resumed from ckpt: ||r||={ck['r']:.3e}", flush=True)

    best = [np.inf, x.copy()]
    r0 = residual_min(codec.unpack(x), mp, M, Mb)
    best[0] = np.linalg.norm(r0)
    best[1] = x.copy()
    total_nfev = 0

    def fn(x_):
        r = residual_min(codec.unpack(x_), mp, M, Mb)
        rn = np.linalg.norm(r)
        if rn < best[0]:
            best[0] = rn
            best[1] = x_.copy()
            _save_pickle(ckpt, {"x": x_.copy(), "r": rn})
        return r

    if best[0] < target_res:
        print(f"    already below target_res: ||r||={best[0]:.3e}", flush=True)
        max_chunks = 0

    for chunk in range(max_chunks):
        tc = time.time()
        try:
            sol = least_squares(
                fn,
                best[1],
                method="trf",
                bounds=(lo, hi),
                ftol=1e-11,
                xtol=1e-11,
                max_nfev=max_nfev_per_chunk,
                verbose=0,
            )
        except Exception as e:
            print(f"    chunk {chunk}: exception {e}", flush=True)
            break
        total_nfev += sol.nfev
        print(
            f"    chunk {chunk}: nfev={sol.nfev}  best={best[0]:.3e}  wall={time.time()-tc:.0f}s",
            flush=True,
        )
        if best[0] < target_res:
            break

    p = codec.unpack(best[1])
    imp1 = imp1_obs(p, mp, M)
    imp2 = imp2_obs(p, mp, M, Mb)
    z = mp.z
    D_lat = (1 - z) * imp1["double_occ"] + z * imp2["double_occ_per_site"]
    n_avg = (1 - z) * imp1["dens"] + z * imp2["dens_per_site"]
    info = dict(
        T=T,
        U=mp.U,
        beta=mp.beta,
        z=mp.z,
        resnorm=best[0],
        nfev=total_nfev,
        n_avg=n_avg,
        n_imp1=imp1["dens"],
        n_imp2=imp2["dens_per_site"],
        D_imp1=imp1["double_occ"],
        D_imp2=imp2["double_occ_per_site"],
        D_lat=D_lat,
        mu=p.mu,
        eta=p.eta.tolist(),
        W=p.W.tolist(),
        eta_b=p.eta_b.tolist(),
        B_h=p.B_h.tolist(),
        eps1=p.eps1.tolist(),
        V1=p.V1.tolist(),
        eta1=p.eta1.tolist(),
        W1=p.W1.tolist(),
        eps2=p.eps2.tolist(),
        V2=p.V2.tolist(),
        eta2=p.eta2.tolist(),
        W2=p.W2.tolist(),
        eta_b2=p.eta_b2.tolist(),
        B_h2=p.B_h2.tolist(),
        eps_b=p.eps_b.tolist(),
        B_g=p.B_g.tolist(),
    )
    return p, info


def _make_bounds_doping(M, Mb, E=5.0, C=3.0, mu_lo=-5.0, mu_hi=5.0):
    fields = [
        ("eta", M, -E, E),
        ("W", M, 1e-8, C),
        ("eta_b", Mb, -E, E),
        ("B_h", Mb, 1e-8, C),
        ("eps1", M, -E, E),
        ("V1", M, 1e-8, C),
        ("eta1", M, -E, E),
        ("W1", M, 1e-8, C),
        ("eps2", M, -E, E),
        ("V2", M, 1e-8, C),
        ("eta2", M, -E, E),
        ("W2", M, 1e-8, C),
        ("eta_b2", Mb, -E, E),
        ("B_h2", Mb, 1e-8, C),
        ("eps_b", Mb, -E, E),
        ("B_g", Mb, 1e-8, C),
    ]
    lo = []
    hi = []
    for _, n, l, h in fields:
        lo.extend([l] * n)
        hi.extend([h] * n)
    lo.append(mu_lo)
    hi.append(mu_hi)
    return np.array(lo), np.array(hi)


def _residual_doping_iter(p, mp_base, M, Mb, filling_target):
    z = mp_base.z
    imp1 = imp1_obs(p, mp_base, M)
    imp2 = imp2_obs(p, mp_base, M, Mb)
    n_avg = (1 - z) * imp1["dens"] + z * imp2["dens_per_site"]
    sigma_inf_iter = mp_base.U * n_avg / 2.0
    mp = replace(mp_base, Sigma_inf=sigma_inf_iter, filling_target=filling_target)
    return residual_min(p, mp, M, Mb)


def _solve_doping_point(
    T: float,
    n_target: float,
    p_init,
    mp0: ModelParamsMin,
    codec: CodecMin,
    M: int,
    Mb: int,
    ckpt_dir: Path,
    target_res: float,
    max_chunks: int,
    max_nfev_per_chunk: int,
):
    mp_base = replace(
        mp0,
        beta=1.0 / T,
        filling_target=n_target,
        Sigma_inf=mp0.U * n_target / 2.0,
    )
    lo, hi = _make_bounds_doping(M, Mb)
    x = codec.pack(p_init)
    x = np.clip(x, lo + 1e-8, hi - 1e-8)
    ckpt = ckpt_dir / f"M{M}_doped_T{T:g}_n{n_target:g}_ckpt.pkl"
    if ckpt.exists():
        ck = _load_pickle(ckpt, None)
        if ck is not None and len(ck["x"]) == len(x):
            x = ck["x"]
            print(f"    resumed ckpt: ||r||={ck['r']:.3e}", flush=True)

    best = [np.inf, x.copy()]
    r0 = _residual_doping_iter(codec.unpack(x), mp_base, M, Mb, n_target)
    best[0] = np.linalg.norm(r0)
    best[1] = x.copy()
    total_nfev = 0

    def fn(x_):
        r = _residual_doping_iter(codec.unpack(x_), mp_base, M, Mb, n_target)
        rn = np.linalg.norm(r)
        if rn < best[0]:
            best[0] = rn
            best[1] = x_.copy()
            _save_pickle(ckpt, {"x": x_.copy(), "r": rn})
        return r

    if best[0] < target_res:
        print(f"    already converged: ||r||={best[0]:.3e}", flush=True)
    else:
        for chunk in range(max_chunks):
            tc = time.time()
            try:
                sol = least_squares(
                    fn,
                    best[1],
                    method="trf",
                    bounds=(lo, hi),
                    ftol=1e-11,
                    xtol=1e-11,
                    max_nfev=max_nfev_per_chunk,
                    verbose=0,
                )
            except Exception as e:
                print(f"    chunk {chunk}: exception {e}", flush=True)
                break
            total_nfev += sol.nfev
            print(
                f"    chunk {chunk}: nfev={sol.nfev}  best={best[0]:.3e}  wall={time.time()-tc:.0f}s",
                flush=True,
            )
            if best[0] < target_res:
                break

    p = codec.unpack(best[1])
    imp1 = imp1_obs(p, mp_base, M)
    imp2 = imp2_obs(p, mp_base, M, Mb)
    z = mp_base.z
    D_lat = (1 - z) * imp1["double_occ"] + z * imp2["double_occ_per_site"]
    n_avg = (1 - z) * imp1["dens"] + z * imp2["dens_per_site"]
    sigma_inf = mp0.U * n_avg / 2.0
    info = dict(
        T=T,
        U=mp0.U,
        n_target=n_target,
        beta=mp_base.beta,
        z=mp_base.z,
        resnorm=best[0],
        nfev=total_nfev,
        n_avg=n_avg,
        n_imp1=imp1["dens"],
        n_imp2=imp2["dens_per_site"],
        D_imp1=imp1["double_occ"],
        D_imp2=imp2["double_occ_per_site"],
        D_lat=D_lat,
        mu=p.mu,
        Sigma_inf=sigma_inf,
        eta=p.eta.tolist(),
        W=p.W.tolist(),
        eta_b=p.eta_b.tolist(),
        B_h=p.B_h.tolist(),
        eps1=p.eps1.tolist(),
        V1=p.V1.tolist(),
        eta1=p.eta1.tolist(),
        W1=p.W1.tolist(),
        eps2=p.eps2.tolist(),
        V2=p.V2.tolist(),
        eta2=p.eta2.tolist(),
        W2=p.W2.tolist(),
        eta_b2=p.eta_b2.tolist(),
        B_h2=p.B_h2.tolist(),
        eps_b=p.eps_b.tolist(),
        B_g=p.B_g.tolist(),
    )
    return p, info


def _get_warm_start(T_target: float, n_target: float, params: dict):
    if not params:
        return None
    same_n = [(T, n, p) for (T, n), p in params.items() if n == n_target]
    if same_n:
        same_n.sort(key=lambda x: abs(x[0] - T_target))
        return same_n[0][2]
    best = min(
        params.items(),
        key=lambda kv: abs(kv[0][0] - T_target) + 5 * abs(kv[0][1] - n_target),
    )
    return best[1]


def run_sweep(args):
    outdir = Path(args.outdir)
    results_path = outdir / args.results_file
    params_path = outdir / args.params_file
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    temps = (
        _parse_floats(args.temps)
        if args.temps
        else [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
    )
    mp0 = ModelParamsMin(
        U=args.U,
        t=args.t,
        eps_d=0.0,
        z=args.z,
        Sigma_inf=args.U / 2.0,
        Nk=args.Nk,
        n_moments=args.n_moments,
        filling_target=1.0,
    )
    codec = CodecMin(args.M, args.Mb)
    lo, hi = make_bounds_min(args.M, args.Mb)

    results = _load_pickle(results_path, [])
    params = _load_pickle(params_path, {})
    done_t = {round(r["T"], 6) for r in results}

    for T in temps:
        if round(T, 6) in done_t:
            print(f"T={T}: already done, skipping", flush=True)
            continue
        done = sorted(params.keys())
        if done:
            nearest = min(done, key=lambda x: abs(x - T))
            p_init = params[nearest]
            src = f"warm start from T={nearest}"
        else:
            p_init = _default_seed(args.M, args.Mb, args.U)
            src = "fresh init"
        print(f"\n===== T={T} ({src}) =====", flush=True)
        t0 = time.time()
        p, info = _solve_sweep_point(
            T=T,
            p_init=p_init,
            mp0=mp0,
            codec=codec,
            lo=lo,
            hi=hi,
            M=args.M,
            Mb=args.Mb,
            ckpt_dir=ckpt_dir,
            target_res=args.target_res,
            max_chunks=args.max_chunks,
            max_nfev_per_chunk=args.max_nfev_per_chunk,
        )
        print(
            f"  done: ||r||={info['resnorm']:.3e} D_lat={info['D_lat']:.5f} "
            f"n_avg={info['n_avg']:.6f} nfev={info['nfev']} wall={time.time()-t0:.0f}s",
            flush=True,
        )
        results.append(info)
        params[T] = p
        _save_pickle(results_path, results)
        _save_pickle(params_path, params)


def run_doping(args):
    outdir = Path(args.outdir)
    results_path = outdir / args.results_file
    params_path = outdir / args.params_file
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fillings = _parse_floats(args.fillings) if args.fillings else [0.95, 0.90, 0.85]
    temps = _parse_floats(args.temps) if args.temps else [0.5, 0.3, 0.2, 0.1, 0.05]

    mp0 = ModelParamsMin(
        U=args.U,
        t=args.t,
        eps_d=0.0,
        z=args.z,
        Sigma_inf=args.U / 2.0,
        Nk=args.Nk,
        n_moments=args.n_moments,
        filling_target=1.0,
    )
    codec = CodecMin(args.M, args.Mb)

    results = _load_pickle(results_path, [])
    params = _load_pickle(params_path, {})
    done = {(round(r["T"], 6), round(r["n_target"], 6)) for r in results}

    for n in fillings:
        for T in temps:
            key = (round(T, 6), round(n, 6))
            if key in done:
                print(f"n={n} T={T}: already done, skipping", flush=True)
                continue
            print(f"\n===== n={n}, T={T} =====", flush=True)
            p_init = _get_warm_start(T, n, params)
            if p_init is None:
                p_init = _default_seed(args.M, args.Mb, args.U)
                print("  fresh init", flush=True)
            else:
                src = next((k for k, v in params.items() if v is p_init), None)
                if src is not None:
                    print(f"  warm start from T,n={src}", flush=True)
            t0 = time.time()
            p, info = _solve_doping_point(
                T=T,
                n_target=n,
                p_init=p_init,
                mp0=mp0,
                codec=codec,
                M=args.M,
                Mb=args.Mb,
                ckpt_dir=ckpt_dir,
                target_res=args.target_res,
                max_chunks=args.max_chunks,
                max_nfev_per_chunk=args.max_nfev_per_chunk,
            )
            print(
                f"  done: ||r||={info['resnorm']:.3e} D_lat={info['D_lat']:.5f} "
                f"n_avg={info['n_avg']:.5f} mu={info['mu']:+.4f} "
                f"Sigma_inf={info['Sigma_inf']:+.4f} wall={time.time()-t0:.0f}s",
                flush=True,
            )
            results.append(info)
            params[(T, n)] = p
            _save_pickle(results_path, results)
            _save_pickle(params_path, params)


def run_check(args):
    M, Mb = args.M, args.Mb
    mp = ModelParamsMin(
        U=args.U,
        t=args.t,
        beta=1.0 / args.check_T,
        Nk=args.Nk,
        n_moments=args.n_moments,
        Sigma_inf=args.U / 2.0,
        filling_target=1.0,
        z=args.z,
    )
    p = _default_seed(M, Mb, args.U)
    r = residual_min(p, mp, M, Mb)
    print(f"check residual size={len(r)} ||r||={np.linalg.norm(r):.3e}", flush=True)
    imp2 = imp2_obs(p, mp, M, Mb)
    print(
        f"check imp2 dens_per_site={imp2['dens_per_site']:.6f} "
        f"D={imp2['double_occ_per_site']:.6f}",
        flush=True,
    )


def make_parser():
    p = argparse.ArgumentParser(description="Internal BHFM2 minimal runner")
    p.add_argument("--mode", choices=["sweep", "doping", "check"], default="sweep")
    p.add_argument("--M", type=int, default=2)
    p.add_argument("--Mb", type=int, default=1)
    p.add_argument("--U", type=float, default=1.3)
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument(
        "--z",
        type=float,
        default=0.5,
        help="BHFM2 mixing weight between 1-site and 2-site impurity sectors",
    )
    p.add_argument("--Nk", type=int, default=16)
    p.add_argument("--n-moments", type=int, default=8)
    p.add_argument("--temps", type=str, default="")
    p.add_argument("--fillings", type=str, default="")
    p.add_argument("--target-res", type=float, default=1e-3)
    p.add_argument("--max-chunks", type=int, default=15)
    p.add_argument("--max-nfev-per-chunk", type=int, default=15)
    p.add_argument("--outdir", type=str, default="results/bhfm2_minimal")
    p.add_argument("--results-file", type=str, default="results.pkl")
    p.add_argument("--params-file", type=str, default="params.pkl")
    p.add_argument("--check-T", type=float, default=2.5)
    return p


def main():
    args = make_parser().parse_args()
    if args.mode == "sweep":
        run_sweep(args)
        return
    if args.mode == "doping":
        if abs(args.U - 1.3) < 1e-12:
            print("note: default doping studies in prof code use U=1.25", flush=True)
        run_doping(args)
        return
    run_check(args)


if __name__ == "__main__":
    main()
