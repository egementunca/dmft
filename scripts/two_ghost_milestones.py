#!/usr/bin/env python3
"""Run notes-faithful two-ghost DMFT milestone cases and a coarse U scan."""

import argparse
import numpy as np

from dmft.config import DMFTParams
from dmft.dmft_loop import dmft_loop_two_ghost
from dmft.solvers.ed import EDSolver


def _run_case(U: float, M: int, beta: float, n_iw: int,
              mix: float, tol: float, max_iter: int,
              h_reg: float, g_reg: float):
    p = DMFTParams.half_filling(U=U, beta=beta, n_matsubara=n_iw, M_g=M, M_h=M)
    p.mix = mix
    p.tol = tol
    p.max_iter = max_iter

    result = dmft_loop_two_ghost(
        p,
        EDSolver(),
        verbose=False,
        ghost_update_mode='correlator',
        symmetric=True,
        bath_mix=mix,
        ghost_mix=mix,
        h_reg_strength=h_reg,
        g_reg_strength=g_reg,
        convergence_metric='sigma',
    )

    last = result['history'][-1]
    converged = (
        len(result['history']) < p.max_iter
        and last['diff'] < p.tol
        and last['causality_ok']
    )
    return result, converged


def _print_iter_sample(history):
    idx = [0, 1, 2, len(history) // 2, len(history) - 3, len(history) - 2, len(history) - 1]
    seen = set()
    for i in idx:
        if i < 0 or i >= len(history) or i in seen:
            continue
        seen.add(i)
        h = history[i]
        print(
            f"    iter={h['iteration']:3d} diff={h['diff']:.3e} "
            f"h_res={h['h_resid']:.3e} g_res={h['g_resid']:.3e} "
            f"causal={h['causality_ok']} "
            f"backtrack={h['causality_backtrack_alpha']:.3f} "
            f"maxImG={h['max_imag_gloc']:.2e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=50.0)
    parser.add_argument('--n-iw', type=int, default=512)
    parser.add_argument('--mix', type=float, default=0.05)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--h-reg', type=float, default=1e-2)
    parser.add_argument('--g-reg', type=float, default=1e-2)
    args = parser.parse_args()

    print("Milestone runs (U=2, half-filling):")
    for M in (1, 2):
        result, converged = _run_case(
            U=2.0,
            M=M,
            beta=args.beta,
            n_iw=args.n_iw,
            mix=args.mix,
            tol=args.tol,
            max_iter=args.max_iter,
            h_reg=args.h_reg,
            g_reg=args.g_reg,
        )
        last = result['history'][-1]
        print(
            f"  M={M} converged={converged} iters={len(result['history'])} "
            f"diff={last['diff']:.3e} causal={last['causality_ok']} "
            f"Z={result['Z']:.4f} n={result['n_imp']:.4f} "
            f"h_res={last['h_resid']:.3e} g_res={last['g_resid']:.3e}"
        )
        _print_iter_sample(result['history'])

    print("\nCoarse U scan (M=2):")
    for U in np.arange(0.5, 4.01, 0.5):
        result, converged = _run_case(
            U=float(U),
            M=2,
            beta=args.beta,
            n_iw=args.n_iw,
            mix=args.mix,
            tol=args.tol,
            max_iter=args.max_iter,
            h_reg=args.h_reg,
            g_reg=args.g_reg,
        )
        last = result['history'][-1]
        reason = []
        if not last['causality_ok']:
            reason.append('causality')
        if last['diff'] >= args.tol:
            reason.append('diff')
        if last['h_resid'] > 1.0:
            reason.append('h_resid')
        if last['g_resid'] > 0.2:
            reason.append('g_resid')
        if not reason:
            reason.append('ok')
        print(
            f"  U={U:.1f} converged={converged} iters={len(result['history'])} "
            f"diff={last['diff']:.3e} causal={last['causality_ok']} "
            f"Z={result['Z']:.4f} h={last['h_resid']:.3e} "
            f"g={last['g_resid']:.3e} reason={','.join(reason)}"
        )


if __name__ == '__main__':
    main()
