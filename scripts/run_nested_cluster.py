#!/usr/bin/env python3
"""CLI for Ghost Nested Cluster scheme.

Usage:
  # Single temperature:
  python3 scripts/run_nested_cluster.py --M 1 --U 1.3 --T 1.0 --verbose

  # Temperature sweep:
  python3 scripts/run_nested_cluster.py --M 1 --U 1.3 --sweep \
    --nT 100 --T_max 2.0 --T_min 0.05 --mix 0.1 --maxiter 5000 --verbose
"""
import argparse
import numpy as np
from dmft.nested_cluster import solve_T, run_sweep


def main():
    p = argparse.ArgumentParser(description='Ghost Nested Cluster DMFT')
    p.add_argument('--M',        type=int,   default=1)
    p.add_argument('--U',        type=float, default=1.3)
    p.add_argument('--z',        type=float, default=4.0)
    p.add_argument('--T',        type=float, default=1.0,
                   help='Temperature for single-T mode')
    p.add_argument('--nquad',    type=int,   default=50,
                   help='k-grid size per direction (nk x nk)')
    p.add_argument('--nT',       type=int,   default=100)
    p.add_argument('--T_max',    type=float, default=2.0)
    p.add_argument('--T_min',    type=float, default=0.05)
    p.add_argument('--mix',      type=float, default=0.1)
    p.add_argument('--tol',      type=float, default=1e-9)
    p.add_argument('--maxiter',  type=int,   default=5000)
    p.add_argument('--sweep',    action='store_true',
                   help='Temperature sweep mode (default: single T)')
    p.add_argument('--verbose',  action='store_true')
    args = p.parse_args()

    M = args.M
    if M == 1:
        x0 = np.array([0.0, 0.3, 0.0, 0.3, 0.05, 0.0, 0.3, 0.05])
    else:
        x0 = np.array([0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2, 0.2,
                        0.05, 0.05, 0.0, 0.0, 0.2, 0.2, 0.05, 0.05])

    if args.sweep:
        run_sweep(Uval=args.U, z=args.z, M=M, nquad=args.nquad,
                  nT=args.nT, T_max=args.T_max, T_min=args.T_min,
                  mix=args.mix, tol=args.tol, maxiter=args.maxiter,
                  verbose=args.verbose)
    else:
        r = solve_T(args.T, x0, Uval=args.U, z=args.z, M=M,
                    nquad=args.nquad, mix=args.mix, tol=args.tol,
                    maxiter=args.maxiter, verbose=True)
        print(f'docc={r["docc"]:.8f}  docc1={r["docc1"]:.8f}  '
              f'docc2={r["docc2"]:.8f}  iters={r["iters"]}  dp={r["dp"]:.2e}')


if __name__ == '__main__':
    main()
