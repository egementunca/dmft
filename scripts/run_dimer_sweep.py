#!/usr/bin/env python3
"""CLI for dimer ghost-DMFT temperature sweep.

Usage:
  python3 scripts/run_dimer_sweep.py --U 1.3 --M 1
  python3 scripts/run_dimer_sweep.py --U 2.0 --M 1 --n_target 1.6 --hop
"""
import argparse
from dmft.dimer import run_sweep, check_atomic_limit, check_halffill


def main():
    p = argparse.ArgumentParser(description='Dimer ghost-DMFT')
    p.add_argument('--U',        type=float, default=1.3)
    p.add_argument('--t_d',      type=float, default=0.5)
    p.add_argument('--t_b',      type=float, default=0.3)
    p.add_argument('--M',        type=int,   default=1)
    p.add_argument('--n_target', type=float, default=2.0)
    p.add_argument('--nk',       type=int,   default=20)
    p.add_argument('--nT',       type=int,   default=20)
    p.add_argument('--T_max',    type=float, default=5.0)
    p.add_argument('--T_min',    type=float, default=0.1)
    p.add_argument('--mix',      type=float, default=0.5)
    p.add_argument('--tol',      type=float, default=1e-8)
    p.add_argument('--maxiter',  type=int,   default=300)
    p.add_argument('--hop',      action='store_true')
    p.add_argument('--verbose',  action='store_true')
    p.add_argument('--no_check', action='store_true')
    args = p.parse_args()

    if not args.no_check:
        print('Sanity checks:')
        check_atomic_limit(Uval=args.U)
        if abs(args.n_target - 2.0) >= 1e-10:
            check_halffill(Uval=args.U)
        print()

    run_sweep(Uval=args.U, t_d=args.t_d, t_b=args.t_b,
              M=args.M, hop=args.hop, n_target=args.n_target,
              nk=args.nk, nT=args.nT,
              T_max=args.T_max, T_min=args.T_min,
              mix=args.mix, tol=args.tol, maxiter=args.maxiter,
              verbose=args.verbose)


if __name__ == '__main__':
    main()
