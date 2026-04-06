#!/usr/bin/env python3
"""CLI for dimer ghost-DMFT (U, n) phase diagram study.

Drop-in replacement for professor's run_ghost_dmft_study.py
using the optimized internal codebase (sector-blocked ED).

Usage:
  python3 scripts/run_dimer_study.py --M 1          # fast baseline
  python3 scripts/run_dimer_study.py --M 2           # cluster
  python3 scripts/run_dimer_study.py --M 1 --quick   # test
"""
import argparse
from dmft.dimer import run_study


def main():
    p = argparse.ArgumentParser(description='Dimer ghost-DMFT study')
    p.add_argument('--M',       type=int,   default=1)
    p.add_argument('--hop',     action='store_true', default=True)
    p.add_argument('--quick',   action='store_true')
    p.add_argument('--U_list',  type=float, nargs='+', default=None)
    p.add_argument('--n_list',  type=float, nargs='+', default=None)
    p.add_argument('--nk',      type=int,   default=32)
    p.add_argument('--nT',      type=int,   default=24)
    p.add_argument('--T_min',   type=float, default=0.03)
    p.add_argument('--T_max',   type=float, default=4.0)
    p.add_argument('--mix',     type=float, default=0.5)
    p.add_argument('--tol',     type=float, default=1e-8)
    p.add_argument('--maxiter', type=int,   default=600)
    args = p.parse_args()

    run_study(M=args.M, hop=args.hop,
              U_list=args.U_list, n_list=args.n_list,
              nk=args.nk, nT=args.nT,
              T_max=args.T_max, T_min=args.T_min,
              mix=args.mix, tol=args.tol, maxiter=args.maxiter,
              quick=args.quick)


if __name__ == '__main__':
    main()
