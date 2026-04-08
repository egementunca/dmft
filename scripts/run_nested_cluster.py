#!/usr/bin/env python3
"""CLI for Ghost Nested Cluster scheme.

Usage:
  python3 scripts/run_nested_cluster.py --M 1 --U 1.3
  python3 scripts/run_nested_cluster.py --M 2 --U 1.3 --nquad 50
  python3 scripts/run_nested_cluster.py --M 1 --U 1.3 --verbose
"""
import argparse
from dmft.nested_cluster import run_sweep


def main():
    p = argparse.ArgumentParser(description='Ghost Nested Cluster DMFT')
    p.add_argument('--M',        type=int,   default=1)
    p.add_argument('--U',        type=float, default=1.3)
    p.add_argument('--z',        type=float, default=4.0)
    p.add_argument('--nquad',    type=int,   default=200,
                   help='k-grid size per direction (nk×nk)')
    p.add_argument('--nT',       type=int,   default=20)
    p.add_argument('--T_max',    type=float, default=5.0)
    p.add_argument('--T_min',    type=float, default=0.1)
    p.add_argument('--mix',      type=float, default=0.4)
    p.add_argument('--tol',      type=float, default=1e-9)
    p.add_argument('--maxiter',  type=int,   default=300)
    p.add_argument('--verbose',  action='store_true')
    args = p.parse_args()

    run_sweep(Uval=args.U, z=args.z, M=args.M, nquad=args.nquad,
              nT=args.nT, T_max=args.T_max, T_min=args.T_min,
              mix=args.mix, tol=args.tol, maxiter=args.maxiter,
              verbose=args.verbose)


if __name__ == '__main__':
    main()
