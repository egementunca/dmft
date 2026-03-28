#!/usr/bin/env python3
"""CLI wrapper for the bond-scheme Ghost-DMFT temperature sweep.

Usage:
  python3 scripts/run_bond_sweep.py --M 2 --U 1.3 --t 0.5 --mode both
  python3 scripts/run_bond_sweep.py --M 1 --U 1.3 --mode ss
"""

import argparse
import sys

from dmft.bond import run_temperature_sweep, save_results


def parse_args():
    p = argparse.ArgumentParser(
        description='Ghost-DMFT bond scheme, square lattice, T sweep')
    p.add_argument('--M', type=int, default=1,
                   help='ghost poles (1 or 2)')
    p.add_argument('--U', type=float, default=1.3,
                   help='Hubbard U')
    p.add_argument('--t', type=float, default=0.5,
                   help='hopping t')
    p.add_argument('--mode', type=str, default='both',
                   help='ss, bond, or both')
    p.add_argument('--nk', type=int, default=30,
                   help='k-grid size per direction')
    p.add_argument('--verbose', action='store_true',
                   help='verbose bond solver iterations')
    p.add_argument('--out', type=str, default=None,
                   help='output filename (.dat)')
    return p.parse_args()


def main():
    args = parse_args()

    results, D = run_temperature_sweep(
        U=args.U, t=args.t, M=args.M,
        mode=args.mode, n_k=args.nk,
        verbose=args.verbose)

    if args.out is None:
        args.out = (f'ghost_dmft_square_M{args.M}_U{args.U}'
                    f'_t{args.t}_{args.mode}.dat')
    save_results(results, args.out, args.mode)


if __name__ == '__main__':
    main()
