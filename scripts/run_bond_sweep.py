#!/usr/bin/env python3
"""CLI wrapper for the bond-scheme Ghost-DMFT temperature sweep.

Usage:
  python3 scripts/run_bond_sweep.py --M 2 --U 1.3 --t 0.5 --mode both
  python3 scripts/run_bond_sweep.py --M 1 --U 1.3 --mode ss
"""

import argparse
import sys

from dmft.bond import run_temperature_sweep, save_results
from dmft.bond_ed import _init_gpu


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
    p.add_argument('--no-gpu', action='store_true',
                   help='disable GPU, force CPU (default: use GPU if available)')
    return p.parse_args()


def main():
    args = parse_args()

    gpu_active = _init_gpu(not args.no_gpu)
    if gpu_active:
        import cupy as cp
        dev = cp.cuda.Device(0)
        print(f'GPU: {cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()}')
    else:
        print('GPU: disabled -- running on CPU')
    sys.stdout.flush()

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
