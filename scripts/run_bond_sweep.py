#!/usr/bin/env python3
"""CLI wrapper for the bond-scheme Ghost-DMFT temperature sweep.

Usage:
  python3 scripts/run_bond_sweep.py --M1g 1 --M2g 1 --Mbg 1 --U 1.3 --t 0.5
  python3 scripts/run_bond_sweep.py --M1g 2 --M2g 2 --Mbg 1 --nT 30 --Tmin 0.02 --Tmax 0.5
"""

import argparse
import sys

from dmft.bond import run_temperature_sweep
from dmft.bond_ed import _init_gpu


def parse_args():
    p = argparse.ArgumentParser(
        description='Ghost-DMFT bond scheme, square lattice, T sweep')
    p.add_argument('--M1g', type=int, default=1,
                   help='g1-ghost poles (single-site)')
    p.add_argument('--M2g', type=int, default=1,
                   help='g2-ghost poles (two-site local)')
    p.add_argument('--Mbg', type=int, default=1,
                   help='gb-ghost poles (two-site bond)')
    p.add_argument('--U', type=float, default=1.3,
                   help='Hubbard U')
    p.add_argument('--t', type=float, default=0.5,
                   help='hopping t')
    p.add_argument('--nk', type=int, default=20,
                   help='k-grid size per direction')
    p.add_argument('--nT', type=int, default=10,
                   help='number of temperature points')
    p.add_argument('--Tmin', type=float, default=0.1,
                   help='minimum temperature')
    p.add_argument('--Tmax', type=float, default=1.0,
                   help='maximum temperature')
    p.add_argument('--mix', type=float, default=0.5,
                   help='bond solver mixing parameter')
    p.add_argument('--tol', type=float, default=1e-7,
                   help='bond solver tolerance')
    p.add_argument('--maxiter', type=int, default=100,
                   help='bond solver max iterations')
    p.add_argument('--verbose', action='store_true',
                   help='verbose bond solver iterations')
    p.add_argument('--tag', type=str, default='',
                   help='output file tag')
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
        U=args.U, t=args.t,
        M1g=args.M1g, M2g=args.M2g, Mbg=args.Mbg,
        n_k=args.nk, nT=args.nT, Tmin=args.Tmin, Tmax=args.Tmax,
        mix_bond=args.mix, tol_bond=args.tol,
        maxiter_bond=args.maxiter,
        verbose=args.verbose, tag=args.tag)

    print(f'\nDone. {len(results)} temperature points.')


if __name__ == '__main__':
    main()
