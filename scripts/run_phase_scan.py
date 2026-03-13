#!/usr/bin/env python3
"""CLI wrapper for the two-ghost phase scan adapter."""

import argparse
from pathlib import Path

import numpy as np

from dmft.phase_scan import run_phase_scan, save_scan_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Run two-ghost DMFT phase scan")
    p.add_argument("--u-min", type=float, default=2.0)
    p.add_argument("--u-max", type=float, default=3.4)
    p.add_argument("--nu", type=int, default=30)
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max", type=float, default=0.20)
    p.add_argument("--nt", type=int, default=20)
    p.add_argument("--M", type=int, default=1)
    p.add_argument("--nquad", type=int, default=120)
    p.add_argument("--n-iw", type=int, default=512, dest="n_iw")
    p.add_argument("--mix", type=float, default=0.5)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--no-compat-mode", action="store_true")
    p.add_argument("--use-branch-filters", action="store_true")
    p.add_argument("--z-metal-min", type=float, default=0.12)
    p.add_argument("--z-ins-max", type=float, default=0.08)
    p.add_argument("--v-ins-max", type=float, default=0.20)
    p.add_argument("--ins-seed-v-clip", type=float, default=None)
    p.add_argument("--coexist-docc-tol", type=float, default=1e-3)
    p.add_argument("--coexist-z-tol", type=float, default=1e-3)
    p.add_argument(
        "--outprefix",
        type=str,
        default="diagnostics/phase_scan/ghost_dmft",
        help="Output path prefix without extension",
    )
    return p.parse_args()


def main():
    args = parse_args()

    U_vals = np.linspace(args.u_min, args.u_max, args.nu)
    T_vals = np.linspace(args.t_min, args.t_max, args.nt)

    outprefix = Path(args.outprefix)
    outprefix.parent.mkdir(parents=True, exist_ok=True)

    df, boundaries = run_phase_scan(
        U_vals=U_vals,
        T_vals=T_vals,
        M=args.M,
        t=0.5,
        nquad=args.nquad,
        n_matsubara=args.n_iw,
        mix=args.mix,
        tol=args.tol,
        maxiter=args.maxiter,
        compat_mode=not args.no_compat_mode,
        use_branch_filters=args.use_branch_filters,
        z_metal_min=args.z_metal_min,
        z_ins_max=args.z_ins_max,
        v_ins_max=args.v_ins_max,
        ins_seed_v_clip=args.ins_seed_v_clip,
        coexist_docc_tol=args.coexist_docc_tol,
        coexist_z_tol=args.coexist_z_tol,
    )
    save_scan_outputs(df, boundaries, outprefix=str(outprefix))

    print("Saved:")
    print(f"  {outprefix}_phase_scan.csv")
    print(f"  {outprefix}_phase_boundaries.csv")
    print(f"  {outprefix}_D_vs_U.png")
    print(f"  {outprefix}_Z_vs_U.png")
    print(f"  {outprefix}_deltaF.png")
    print(f"  {outprefix}_coexistence.png")
    print(f"  {outprefix}_phase_boundaries.png")
    print("")
    print("Boundary summary:")
    print(boundaries.to_string(index=False))


if __name__ == "__main__":
    main()
