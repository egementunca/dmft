#!/usr/bin/env python3
"""Sanity check: verify all imports work and solver runs a trivial test.

Run from the handoff directory:
    python check_setup.py
"""
import sys
import os

print("=" * 60)
print("Environment check")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print()

# Check required packages
print("Required packages:")
missing = []
for pkg in ['numpy', 'scipy', 'numba', 'matplotlib']:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {pkg:12s} {version}")
    except ImportError:
        print(f"  ✗ {pkg:12s} NOT INSTALLED")
        missing.append(pkg)

if missing:
    print(f"\nMissing packages. Install with:")
    print(f"    pip install {' '.join(missing)}")
    sys.exit(1)

# Check files present
print("\nRequired source files:")
files = [
    'solve_min.py', 'ed_fast.py', 'ed_sparse.py', 'ghost_dmft_bond.py',
    'run_M2_sweep.py', 'run_M2_doping.py', 'sigma_k.py', 'plot_fermi_arcs.py',
]
for f in files:
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} MISSING")

# Check checkpoint files (optional but useful)
print("\nReference data / checkpoints (optional):")
pkls = [
    'M2_min_T1_best.pkl', 'M2_T2.5_ckpt.pkl',
    'Tsweep_U1.3.pkl', 'Tsweep_min_U1.3.pkl',
]
for f in pkls:
    status = "✓" if os.path.exists(f) else "(not present)"
    print(f"  {status} {f}")

# Import our code
print("\nImporting project modules...")
try:
    sys.path.insert(0, '.')
    from solve_min import (ModelParamsMin, CodecMin, init_min,
                            imp1_obs, imp2_obs, residual_min,
                            make_bounds_min)
    print("  ✓ solve_min")
    from ed_fast import build_H_sector_fast, make_lookup
    print("  ✓ ed_fast")
    from ghost_dmft_bond import build_sector
    print("  ✓ ghost_dmft_bond")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Run a trivial test at M=1
print("\nRunning trivial test at M=1 Mb=1, U=1.3, T=2.5, half filling...")
print("(this will take ~10-30s first time due to numba JIT compilation)")
import time
import numpy as np

mp = ModelParamsMin(U=1.3, t=0.5, beta=0.4, Nk=16, n_moments=4,
                    Sigma_inf=0.65, filling_target=1.0)
p = init_min(1, 1, W0=0.3, V0=0.3, B0=0.1, base_mu=0.65)

t0 = time.time()
r1 = imp1_obs(p, mp, 1)
t_imp1 = time.time() - t0
print(f"  imp1_obs: {t_imp1:.2f}s  (n_d={r1['dens']:.6f}  D={r1['double_occ']:.5f})")

t0 = time.time()
r2 = imp2_obs(p, mp, 1, 1)
t_imp2 = time.time() - t0
print(f"  imp2_obs: {t_imp2:.2f}s  (n_d={r2['dens_per_site']:.6f}  D={r2['double_occ_per_site']:.5f})")

t0 = time.time()
r = residual_min(p, mp, 1, 1)
t_res = time.time() - t0
print(f"  residual_min: {t_res:.2f}s  (||r||={np.linalg.norm(r):.3e}, shape={len(r)} eqs)")

# Timing at M=2 (dominant cost)
print("\nTiming M=2 Mb=1 imp2 call (dominant cost)...")
p2 = init_min(2, 1, W0=0.3, V0=0.3, B0=0.1, base_mu=0.65)
p2.eta = np.array([-0.5, 0.5])
p2.eps1 = np.array([-0.5, 0.5])
p2.eps2 = np.array([-0.5, 0.5])
p2.eta1 = np.array([-0.5, 0.5])
p2.eta2 = np.array([-0.5, 0.5])
mp2 = ModelParamsMin(U=1.3, t=0.5, beta=1.0, Nk=16, n_moments=8,
                     Sigma_inf=0.65, filling_target=1.0)
t0 = time.time()
r2_M2 = imp2_obs(p2, mp2, 2, 1)
t_imp2_M2 = time.time() - t0
print(f"  imp2_obs M=2: {t_imp2_M2:.2f}s")

print()
print("=" * 60)
print("READY" if t_imp2_M2 < 30 else "READY (but imp2 at M=2 is slow)")
print("=" * 60)
print()
print(f"Next step: run one of")
print(f"    python run_M2_sweep.py       # half filling M=2 T-sweep at U=1.3")
print(f"    python run_M2_doping.py      # Ferrero doping sweep at U=1.25")
print()
print(f"Expected cost per (T, n) point at M=2: ~{t_imp2_M2 * 50:.0f}-{t_imp2_M2 * 150:.0f}s "
      f"(~{t_imp2_M2*100/60:.0f} min)")
