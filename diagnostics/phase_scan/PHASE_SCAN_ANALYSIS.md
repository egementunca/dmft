# Two-Ghost Phase Scan Analysis (2026-03-13)

## Scope

This note summarizes the first adapter-based branch-continuation scans using:

- `src/dmft/phase_scan.py` (`GhostDMFT_M` compatibility layer)
- `dmft_loop_two_ghost(..., ghost_update_mode="correlator")`
- ED solver, M=1
- Bethe: `t=0.5`, half-filling conventions from project config

## Runs Compared

| Run | Grid | Core setting difference | Coexistence points | Stable metal | Stable insulator | Unknown |
|---|---:|---|---:|---:|---:|---:|
| `2026-03-13_coarse_v1` | 10 U × 5 T | strict branch validity filters | 0 | 16 | 0 | 34 |
| `2026-03-13_coarse_v2` | 10 U × 5 T | accept converged branches; coexistence by branch split tolerances | 12 | 18 | 27 | 5 |
| `2026-03-13_cli_smoke` | 6 U × 3 T | short smoke settings | 0 | 4 | 0 | 14 |

Raw summary file: `diagnostics/phase_scan/run_summary.csv`

## Boundary Extraction

### coarse_v1 (`runs/2026-03-13_coarse_v1/phase_boundaries.csv`)

| T | Uc1 | Uc2 | Uc_eq |
|---:|---:|---:|---:|
| 0.0500 | NaN | NaN | NaN |
| 0.0775 | NaN | NaN | NaN |
| 0.1050 | NaN | 2.3111 | NaN |
| 0.1325 | NaN | 2.9333 | NaN |
| 0.1600 | NaN | 3.2444 | NaN |

Interpretation: this setup was too restrictive to classify an insulating
branch and produced mostly `unknown` points.

### coarse_v2 (`runs/2026-03-13_coarse_v2/phase_boundaries.csv`)

| T | Uc1 | Uc2 | Uc_eq |
|---:|---:|---:|---:|
| 0.0500 | 2.0000 | 3.4000 | 2.6166 |
| 0.0775 | 2.0000 | 3.4000 | NaN |
| 0.1050 | 2.0000 | 3.0889 | NaN |
| 0.1325 | 2.0000 | 3.4000 | NaN |
| 0.1600 | 2.0000 | 3.2444 | NaN |

Interpretation: coexistence is now detected, but `Uc1=2.0` at most temperatures
hits the scanned lower bound, so this is not a reliable physical `Uc1(T)` yet.

## Branch-Separation Quality

For `coarse_v2`:

- median `|Z_metal - Z_insulator|` = `9.68e-4`
- max `|Z_metal - Z_insulator|` = `4.41e-2`
- median `|D_metal - D_insulator|` = `2.34e-4`
- max `|D_metal - D_insulator|` = `1.69e-2`

Most points show very small branch splitting; only a few points show moderate
separation. This is consistent with the observed tendency of both seeds to
flow toward similar solutions for M=1 in this implementation.

## What This Means For “Coexistence Everywhere”

The adapter is working and producing reproducible outputs, but with current
M=1 settings the branch split is often weak. Coexistence detection becomes very
sensitive to thresholds and classification logic.

So the current phase boundaries should be treated as **diagnostic**, not final.

## Recommended Next Pass

1. Increase resolution near candidate transition ranges:
   - use denser `U` around where `deltaF` changes sign and where `|ΔZ|` peaks.
2. Promote to `M=2` for physical scans:
   - M=1 is often too rigid to sustain robust metal/insulator branch splitting.
3. Separate “numerical convergence” from “phase classification”:
   - keep `conv_*` flags,
   - classify coexistence only when both branches converge and branch split is
     above explicit tolerances.
4. Track branch diagnostics in output tables:
   - add `dZ`, `dD`, and `dV` columns directly in `phase_scan.csv`.

## Repro Commands

Coarse v2 style:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
python3 scripts/run_phase_scan.py \
  --u-min 2.0 --u-max 3.4 --nu 10 \
  --t-min 0.05 --t-max 0.16 --nt 5 \
  --M 1 --n-iw 256 --maxiter 120 --mix 0.05 \
  --outprefix diagnostics/phase_scan/runs/2026-03-13_coarse_v2
```

Fast smoke:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
python3 scripts/run_phase_scan.py \
  --nu 6 --nt 3 --n-iw 128 --maxiter 60 \
  --outprefix diagnostics/phase_scan/runs/2026-03-13_cli_smoke
```
