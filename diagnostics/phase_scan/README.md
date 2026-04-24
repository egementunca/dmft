# Phase Scan Artifacts

This folder contains branch-continuation phase-scan outputs for the two-ghost
ED adapter (`src/dmft/phase_scan.py`).

## Layout

- `runs/2026-03-19_preflight/`
  - Initial compatibility preflight (`compat_mode=True`) using a small grid.
- `runs/2026-03-19_preflight_require/`
  - Initial preflight with convergence gate enabled (`--require-converged`).
- `runs/2026-03-20_m1_baseline/`
  - M=1 dense-grid baseline scan.
- `runs/2026-03-20_m2_quality/`
  - M=2 dense-grid quality scan (initial).
- `runs/2026-03-20_m2_quality_v2/`
  - M=2 dense-grid quality scan (rerun, see `PHASE_SCAN_FULL_GRID_2026-03-20.md`).
- `run_summary.csv`
  - One-row-per-run summary of counts and branch-separation metrics.
- `PHASE_SCAN_INITIAL_CHECK_2026-03-19.md`
  - Initial validation: what worked, expected behaviors, and caveats.
- `PHASE_SCAN_FULL_GRID_2026-03-20.md`
  - Full-grid scan results and branch-boundary analysis.

Each run directory has a consistent file set:

- `<prefix>_phase_scan.csv`
- `<prefix>_phase_boundaries.csv`
- `<prefix>_D_vs_U.png`
- `<prefix>_Z_vs_U.png`
- `<prefix>_deltaF.png`
- `<prefix>_coexistence.png`
- `<prefix>_phase_boundaries.png`

## Reproduce

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
python3 scripts/run_phase_scan.py \
  --nu 4 --nt 2 --n-iw 128 --maxiter 40 \
  --outprefix diagnostics/phase_scan/runs/2026-03-19_preflight/main
```

Convergence-gated preflight:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
python3 scripts/run_phase_scan.py \
  --nu 3 --nt 2 --n-iw 128 --maxiter 20 --require-converged \
  --outprefix diagnostics/phase_scan/runs/2026-03-19_preflight_require/main
```
