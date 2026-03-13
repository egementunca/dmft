# Phase Scan Artifacts

This folder contains branch-continuation phase-scan outputs for the two-ghost
ED adapter (`src/dmft/phase_scan.py`).

## Layout

- `runs/2026-03-13_coarse_v1/`
  - First coarse scan (M=1, 10 U points, 5 T points), strict validity
    filters active; produced many `unknown` points and no coexistence.
- `runs/2026-03-13_coarse_v2/`
  - Coarse scan with raw converged-branch acceptance + coexistence based on
    explicit branch split (`|ΔZ|`, `|ΔD|` thresholds).
- `runs/2026-03-13_cli_smoke/`
  - Lightweight CLI smoke run.
- `run_summary.csv`
  - One-row-per-run summary of counts and branch-separation metrics.
- `PHASE_SCAN_ANALYSIS.md`
  - Interpretation of the current runs and next steps.

Each run directory has a consistent file set:

- `phase_scan.csv`
- `phase_boundaries.csv`
- `D_vs_U.png`
- `Z_vs_U.png`
- `deltaF.png`
- `coexistence.png`
- `phase_boundaries.png`

## Reproduce

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
python3 scripts/run_phase_scan.py \
  --nu 10 --nt 5 --n-iw 256 --maxiter 120 \
  --outprefix diagnostics/phase_scan/runs/2026-03-13_coarse_v2
```
