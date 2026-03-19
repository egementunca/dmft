# Phase Scan Initial Check (2026-03-19)

## What Was Checked

Runs checked:

- `runs/2026-03-19_preflight/main_phase_scan.csv`
- `runs/2026-03-19_preflight_require/main_phase_scan.csv`

Metrics inspected:

- file generation success (`csv + png` set)
- branch convergence rates (`conv_metal`, `conv_insulator`)
- stable-phase labeling
- coexistence flags
- boundary extraction (`Uc1`, `Uc2`, `Uc_eq`)

## Results

### 1) `2026-03-19_preflight` (compatibility mode)

- status: pipeline executes end-to-end and produces all outputs
- convergence rates: metal `0.875`, insulator `0.750`
- stable labels: `metal=4`, `insulator=4`, `unknown=0`
- coexistence points: `8/8`

Interpretation:
- This behavior is expected under the professor-compatible coexistence rule:
  `|D_m - D_i| > 1e-6 OR |deltaF| > 1e-10`.
- On small grids, this can easily mark nearly all points as coexistence.

### 2) `2026-03-19_preflight_require` (`--require-converged`)

- status: pipeline executes end-to-end and produces all outputs
- convergence rates: metal `0.667`, insulator `0.667`
- stable labels: `metal=4`, `unknown=2`
- coexistence points: `4/6`

Interpretation:
- Convergence gating works as intended and removes non-converged points from
  validity/stability accounting.
- Boundaries become stricter but can be sparse on very small grids.

## Expected vs Unexpected

Expected:

- compatibility mode can over-report coexistence
- convergence-gated mode reduces false positives
- small preflight grids are only sanity checks, not final phase diagrams

Unexpected:

- none blocking in these initial checks

## Cleanup Performed

Earlier exploratory 2026-03-13 runs and the older analysis note were removed.
Active `runs/` now only contains current preflight checks.

## Next Step for “works as expected” at study quality

Use two-step strategy:

1. `compat_mode=True` baseline run to compare with professor sketch behavior.
2. quality run with:
   - `--no-compat-mode`
   - `--require-converged`
   - branch filters/tolerances
   - `M=2` grid for physically meaningful boundaries.
