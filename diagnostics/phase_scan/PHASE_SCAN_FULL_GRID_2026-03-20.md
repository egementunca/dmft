# Phase Scan Full-Grid Results (2026-03-20)

## What Was Run

Two-step strategy from the 2026-03-19 preflight analysis, now on the full 30×20
production grid (U ∈ [2.0, 3.4], T ∈ [0.02, 0.20]).

| Run | M | mix | tol | maxiter | compat | filters |
|-----|---|-----|-----|---------|--------|---------|
| `2026-03-20_m1_baseline` | 1 | 0.5 | 1e-8 | 100 | yes | none |
| `2026-03-20_m2_quality` | 2 | 0.2 | 1e-6 | 180 | no | `--require-converged`, branch filters |

Job scheduler: SGE (`qsub`), project `compcircuits`.
Runtime: M1 ~28 min (2 cores, 8G), M2 ~56 min (4 cores, 16G).

## Results

### M1 Baseline (compat mode)

- 600/600 points converged for both branches (metal and insulator)
- 600/600 metal-valid, 600/600 insulator-valid
- coexistence: 600/600 (all points, expected in compat mode)
- stable phase: metal=322, insulator=278
- `Uc1 = 2.0`, `Uc2 = 3.4` at every T (boundaries pinned to scan edges)
- `Uc_eq`: 17/20 temperatures have values, range [2.00, 3.32], 3 NaN

Interpretation:
- compat-mode coexistence rule `|D_m - D_i| > 1e-6 OR |deltaF| > 1e-10` fires
  everywhere, as expected.
- Boundaries pinned to edges means the scan range does not fully contain the
  physical transition for M=1 at this tolerance.
- Phase split (metal ~54%, insulator ~46%) is qualitatively reasonable for a
  Bethe-lattice Mott transition in this U range.

### M2 Quality (strict filters)

- convergence: metal 575/600 (95.8%), insulator 505/600 (84.2%)
- validity: metal 337/600, insulator 141/600
- coexistence: 8/600
- stable phase: metal=337, insulator=133, unknown=130
- `Uc1`: 11/20 temperatures have values, range [2.39, 3.35]
- `Uc2`: 18/20 temperatures have values, range [2.00, 3.40]
- `Uc_eq`: all NaN (no free-energy crossing found)

Interpretation:
- Validity gating is aggressive: branch filters (`z_metal_min=0.10`,
  `z_ins_max=0.08`, `v_ins_max=0.28`) reject the majority of insulator-branch
  solutions and a substantial fraction of metal-branch solutions.
- 130 points labeled `unknown` suggests the M=2 ansatz struggles to satisfy
  both convergence and branch-filter criteria simultaneously in much of the
  scan region, especially at higher T.
- `Uc_eq` all NaN: free-energy crossing requires both branches valid at the
  same (U, T), which the strict filters prevent in most cases.
- The 8 coexistence points cluster near mid-U, low-T — the only region where
  both branches pass all filters.

## Expected vs Unexpected

Expected:
- M1 compat over-reporting coexistence (confirmed preflight behavior on full grid)
- M2 strict filters reducing coexistence (intended design)
- M2 convergence rates lower than M1 (larger ansatz is harder to converge)

Unexpected:
- M2 `Uc_eq` entirely NaN — no free-energy crossing at any temperature. This
  indicates the branch filters may be too restrictive for the current M=2
  parametrization, or `maxiter=180` is insufficient for reliable insulator
  convergence at these tolerances.

## Suggested Next Steps

1. Relax M2 branch filters (e.g. `z_ins_max → 0.12`, `v_ins_max → 0.35`) and
   rerun to recover more valid insulator points.
2. Increase `maxiter` to 250–300 for M2 to improve insulator convergence rate.
3. Widen U range to [1.5, 4.0] for M1 baseline to check whether boundaries
   move away from scan edges.
4. If `Uc_eq` remains NaN with relaxed filters, investigate the free-energy
   functional implementation for possible sign or offset issues in the M=2 case.
