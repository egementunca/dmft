# DMFT Code Documentation

This document describes the current code layout in `src/dmft`, the schemes
implemented, and how to run and extend the project.

## 0) Schemes Overview

The codebase implements multiple ghost-DMFT schemes, in chronological order:

| Scheme | Location | Status | Key idea |
|--------|----------|--------|----------|
| **Two-Ghost (Variant A/B)** | `dmft_loop.py`, `gateway.py`, `matching.py` | Two-ghost reference | Single-site impurity + ghost poles; Bethe self-consistency (A) or correlator matching (B) |
| **Bond** | `bond.py`, `bond_ed.py` | March 2026 | Two-site cluster with shared bond orbitals; corrected matching, alternating loop |
| **Dimer** | `dimer.py`, `dimer_ed.py`, `dimer_gateway.py`, `dimer_lattice.py` | April 2026, superseded by Nested Cluster | Dimer impurity (dA, dB) on square lattice; (U, n) phase diagram study |
| **Nested Cluster** | `nested_cluster.py` | April 2026, current | BPK combination of single-site + two-site impurities with moment conditions; reuses bond/dimer ED kernels |

A standalone reference combining Dimer + Nested Cluster lives at the repo root:
`ghost_cluster_standalone.py` (professor's code).

## 1) Project Structure

### Two-Ghost variants (single-site, original)

- `src/dmft/config.py` — `DMFTParams`, `PoleParams`. Central sign and unit conventions.
- `src/dmft/dmft_loop.py` — `dmft_loop(...)` (Variant A) and `dmft_loop_two_ghost(...)` (Variant B).
- `src/dmft/lattice.py` — Bethe-lattice `G_loc` and lattice h-sector correlators. Also exposes square-lattice helpers (`make_square_lattice`, `lattice_statics`) used by the bond / nested-cluster schemes.
- `src/dmft/gateway.py` — Quadratic gateway model `H_imp^(0)` and exact gateway correlators. Also exposes `gateway1_statics` / `gateway2_statics` for the bond and nested-cluster schemes.
- `src/dmft/matching.py` — Pole fitting and correlator matching (`match_h_correlators`, `match_g_correlators`, `fit_*_poles`).
- `src/dmft/greens_function.py` — Pole-form `Delta(iw)`, `Sigma(iw)`, impurity Dyson helpers.
- `src/dmft/matsubara.py` — Matsubara grids, Fermi function, summation helpers.
- `src/dmft/observables.py` — `Z`, spectral functions, impurity g-correlators.
- `src/dmft/phase_scan.py` — Professor-style `GhostDMFT_M` adapter, free-energy terms (`Omega_lat + Omega_imp - Omega_gateway`), branch-continuation scan, phase-boundary plotting.
- `src/dmft/schur.py` — Schur complement utilities and block Green's function helpers.
- `src/dmft/solvers/` — Lehmann-Green's-function impurity solvers for Variant A/B:
  - `ipt.py` — fast half-filling IPT solver.
  - `ed.py` — exact diagonalization solver for discrete baths.
  - `base.py` — solver interface (`ImpuritySolver.solve`).

### Bond scheme (two-site cluster, March 2026)

- `src/dmft/bond.py` — `solve_singlesite`, `solve_bond` (alternating outer loop with dmu bisection), `run_temperature_sweep`, `save_results`.
- `src/dmft/bond_ed.py` — Sector-blocked Fock-space ED kernels: `impurity1_statics` (single-site) and `build_H2` / `impurity2_statics` (two-site, blocked by (n_up, n_down)). Bitmask encoding with `@lru_cache`. Includes optional CuPy / cuSOLVER GPU dispatch via `_eigh`.

### Dimer scheme (dimer impurity on square lattice, April 2026)

- `src/dmft/dimer.py` — `solve_T` (single-temperature self-consistency), temperature sweep, (U, n) study. Supports half-filling (`n_target=2.0`) and arbitrary doping via `mu` bisection.
- `src/dmft/dimer_ed.py` — Sector-blocked ED for dimer impurity (`Norb = 2 + 2*M`). Reuses `_eigh` and `_hop_element` from `bond_ed`.
- `src/dmft/dimer_gateway.py` — Quadratic dimer gateway, one-body diagonalization (matrix size `2 + 4M`).
- `src/dmft/dimer_lattice.py` — Vectorized k-sum for dimer unit cell on square lattice (matrix size `2 + 2M` per k).

### Nested Cluster scheme (BPK combination, April 2026, current)

- `src/dmft/nested_cluster.py` — Combines single-site (Impurity 1) and two-site (Impurity 2) clusters via BPK: `(1-z)*<O>_1 + z*<O>_2 = <O>_lattice`. Uses `impurity1_statics` from `bond_ed.py` and `dimer_impurity_obs` from `dimer_ed.py`. Includes moment conditions and `t_g` / `t_h` inter-site hopping.

### Standalone (professor's reference)

- `ghost_cluster_standalone.py` (repo root) — Combination of `dimer_ghost_dmft_faster.py` + `ghost_nested_cluster_v2.py`. Original code preserved for cross-validation; bug fix `n_g2* -> n_g2_` in impurity2 / gateway2 applied.

### Tests and notebooks

- `tests/` — End-to-end and module-level tests for loops, matching, gateway, lattice (Bethe + square), solvers, bond ED kernels, and utilities.
- `notebooks/` — Milestone notebooks for the project progression.

## 2) Core Conventions

From `config.py`, these are the locked conventions used throughout (all schemes):

- Energy unit: `D = 2t = 1` by default (`t = 0.5`).
- Dyson sign convention: `G^{-1}(iw) = iw + mu - (...)`.
- Pole forms:
  - `Delta(iw) = sum_l |V_l|^2 / (iw - eps_l)`
  - `Sigma(iw) = sigma_inf + sum_l |W_l|^2 / (iw - eta_l)`
- Half-filling defaults: `mu = U/2`, `eps_d = 0`.

When adding new code, keep these sign conventions consistent with existing modules and tests.

## 3) Execution Flows

### Two-Ghost Variant A (`dmft_loop`)

1. Build `Sigma(iw)` from current ghost poles.
2. Compute `G_loc(iw)` on Bethe lattice.
3. Set `Delta(iw) = t^2 G_loc(iw)`.
4. Fit `Delta(iw)` to bath poles `{eps, V}`.
5. Solve impurity model with `{eps, V}`.
6. Update `sigma_inf = U * n_imp`.
7. Mix self-energy and refit ghost poles `{eta, W}` to the mixed `Sigma`.
8. Check convergence.

Notes:
- Variant A keeps the output pole self-energy synchronized with the converged Matsubara `Sigma`, which is required for reliable `spectral_function_from_poles(...)`.
- For ED in Variant A, the loop applies conservative Sigma mixing internally to reduce oscillatory updates.

### Two-Ghost Variant B (`dmft_loop_two_ghost`)

1. Build `Sigma(iw)` from ghost poles.
2. Compute `G_loc(iw)` and lattice h-sector correlators.
3. Match lattice ↔ gateway h-correlators to update bath poles `{eps, V}`.
4. Solve impurity with `{eps, V}`.
5. Update `sigma_inf = U * n_imp` (tail constraint, Option A).
6. Match impurity ↔ gateway g-correlators to update ghost poles `{eta, W}`.
7. Rebuild `Sigma(iw)` from updated poles and iterate.

Default behavior is the notes-faithful correlator-matching update for both sectors. A debug fallback remains available via `ghost_update_mode="fit"`.

### Bond scheme (`solve_bond`)

1. Solve single-site `solve_singlesite` first to obtain warm-start `eta, W`.
2. Alternating outer loop:
   - Fix impurity correlators, solve gateway for bath / ghost params.
   - Re-solve impurity (single-site + two-site, sector-blocked ED).
   - `dmu` Newton-bisection for half-filling (`n_d = 1`).
3. Direct matching conditions (not BPK combination).
4. Independent ghost counts per family: `M1g`, `M2g`, `Mbg` (`M1h = M2h = Mbh = 1` fixed).
5. Bond form factor `gamma_k = eps_k / 4`.

### Dimer scheme (`solve_T`)

Per iteration:
1. Lattice → h-targets.
2. Gateway `least_squares` fit (h-sector: gateway = lattice).
3. Impurity ED → g-targets; if doped, update `Sigma_inf` and bisect `mu`.
4. Gateway `least_squares` fit (g-sector: gateway = impurity).

Layout: `x0 = [eta_h(M), W_h(M), t_h(M), t_g(M), eps_g(M), V_g(M)]`, optionally appended with `[mu, Sigma_inf]` for doping.

### Nested Cluster (`nested_cluster.py`)

Per iteration:
1. Lattice → h-targets (`n_h, d_h, n_h2, d_h2, h2hop`).
2a. Fit `V1, V2` via BPK: `(1-z)*gw1 + z*gw2 = lattice` (h-sector).
2b. Fit `t_h` from `h2hop`.
3. Impurity 1 + Impurity 2 → g-targets.
4a. Fit `eta1, W1, eta2, W2, W` via g-sector matching + moment conditions.
4b. Fit `t_g` from `g2hop`.

## 4) Solver Interface

Variant A/B impurity solvers (`src/dmft/solvers/`) implement:

```python
solve(iw, mu, eps_d, U, V, eps, beta, sigma_inf) -> dict
```

Minimum expected keys in the return dictionary:

- `G_imp`
- `Sigma_imp`
- `n_imp`
- `n_double` (where available)

Optional keys used by Variant B diagnostics / matching:

- `bath_gg`
- `bath_dg`

The bond / dimer / nested-cluster schemes use **statics-only ED** (no Lehmann Green's function) — see `bond_ed.py` (`impurity1_statics`, `impurity2_statics`) and `dimer_ed.py` (`dimer_impurity_obs`). These return equal-time correlators directly from the Boltzmann-weighted sector spectra and never build the full Fock-space matrix.

## 5) Running the Project

### Install (editable + dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For GPU (bond / nested-cluster ED):

```bash
module load cuda/12.2     # or matching cluster CUDA
pip install cupy-cuda12x
```

### Run tests

```bash
PYTHONPATH=src pytest -q
```

### Two-Ghost Variant B (programmatic)

```python
from dmft.config import DMFTParams
from dmft.solvers.ed import EDSolver
from dmft.dmft_loop import dmft_loop_two_ghost

p = DMFTParams.half_filling(U=2.0, beta=50.0, M_g=2, M_h=2)
p.tol = 1e-4
p.mix = 0.05
solver = EDSolver()
r = dmft_loop_two_ghost(
    p,
    solver,
    verbose=False,
    ghost_update_mode="correlator",
    bath_mix=0.05,
    ghost_mix=0.05,
    h_reg_strength=1e-2,
    g_reg_strength=1e-2,
)
print(r["Z"], r["n_imp"])
```

### Phase scan (Two-Ghost adapter)

```python
import numpy as np
from dmft.phase_scan import run_phase_scan, save_scan_outputs

df, boundaries = run_phase_scan(
    U_vals=np.linspace(2.0, 3.4, 30),
    T_vals=np.linspace(0.02, 0.20, 20),
    M=1,
    n_matsubara=512,
    mix=0.05,
    tol=1e-4,
    maxiter=160,
)
save_scan_outputs(df, boundaries, outprefix="diagnostics/phase_scan/ghost_dmft")
```

CLI: `scripts/run_phase_scan.py` (see `docs/notes/BU_SCC_PHASE_SCAN_RUNBOOK.md`).

### Bond scheme (CLI)

```bash
python3 scripts/run_bond_sweep.py --M1g 1 --M2g 1 --Mbg 1 --U 1.3 --t 0.5
```

See `jobs/RUN_BOND_INSTRUCTIONS.md` for cluster submission, parameter choices, and validation against the professor's `ghost_dmft_bond_new.py`.

### Dimer scheme (CLI)

```bash
python3 scripts/run_dimer_sweep.py --U 1.3 --M 1     # temperature sweep
python3 scripts/run_dimer_study.py ...                # (U, n) phase diagram study
```

### Nested Cluster scheme (CLI)

```bash
python3 scripts/run_nested_cluster.py ...
```

See `jobs/README.md` for SGE templates (`nested_cluster_m1.sh`, `_m1_gpu.sh`, `_m2.sh`, `_m2_gpu.sh`, `_m1_prof.sh`).

## 6) Extension Guidelines

- Add new Variant A/B impurity solvers under `src/dmft/solvers/` and inherit `ImpuritySolver`.
- For new statics-only ED kernels (bond / dimer / nested-cluster style), follow the sector-blocked pattern in `bond_ed.py` — bitmask encoding, `@lru_cache` for sector basis, and dispatch through `_eigh` so GPU acceleration applies automatically.
- If you introduce new matching rules for Variant B, keep them in `matching.py` and add round-trip tests similar to `tests/test_matching.py`.
- If you change sign conventions or Dyson definitions, update:
  - `config.py` conventions,
  - relevant module docstrings,
  - tests that encode those conventions.
- For new observables, prefer `observables.py` and add dedicated tests.
- Cross-validate any new scheme against `ghost_cluster_standalone.py` or a professor's reference where one exists; record validation deltas under `results/reports/`.
