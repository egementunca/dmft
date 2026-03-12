# DMFT Code Documentation

This document describes the current code layout in `src/dmft`, the two DMFT loop variants, and how to run and extend the project.

## 1) Project Structure

- `src/dmft/config.py`
  - Defines `DMFTParams` (physical + numerical controls) and `PoleParams` (bath/ghost pole parameters).
  - Central sign and unit conventions are documented here.
- `src/dmft/dmft_loop.py`
  - `dmft_loop(...)`: Variant A (standard DMFT with Bethe self-consistency and bath pole fitting).
  - `dmft_loop_two_ghost(...)`: Variant B (two-ghost loop with correlator matching).
- `src/dmft/lattice.py`
  - Bethe-lattice local Green's function and lattice h-sector correlators.
- `src/dmft/gateway.py`
  - Quadratic gateway model (`H_imp^(0)`) and exact gateway correlators.
- `src/dmft/matching.py`
  - Pole fitting (`fit_hybridization_poles`, `fit_self_energy_poles`) and correlator matching (`match_h_correlators`, `match_g_correlators`).
- `src/dmft/greens_function.py`
  - Pole-form `Delta(iw)`, `Sigma(iw)`, and impurity Dyson helpers.
- `src/dmft/solvers/`
  - `ipt.py`: fast half-filling IPT solver.
  - `ed.py`: exact diagonalization solver for discrete baths.
  - `base.py`: solver interface.
- `src/dmft/matsubara.py`
  - Matsubara grids, Fermi function, and summation helpers.
- `src/dmft/observables.py`
  - `Z`, spectral functions, impurity g-correlators.
- `src/dmft/schur.py`
  - Schur complement utilities and block Green's function helpers.
- `tests/`
  - End-to-end and module-level tests for loops, matching, gateway, lattice, solvers, and utilities.
- `notebooks/`
  - Milestone notebooks for the project progression.

## 2) Core Conventions

From `config.py`, these are the locked conventions used throughout:

- Energy unit: `D = 2t = 1` by default (`t = 0.5`).
- Dyson sign convention: `G^{-1}(iw) = iw + mu - (...)`.
- Pole forms:
  - `Delta(iw) = sum_l |V_l|^2 / (iw - eps_l)`
  - `Sigma(iw) = sigma_inf + sum_l |W_l|^2 / (iw - eta_l)`
- Half-filling defaults: `mu = U/2`, `eps_d = 0`.

When adding new code, keep these sign conventions consistent with existing modules and tests.

## 3) DMFT Execution Flow

### Variant A (`dmft_loop`)

1. Build `Sigma(iw)` from current ghost poles.
2. Compute `G_loc(iw)` on Bethe lattice.
3. Set `Delta(iw) = t^2 G_loc(iw)`.
4. Fit `Delta(iw)` to bath poles `{eps, V}`.
5. Solve impurity model with `{eps, V}`.
6. Update `sigma_inf = U * n_imp`.
7. Mix self-energy and refit ghost poles `{eta, W}` to the mixed `Sigma`.
8. Check convergence.

Notes:
- Variant A keeps the output pole self-energy synchronized with the converged
  Matsubara `Sigma`, which is required for reliable `spectral_function_from_poles(...)`.
- For ED in Variant A, the loop applies conservative Sigma mixing internally to
  reduce oscillatory updates.

### Variant B (`dmft_loop_two_ghost`)

1. Build `Sigma(iw)` from ghost poles.
2. Compute `G_loc(iw)` and lattice h-sector correlators.
3. Match lattice ↔ gateway h-correlators to update bath poles `{eps, V}`.
4. Solve impurity with `{eps, V}`.
5. Update `sigma_inf = U * n_imp` (tail constraint, Option A).
6. Match impurity ↔ gateway g-correlators to update ghost poles `{eta, W}`.
7. Rebuild `Sigma(iw)` from updated poles and iterate.

Default behavior is now the notes-faithful correlator-matching update for both
sectors. A debug fallback remains available via `ghost_update_mode="fit"`.

## 4) Solver Interface

All impurity solvers implement:

```python
solve(iw, mu, eps_d, U, V, eps, beta, sigma_inf) -> dict
```

Minimum expected keys in the return dictionary:

- `G_imp`
- `Sigma_imp`
- `n_imp`
- `n_double` (where available)

Optional keys used by Variant B diagnostics/matching:

- `bath_gg`
- `bath_dg`

## 5) Running the Project

### Install (editable + dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Run tests

```bash
PYTHONPATH=src pytest -q
```

### Minimal programmatic run

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

## 6) Extension Guidelines

- Add new impurity solvers under `src/dmft/solvers/` and inherit `ImpuritySolver`.
- If you introduce new matching rules, keep them in `matching.py` and add round-trip tests similar to `tests/test_matching.py`.
- If you change sign conventions or Dyson definitions, update:
  - `config.py` conventions,
  - relevant module docstrings,
  - tests that encode those conventions.
- For new observables, prefer `observables.py` and add dedicated tests.
