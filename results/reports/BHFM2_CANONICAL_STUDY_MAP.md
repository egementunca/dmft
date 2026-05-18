# BHFM2 Canonical Study Map

This document sets the working convention for this repo:

- **"Prof's code" = `BHFM2/` only.**
- Other legacy scripts at repo root or `old_scripts/` are historical unless explicitly needed for archaeology.

## 1) Canonical Scope (`BHFM2/`)

Code files:
- `solve_min.py` — minimal production solver (M=2-ready), no anti-bond.
- `ed_fast.py` — numba-accelerated dense sector ED kernels used by `solve_min`.
- `ed_sparse.py` — sparse/Lanczos prototype for larger sectors.
- `run_M2_sweep.py` — half-filling T sweep runner with checkpoints.
- `run_M2_doping.py` — Ferrero-parameter doping/T sweep runner with checkpoints.
- `sigma_k.py` — momentum-resolved Sigma(k, iwn) extraction from fitted poles.
- `plot_fermi_arcs.py` — Ferrero-style diagnostics/plots (AN vs N vs arc).
- `ghost_dmft_bond.py` — full-bond prototype (richer x/y channel structure, research sandbox).
- `check_setup.py` — environment + quick timing sanity checks.

Data assets:
- checkpoints and pkl outputs in `BHFM2/` (`M2_min_T1_best.pkl`, `Tsweep_*.pkl`, etc.).

## 2) What BHFM2 Actually Solves

Primary production path is:
1. Build residual equations in `solve_min.py` (`residual_min`).
2. Use chunked bounded `least_squares` with warm starts.
3. Run either:
   - half-filling sweep (`run_M2_sweep.py`)
   - finite doping sweep (`run_M2_doping.py`)
4. Postprocess to Sigma(k) and Ferrero-style diagnostics (`sigma_k.py`, `plot_fermi_arcs.py`).

Important details:
- Minimal model is explicitly bonding-only (no anti-bond branch).
- Uses BPK-like mixing (`z=0.5` in this minimal formulation) and moment constraints.
- Enforces density and bond-kinetic consistency in residual.
- Uses per-point checkpoint/restart.

## 3) Mapping BHFM2 to Current `src/dmft`

Closest equivalents:
- BHFM2 minimal residual logic:
  - **No exact packaged equivalent** right now.
  - Current "mainline" uses `nested_cluster.py` and `dimer.py` formulations.
- BHFM2 dense ED kernels (`ed_fast.py`):
  - closest in spirit: `bond_ed.py` + `dimer_ed.py` (sector blocked, CPU/GPU option).
- BHFM2 sparse ED (`ed_sparse.py`):
  - **missing in packaged production path**.
- BHFM2 doping sweep runner:
  - nearest: `dimer.py` with `n_target` support.
  - nested cluster path currently half-filling-centric.
- BHFM2 Sigma(k)/Ferrero plots:
  - **missing as a first-class `src/dmft` module/CLI**.

## 4) Item-by-Item Clarification from Gap Review

### Item 1 (expand now): Minimal M=2 residual parity
- Goal: bring BHFM2 `solve_min` into packaged `src/dmft` as canonical reproducible solver path.
- Why: this is the exact objective function used for your current professor-aligned numerical targets.

### Item 2 (confusion on "we ran many numerics")
- Not wasted work. Current numerics split across different formulations:
  - bond / dimer / nested-cluster.
- The issue is **formulation drift**, not useless results.
- Action: annotate every result with solver lineage (BHFM2-minimal vs dimer vs nested).

### Item 3 (x/y bond channels)
- Valid and valuable.
- BHFM2 already contains richer x/y channel ideas in `ghost_dmft_bond.py`.
- Action: treat this as phase-2 extension after minimal parity is packaged and validated.

### Item 4 (Ferrero-style processing, definition)
- "Ferrero-style processing" here means postprocessing for pseudogap/arc signatures:
  - compute Sigma(k, iwn) from fitted poles,
  - compare antinode `(pi, 0)` vs node `(pi/2, pi/2)`,
  - inspect Z(k) along FS arc,
  - make the three diagnostic figures.
- Implemented in BHFM2 by `sigma_k.py` + `plot_fermi_arcs.py`.

### Item 5 (checkpoint/restart importance)
- Agreed: critical for expensive sweeps and cluster reliability.
- BHFM2 has robust per-point checkpoint behavior; our packaged paths should adopt a unified checkpoint utility.

### Item 6 (GPU report + restructure)
- Agreed.
- Existing `docs/gpu_feasibility_report.md` is marked historical and references retired scripts.
- Needs rewrite around:
  - current `bond_ed` CPU/GPU path,
  - M-dependent sector sizes in active solvers,
  - dense vs sparse crossover strategy,
  - practical runbook per solver family.

## 5) Proposed Execution Order (Pragmatic)

1. **Package BHFM2 minimal solver parity** in `src/dmft` (item 1).
2. **Unify checkpoint/restart API** and retrofit dimer/nested runners (item 5).
3. **Add Ferrero diagnostics module + CLI** (item 4).
4. **Introduce x/y bond-channel extension branch** from BHFM2 full-bond prototype (item 3).
5. **Refresh GPU feasibility report + repository structure cleanup** (item 6).

## 6) Suggested Target Repo Structure (post-cleanup)

- `src/dmft/prof_minimal.py` (or `src/dmft/bhfm2_minimal.py`)
- `src/dmft/postprocess/sigma_k.py`
- `src/dmft/postprocess/ferrero.py`
- `src/dmft/io/checkpoint.py`
- `scripts/run_prof_minimal_sweep.py`
- `scripts/run_prof_minimal_doping.py`
- `scripts/plot_ferrero_diagnostics.py`

This keeps professor-canonical workflows first-class while preserving existing solvers.
