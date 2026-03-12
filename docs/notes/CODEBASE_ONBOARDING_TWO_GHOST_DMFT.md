# CODEBASE_ONBOARDING_TWO_GHOST_DMFT

## 0) Locked Conventions (as implemented)
- Bethe lattice: `D=1`, `t=0.5`.
- Green's-function convention: `G^{-1}(iw)=iw+mu-h` (`mu` is not inside `h`).
- Pole forms:
  - `Delta(iw)=sum_l |V_l|^2/(iw-eps_l)`
  - `Sigma(iw)=sigma_inf + sum_l |W_l|^2/(iw-eta_l)`
- Half-filling first: `mu=U/2`, `eps_d=0`, `symmetric=True`.
- Bethe self-consistency: `Delta_cav(iw)=t^2 G_loc(iw)`.
- Option A: `sigma_inf` is tail-only; impurity Hamiltonians are unshifted.
- Variant B default is notes-faithful correlator matching; ghost `"fit"` mode is debug-only fallback.

---

## A) One-Page Architecture Map (Modules and Responsibilities)

| Area | Module | Key APIs | Inputs -> Outputs (physical meaning) | Notes concept |
| --- | --- | --- | --- | --- |
| Parameters/config | `src/dmft/config.py` | `DMFTParams`, `PoleParams` | Physical knobs (`U,beta,mu,t,M_g,M_h`) + pole state `{eps,V,eta,W,sigma_inf}` | Global conventions and state vector |
| Matsubara grid + sums | `src/dmft/matsubara.py` | `matsubara_frequencies`, `matsubara_sum_numerical`, `matsubara_sum_pair_numerical`, convergence helpers | Matsubara arrays -> robust equal-time sums with tail subtraction | Finite-T Matsubara formalism; correlators from frequency sums |
| Pole algebra / Dyson pieces | `src/dmft/greens_function.py` | `hybridization`, `self_energy_poles`, `weiss_field_inverse`, `greens_function_impurity` | Pole params + `iw` -> `Delta(iw)`, `Sigma(iw)`, `G_imp(iw)` | Pole ansatz and Dyson equation |
| Lattice (Bethe) | `src/dmft/lattice.py` | `bethe_local_gf`, `bethe_self_consistency`, `lattice_h_sector_gf`, `lattice_correlators` | `Sigma(iw)` -> `G_loc(iw)` and lattice h-sector targets `<h†h>_lat`, `<d†h>_lat` | Bethe closure + Schur-derived lattice correlators |
| Gateway quadratic model `H_imp^(0)` | `src/dmft/gateway.py` | `gateway_onebody_matrix`, `gateway_greens_functions`, `gateway_correlators`, `gateway_correlators_from_matsubara` | `{eps,V,eta,W,sigma_inf}` -> exact quadratic correlators in h and g sectors | Quadratic reference in functional `Omega[H_imp^(0)]` |
| Schur complement utilities | `src/dmft/schur.py` | `schur_complement_diag`, `block_greens_functions` | Block inverse pieces -> `G_dd`, off-diagonal blocks | Integrating out bath/ghost sectors |
| Matching / stationarity solve | `src/dmft/matching.py` | `match_h_correlators`, `match_g_correlators`, plus `fit_*_poles` | Target correlators + fixed sector -> updated poles via least squares | HF stationarity constraints (`∂F/∂lambda=0`) |
| Impurity solvers | `src/dmft/solvers/base.py`, `ed.py`, `ipt.py` | `ImpuritySolver.solve`, `EDSolver.solve`, `IPTSolver.solve` | Bath params -> interacting `G_imp`, `Sigma_imp`, `n_imp`, `n_double`, and ED bath correlators | `Omega[H_imp]` side of functional |
| DMFT loops | `src/dmft/dmft_loop.py` | `dmft_loop` (A), `dmft_loop_two_ghost` (B) | Iteration glue for lattice, matching, impurity solve, convergence | Variant A standard DMFT; Variant B notes-faithful two-ghost |
| Observables / plots / diagnostics | `src/dmft/observables.py`, `plotting.py`, `scripts/*.py` | `quasiparticle_weight`, `impurity_g_correlators`, plot helpers, residual scripts | Derived quantities and diagnostics tables/plots | Interpretation and validation tooling |

---

## B) Notes ↔ Code Mapping (Equation-to-Function)

| Notes object / equation concept | Code location | Concrete functions |
| --- | --- | --- |
| Impurity action with Weiss field `S_imp[Delta]` | `src/dmft/greens_function.py`, solvers | `weiss_field_inverse`, `hybridization`, solver `solve(...)` |
| Dyson for impurity `G_imp^{-1}=G0^{-1}-Sigma` | `src/dmft/greens_function.py`, `src/dmft/solvers/ed.py` | `greens_function_impurity`; ED computes `Sigma_imp = G0_inv - 1/G_imp` |
| Bethe closure `Delta=t^2 G_loc` | `src/dmft/lattice.py` | `bethe_self_consistency` |
| Bethe local GF from `Sigma` | `src/dmft/lattice.py` | `bethe_local_gf` |
| Schur relations for block Green's functions | `src/dmft/schur.py`, `src/dmft/gateway.py`, `src/dmft/lattice.py` | `block_greens_functions`, `gateway_greens_functions`, `lattice_h_sector_gf` |
| Pole representations for `Delta` and `Sigma` | `src/dmft/greens_function.py` | `hybridization`, `self_energy_poles` |
| Functional structure `F = Omega_lat + Omega_imp - Omega_imp^(0)` | Implicitly realized in loop + matching | `dmft_loop_two_ghost` coordinates lattice targets + impurity targets + gateway predictions |
| Stationarity wrt bath params (h-sector matching) | `src/dmft/matching.py` | `match_h_correlators` |
| Stationarity wrt ghost params (g-sector matching) | `src/dmft/matching.py` | `match_g_correlators` |
| Gateway correlators `<h†h>,<d†h>,<g†g>,<d†g>` | `src/dmft/gateway.py` | `gateway_correlators` (exact diagonalization of one-body matrix) |
| Lattice h-target correlators | `src/dmft/lattice.py` | `lattice_correlators` |
| Impurity g-target correlators | `src/dmft/observables.py` or ED direct outputs | `impurity_g_correlators`; ED also returns `bath_gg`, `bath_dg` |
| Tail constraint `sigma_inf = U <n_-sigma>` | `src/dmft/dmft_loop.py` | `sigma_inf_new = params.U * n_imp` in both loops |
| Option A (unshifted impurity H) | `src/dmft/dmft_loop.py`, solvers | `solver.solve(..., sigma_inf=0.0)` in loop; ED/IPT ignore shift in impurity Hamiltonian |

---

## C) Reading Path (Low-Overwhelm Order)

### Step 1: Conventions + basic formulas
Read:
1. `src/dmft/config.py`
2. `src/dmft/greens_function.py`
3. `src/dmft/lattice.py` (only `bethe_local_gf`, `bethe_self_consistency`)

Verify:
- Your sign convention is exactly what code uses.
- `Delta` and `Sigma` pole definitions match your notes.
- Bethe closure is exactly `t^2 G_loc`.

### Step 2: Gateway model as the central bridge
Read:
1. `src/dmft/gateway.py`
2. `src/dmft/schur.py`

Verify:
- `gateway_onebody_matrix` is Hermitian with your ordering `[d, g..., h...]`.
- Correlator orientation is consistent:
  - `<d†h_l> = f(K)[h_l,d]`
  - `<d†g_l> = f(K)[g_l,d]`
- `gateway_correlators` and Schur-derived blocks tell the same physics.

### Step 3: Correlator construction and truncation handling
Read:
1. `src/dmft/matsubara.py`
2. `src/dmft/lattice.py` (`lattice_correlators`)
3. `src/dmft/observables.py` (`impurity_g_correlators`)

Verify:
- Off-diagonal sums use paired formula (`matsubara_sum_pair_numerical`) instead of naive `2*Re`.
- Tail-subtraction logic is used for diagonal sums.
- Target correlators used in matching are real parts for symmetric workflows.

### Step 4: Matching/stationarity solvers
Read:
1. `src/dmft/matching.py`

Verify:
- h-match updates `{eps,V}` from lattice targets.
- g-match updates `{eta,W}` from impurity targets.
- Residual scaling, regularization, and bounds are visible and physically sensible.
- PH-symmetric parameterization reduces unknowns as expected.

### Step 5: Full loop orchestration
Read:
1. `src/dmft/dmft_loop.py` (`dmft_loop_two_ghost`)
2. `src/dmft/solvers/ed.py` and `src/dmft/solvers/ipt.py`

Verify:
- Variant B sequence is exactly: lattice h-targets -> bath update -> impurity solve -> `sigma_inf` update -> g-targets -> ghost update.
- Option A is enforced (`sigma_inf_impurity = 0.0` for impurity/g matching side).
- Stop conditions, causality checks, and backtracking behavior are clear.

### Step 6: Validate with targeted tests
Run:
```bash
PYTHONPATH=src pytest -q tests/test_gateway.py tests/test_lattice.py tests/test_matching.py tests/test_observables.py tests/test_dmft_loop.py
```
Focus:
- Schur consistency, correlator orientation, causality, and loop behavior.

---

## D) Observable Interpretation Guide

### `G_loc(iw_n)`
- Meaning: local lattice Green's function from Bethe + current `Sigma`.
- Healthy:
  - `Im G_loc(iw_n) < 0` for all positive `w_n`.
  - High-frequency tail `G_loc ~ 1/(iw_n)`.
- Warning:
  - Positive `Im G_loc` indicates causality violation or unstable updates.

### `Sigma(iw_n)` and `sigma_inf`
- Meaning: self-energy from ghost poles with explicit static tail.
- Healthy:
  - `Re Sigma(iw_n)` approaches `sigma_inf` at large `w_n`.
  - `Im Sigma(iw_n)` behaves smoothly at low `w_n`.
- Warning:
  - Large oscillations at low `w_n` often signal unstable matching/mixing.

### `Z` proxy
- Definition: `Z = [1 - Im Sigma(iw0)/w0]^{-1}`.
- Meaningful when:
  - metallic/Fermi-liquid-like regime (small enough `U`, low `T`).
- Less meaningful when:
  - near/inside insulating regime, strong non-FL behavior, coarse Matsubara resolution.

### `n_imp` (per spin) and `n_double`
- At half-filling, healthy `n_imp ≈ 0.5`.
- `n_double` decreases as `U` increases (for fixed half-filling).
- Drift from 0.5 at symmetric setup usually indicates solver/noise/convention bug.

### Matching residuals (scaled vs absolute)
- Scaled residuals (`h_resid`, `g_resid`): optimizer objective diagnostics.
- Absolute deltas (`max_abs_ddh`, `max_abs_ddg`, etc.): physical mismatch magnitude.
- Interpretation at finite `M`:
  - Scaled small but absolute non-negligible can happen when target magnitudes vary by component.
  - Absolute residuals are the safer gauge for “real mismatch”.

### Causality backtracking
- Meaning: proposed ghost update made `G_loc` noncausal; step is reduced.
- Occasional backtracking is acceptable.
- Frequent `backtrack_alpha=0` indicates poor conditioning, too aggressive mixing, or pathological pole proposals.

### Pole parameters `{eps,V,eta,W}`
- `eps,eta`: effective level positions.
- `V,W`: coupling strengths (in symmetric mode often gauged positive and pair-symmetric).
- Warning patterns:
  - poles bunching/collisions,
  - repeated clipping to bounds,
  - very large couplings with little residual improvement,
  - violent iteration-to-iteration jumps.

### Finite-`M` projection error
- Expect imperfect stationarity at small `M`.
- Typical trend with larger `M`: better absolute matching, but not always monotone every run due to conditioning and local minima.

---

## E) Minimal Reproducible Experiments (Lab Manual)

Use this runner skeleton:
```bash
PYTHONPATH=src python3 - <<'PY'
import numpy as np
from dmft.config import DMFTParams
from dmft.solvers.ed import EDSolver
from dmft.dmft_loop import dmft_loop_two_ghost

def run(U, M, n_iw=512, beta=50.0, mix=0.05, tol=1e-4, max_iter=300):
    p = DMFTParams.half_filling(U=U, beta=beta, n_matsubara=n_iw, M_g=M, M_h=M)
    p.mix = mix
    p.tol = tol
    p.max_iter = max_iter
    r = dmft_loop_two_ghost(
        p, EDSolver(), verbose=False, ghost_update_mode='correlator',
        symmetric=True, bath_mix=mix, ghost_mix=mix,
        h_reg_strength=1e-2, g_reg_strength=1e-2, convergence_metric='sigma'
    )
    h = r["history"][-1]
    print(f"U={U:.2f} M={M} n_iw={n_iw} iters={len(r['history'])} "
          f"diff={h['diff']:.3e} causal={h['causality_ok']} "
          f"h_res={h['h_resid']:.3e} g_res={h['g_resid']:.3e} "
          f"max|ddh|={h.get('max_abs_ddh',0):.3e} max|ddg|={h.get('max_abs_ddg',0):.3e} "
          f"sigma_inf={h['sigma_inf']:.6f} n_imp={r['n_imp']:.6f} Z={r['Z']:.4f}")

for U in [0.0, 1.0, 2.0, 4.0]:
    for M in [1,2,3]:
        run(U=U, M=M, n_iw=512)
PY
```

### Experiment 1: `U=0` sanity
Expected qualitative outcome:
- `Sigma` close to 0 (including `sigma_inf≈0`).
- Causal `G_loc`.
- Residuals should be very small; finite numerical residual may remain from optimization tolerance.

Checks:
1. `sigma_inf` near zero.
2. `Im G_loc < 0`.
3. `n_imp ≈ 0.5`.
4. `max|Δdh|`, `max|Δdg|` small.
5. No persistent backtrack failures.

### Experiment 2: Small `U` (e.g. `U=1`)
Expected:
- Metallic behavior.
- `Z` near 1.
- Small-to-moderate self-energy and stable convergence.

Checks:
1. Causality.
2. Tail consistency: `Re Sigma(iw_n→∞) -> sigma_inf`.
3. `n_imp≈0.5`.
4. Residuals reduce versus iterations.
5. Pole updates stay within moderate range.

### Experiment 3: Intermediate `U` (e.g. `U=2`)
Expected:
- Stronger correlations; `Z` decreases versus `U=1`.
- Finite-`M` stationarity tension appears (especially g-sector).

Checks:
1. Compare scaled and absolute residuals.
2. Confirm no sign-flip causality issues.
3. Inspect backtracking frequency.
4. Compare `M=1->2->3`: absolute mismatches should generally improve.
5. Check whether convergence by `diff` alone still leaves stationarity error.

### Experiment 4: Larger `U` (e.g. `U=4`)
Expected:
- Insulating tendency; larger self-energy magnitude.
- `Z` may become very small or lose clear FL interpretation.

Checks:
1. Causality still holds.
2. `sigma_inf≈U*n_imp` remains consistent.
3. Residuals may plateau at finite `M`.
4. Backtracking may increase.
5. Sensitivity to mixing/regularization likely stronger.

### How to check `M=1->2->3`
- Track both scaled residuals and absolute component deltas.
- If absolute residual drops with `M` but scaled residual is noisy, that is often scaling/conditioning, not physics failure.
- If both stay large, likely true projection mismatch or ill-posed optimization.

---

## F) Debug Checklist (When Something Looks Wrong)

### 1) Sign convention mistakes
- Symptom: wrong tails, wrong occupancy, unstable loop.
- Inspect:
  - `src/dmft/greens_function.py`
  - `src/dmft/lattice.py`
  - solver Dyson construction in `src/dmft/solvers/ed.py`
- Quick test: `tests/test_greens_function.py`, `tests/test_lattice.py`.

### 2) Correlator orientation (`<d†g>` vs `<g†d>`, `<d†h>` vs `<h†d>`)
- Symptom: large persistent off-diagonal mismatch.
- Inspect:
  - `gateway_correlators` extraction in `src/dmft/gateway.py`
  - pair-sum usage in `src/dmft/lattice.py` and `src/dmft/observables.py`
- Quick test: `tests/test_gateway.py::test_gateway_correlators_diag_vs_matsubara`, `tests/test_observables.py`.

### 3) Matsubara truncation / tail handling
- Symptom: non-monotone correlator estimates vs `n_iw`, noisy residuals.
- Inspect:
  - `matsubara_sum_numerical`, `matsubara_sum_pair_numerical` in `src/dmft/matsubara.py`
  - diagnostics flags in correlator builders.
- Action: run with `return_diagnostics=True` and compare convergence curves.

### 4) Optimizer scaling problems
- Symptom: scaled residual huge while absolute mismatch tiny (or opposite).
- Inspect:
  - scale floors and normalized residual construction in `src/dmft/matching.py`.
  - absolute mismatch fields in `history` from `src/dmft/dmft_loop.py`.
- Action: compare `h_resid/g_resid` with `max_abs_ddh/max_abs_ddg`.

### 5) Underdetermined matching geometry
- Symptom: unstable pole drift, many equivalent solutions.
- Inspect:
  - `h_constraints/g_constraints` vs unknown counts in `src/dmft/dmft_loop.py`.
  - symmetry reductions and regularization use.
- Action: keep `symmetric=True`, nonzero regularization, bounded energies/couplings.

### 6) Causality violations
- Symptom: `Im G_loc(iw_n)` positive; repeated backtracking failures.
- Inspect:
  - backtracking branch in `dmft_loop_two_ghost`.
- Action:
  - reduce `mix_ghost`,
  - increase regularization,
  - check for pole clipping saturation/collisions.

### 7) ED size limits / numerical noise
- Symptom: very slow or noisy results for high `M`.
- Inspect:
  - Hilbert space growth in `src/dmft/solvers/ed.py`.
- Action:
  - keep `M<=3` for routine checks,
  - use IPT for fast trend scans, then confirm with ED.

---

## Practical “Start Tomorrow Morning” Plan
1. Run the test subset in Section C Step 6.
2. Do the four experiments in Section E at `M=1,2`.
3. Repeat only `U=2` for `M=3` and `n_iw=1024`.
4. Judge quality using absolute mismatch + causality + stability, not `diff` alone.
5. Only then tune mixing/regularization.
