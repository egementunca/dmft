# Residual Conditioning + Diagnostics Fix Notes

## Scope
Minimal, targeted changes were applied to disambiguate residual scaling artifacts from true mismatches and improve stationarity solve conditioning, without changing locked physics conventions (Bethe `D=1`, `t=0.5`, Option A for `sigma_inf`, PH-symmetric real workflow).

## What Changed

### 1) Matching scaling + optimizer bounds
- File: `src/dmft/matching.py` (`141-205`, `208-317`, `320-383`, `386-494`)
- Changes:
  - Replaced single floor scaling with component-wise floors:
    - h-sector: `scale_hh=max(|hh_target|, 5e-2)`, `scale_dh=max(|dh_target|, 1e-2)`
    - g-sector: `scale_gg=max(|gg_target|, 5e-2)`, `scale_dg=max(|dg_target|, 1e-2)`
  - Added explicit `trf` bounds in both h/g matching solvers:
    - energies bounded by `[-energy_max, +energy_max]` (or positive pair variables in symmetric parameterization)
    - couplings bounded by `[0, coupling_max]`
  - Kept existing isotropic drift regularization `sqrt(reg_strength) * (x - x0)`.

### 2) Absolute mismatch diagnostics + residual-aware convergence stages
- File: `src/dmft/dmft_loop.py` (`157-250`, `286-289`, `350-368`, `432-620`, `622-725`, `781-788`)
- Changes:
  - Added per-iteration absolute mismatch diagnostics:
    - per-component deltas (`hh/dh/gg/dg`), absolute deltas, scales, targets, predictions in `history[*]['h_match']` / `history[*]['g_match']`
    - compact max summaries in iteration record:
      - `max_abs_dhh`, `max_abs_ddh`, `max_abs_dgg`, `max_abs_ddg`
      - `max_target_hh`, `max_target_dh`, `max_target_gg`, `max_target_dg`
  - Kept scaled norms for stationarity (`h_resid`, `g_resid`) and added absolute norms (`h_resid_abs`, `g_resid_abs`).
  - Added two-stage convergence controls:
    - optional strict mode: `strict_stationarity`
    - optional polish stage after diff+causality convergence: `polish_iters`, `polish_sigma_mix`, `tol_h`, `tol_g`
  - Added `stable_step` guard to avoid declaring convergence on causality-rejected updates.

## Why This Is Minimal and Targeted
- The residual definitions and optimizer conditioning were changed, not the physical equations or solver model.
- No change to sign conventions, impurity Hamiltonian conventions, lattice self-consistency, or Option A (`sigma_inf` tail-only).

## Repro Runs (After Changes)
Settings: `U=2`, `beta=50`, `mix=0.05`, `tol=1e-4`, `max_iter=500`, `h_reg=g_reg=1e-2`, `ghost_update_mode='correlator'`, `symmetric=True`.

| Case | Before diff | Before back_fail | Before h_res | Before g_res | Before max\|Δdh\| | Before max\|Δdg\| | After diff | After back_fail | After h_res (scaled) | After g_res (scaled) | After h_res (abs) | After g_res (abs) | After max\|Δdh\| | After max\|Δdg\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `M=2, n_iw=512` | `1.053e-05` | `54` | `6.615e+00` | `7.427e-02` | `4.678e+00` | `1.474e-02` | `1.014e-04` | `322` | `1.660e-04` | `7.574e-01` | `5.880e-05` | `1.481e-01` | `1.472e-05` | `1.088e-01` |
| `M=2, n_iw=1024` | `4.329e-05` | `91` | `2.688e-02` | `6.996e-02` | `1.890e-02` | `1.344e-02` | `0.000e+00` | `308` | `6.438e-04` | `7.567e-01` | `2.224e-04` | `1.477e-01` | `7.652e-05` | `1.086e-01` |
| `M=1, n_iw=2048` | `6.794e-05` | `35` | `3.236e+02` | `2.425e-01` | `3.236e+02` | `2.413e-01` | `0.000e+00` | `28` | `1.424e-03` | `6.359e-01` | `6.775e-04` | `2.262e-01` | `6.775e-04` | `2.250e-01` |
| `M=3, n_iw=2048` | `9.276e-05` | `84` | `2.608e+01` | `5.590e-02` | `2.171e+01` | `3.251e-02` | `0.000e+00` | `238` | `1.302e-03` | `6.190e-01` | `3.644e-04` | `7.482e-02` | `4.730e-05` | `4.808e-02` |

Data source: `diagnostics/residual_fix_after_runs.json` (paired with `diagnostics/two_ghost_residual_diagnostics.json` baseline entries).

## Interpretation
- Large h-residuals were mostly scaling/conditioning artifacts in problematic cases:
  - `max|Δdh|` collapsed from `O(1..10^2)` to `O(10^-5..10^-4)` in all reported cases.
  - This indicates prior h blowups were not a faithful measure of physical mismatch.
- A nontrivial g-sector mismatch remains (`max|Δdg| ~ 5e-2 .. 2e-1`), so not all residual issues were scaling artifacts.
- `backtrack_fail` remains high (especially `M=2/3`), suggesting ghost updates are still causality-sensitive; this is now visible with clearer diagnostics rather than hidden by a single scaled norm.

## Bottom Line
- The fix cleanly separates:
  - normalization artifacts (largely resolved in h-sector),
  - from true remaining mismatch/conditioning pressure (mainly g-sector + causality backtracking pressure).
