# Phase Scan Critical Review

Date: 2026-03-13  
Scope: professor-style phase-scan sketch vs current adapter implementation

## Executive Summary

The scan pipeline is usable for exploration, but the current “compatibility”
logic can over-report coexistence and can produce unstable `Uc1/Uc2` boundaries.
The core numerical loop is not the main problem; phase-classification logic is.

## High-Severity Findings

### 1) Coexistence criterion is too permissive

In compatibility mode, coexistence follows:

`coexistence = (|D_m - D_i| > 1e-6) OR (|deltaF| > 1e-10)`

This can mark coexistence at many points even when branches are nearly
indistinguishable, because `|deltaF| > 1e-10` is easy to satisfy numerically.

Impact:
- “coexistence everywhere” artifacts
- unreliable coexistence maps

### 2) Boundary extraction depends on weak validity rules

In compatibility mode, branch validity is effectively “finite free energy”.
This can include branches that are converged but not physically distinct.

Impact:
- `Uc1/Uc2` may be dominated by classification thresholds
- extracted lines can pin to scan limits

## Medium-Severity Findings

### 3) Sketch-level linear algebra assumes real-valued operators

The professor sketch uses transpose `.T` in impurity free-energy construction.
That is only safe for real operators/couplings; for complex flows this is
mathematically wrong (must use conjugate transpose).

Status in your code:
- fixed in adapter (`conj().T`)

### 4) Free-energy consistency depends on `Sigma_inf` state handling

If `GhostDMFT_M.ghost_dmft()` does not update model state, free-energy terms can
be evaluated with stale `Sigma_inf`.

Status in your code:
- safeguarded by passing `sigma_inf` from result when available

## Low-Severity Findings

### 5) Performance and scaling limits in scan driver

Impurity and gateway/lattice free energies are recomputed point-by-point with no
caching. This is acceptable for small exploratory grids but expensive at quality
resolution (`M=2` / dense `(U,T)`).

### 6) Test coverage is smoke-level for phase boundaries

Current tests verify API/smoke behavior, not phase-boundary stability quality.

## What Is Solid

- Core two-ghost Variant B loop contains causality checks, bounded matching,
  and residual diagnostics.
- Adapter matches the professor script contract and free-energy structure.
- Reproducible outputs are now structured by run directories.

## Practical Recommendation

Use two modes explicitly:

- `compat_mode=True`
  - purpose: reproduce professor sketch behavior as a baseline
- `compat_mode=False`
  - purpose: production-quality phase diagram with strict branch validation and
    explicit branch-separation thresholds

For final figures:
- start from `M=2`
- use denser `U` grids near candidate transitions
- report boundary uncertainty from grid resolution

