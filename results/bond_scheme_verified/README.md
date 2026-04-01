# Ghost DMFT Bond Scheme Results

**Date:** 2026-04-01
**System:** Square lattice, U=1.3, t=0.5, half-filling
**Method:** Bond-scheme DMFT with corrected matching conditions

## Data Files

### `bond_M1_U1.3_t0.5.dat` (Low-T only)
- **Ghost counts:** M1g=1, M2g=1, Mbg=1 (M1h=1, M2h=1, Mbh=1)
- **Points:** 30 temperature points, T ∈ [0.02, 0.5]
- **Validated:** Agrees with professor's corrected code (max diff < 2×10⁻⁵)
- **Runtime:** ~5 min CPU

### `bond_M2_U1.3_t0.5.dat` (Low-T only)
- **Ghost counts:** M1g=2, M2g=2, Mbg=1 (M1h=2, M2h=2, Mbh=1)
- **Points:** 30 temperature points, T ∈ [0.02, 0.5]
- **Method:** GPU-accelerated with sector-blocking
- **Runtime:** ~47 min GPU (A40)

### `bond_M1_U1.3_t0.5_full.dat` (Complete)
- **Ghost counts:** M1g=1, M2g=1, Mbg=1 (M1h=1, M2h=1, Mbh=1)
- **Points:** 59 temperature points, T ∈ [0.02, 1.0]
- **Validated:** Complete temperature range matching professor's parameters
- **Runtime:** ~12 min CPU total

### `bond_M2_U1.3_t0.5_full.dat` (Complete)
- **Ghost counts:** M1g=2, M2g=2, Mbg=1 (M1h=2, M2h=2, Mbh=1)
- **Points:** 59 temperature points, T ∈ [0.02, 1.0]
- **Method:** GPU-accelerated with sector-blocking
- **Runtime:** ~1.5 hours GPU total

## Data Format

All files have the same column structure:

```
# T  docc_ss  docc_bpk  docc_1  docc_2  hop
```

Where:
- `T`: Temperature
- `docc_ss`: Single-site double occupancy (reference)
- `docc_bpk`: Bond-scheme double occupancy (BPK average)
- `docc_1`: Site-1 double occupancy
- `docc_2`: Site-2 double occupancy
- `hop`: Hopping parameter ⟨d₁†d₂⟩

## Parameters

All runs use:
- **U** = 1.3 (Hubbard interaction)
- **t** = 0.5 (hopping)
- **nk** = 30 (k-mesh: 30×30)
- **nT** = 30 per temperature range (59 unique points total in full datasets)
- **maxiter** = 100 (bond solver)

Comparison with professor's example:
- Professor used: nk=20, nT=10, T∈[0.1, 1.0]
- We use: nk=30 (finer k-mesh), nT=59 (finer T-mesh), T∈[0.02, 1.0]

## Key Corrections Implemented

This data includes all 5 bug fixes from the corrected bond scheme:

1. ✓ Direct matching conditions (not BPK combination)
2. ✓ Alternating outer loop with dμ Newton-bisection for half-filling
3. ✓ Independent ghost counts per family (M1g, M2g, Mbg)
4. ✓ Corrected bond form factor γₖ = εₖ/4
5. ✓ Unified lattice statics

## Convergence

- **M1 vs M2 at T=1.0:** Differ by ~1% in docc_bpk (high T, weak correlations)
- **M1 vs M2 at T=0.5:** Differ by 0.16% in docc_bpk
- **M1 vs M2 at T=0.02:** Identical (converged, strong correlations)
- **Conclusion:** M=1 is converged for this system at low T

## Code Validation

- Internal code vs professor's corrected code: **VERIFIED ✓**
- Maximum difference: 1.76×10⁻⁵ (numerical precision)
- All observables agree across all temperatures
- Overlap at T=0.5 between low-T and high-T runs: max diff 3.6×10⁻⁵ ✓

## Performance

| Config | Method | Time | Memory | Notes |
|--------|--------|------|--------|-------|
| M=1 Low-T | CPU | 5 min | 4 GB | Fast, converged |
| M=1 High-T | CPU | 7 min | 4 GB | Faster (weaker correlations) |
| M=2 Low-T | GPU | 47 min | 16 GB | Sector-blocking required |
| M=2 High-T | GPU | 45 min | 16 GB | Similar runtime |
| M=2 CPU | Failed | 16 GB | Numerical instability |
| M=2 Prof (CPU) | Timeout | 64 GB | >12 hrs, infeasible |
| M=2 Prof (GPU) | Timeout | 64 GB | >8 hrs, infeasible |

## Citation

If using this data, please cite:
- Corrected bond-scheme DMFT implementation (2026)
- Original BPK bond scheme formalism
