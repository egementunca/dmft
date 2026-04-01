# Ghost DMFT Bond Scheme Results

**Date:** 2026-04-01
**System:** Square lattice, U=1.3, t=0.5, half-filling
**Method:** Bond-scheme DMFT with corrected matching conditions

## Data Files

### `bond_M1_U1.3_t0.5.dat`
- **Ghost counts:** M1g=1, M2g=1, Mbg=1 (M1h=1, M2h=1, Mbh=1)
- **Points:** 30 temperature points from T=0.5 to T=0.02
- **Validated:** Agrees with professor's corrected code to < 2×10⁻⁵
- **Runtime:** ~5 minutes on CPU

### `bond_M2_U1.3_t0.5.dat`
- **Ghost counts:** M1g=2, M2g=2, Mbg=1 (M1h=2, M2h=2, Mbh=1)
- **Points:** 30 temperature points from T=0.5 to T=0.02
- **Method:** GPU-accelerated with sector-blocking
- **Runtime:** ~47 minutes on GPU (A40)

## Data Format

Both files have the same column structure:

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

## Key Corrections Implemented

This data includes all 5 bug fixes from the corrected bond scheme:

1. ✓ Direct matching conditions (not BPK combination)
2. ✓ Alternating outer loop with dμ Newton-bisection for half-filling
3. ✓ Independent ghost counts per family (M1g, M2g, Mbg)
4. ✓ Corrected bond form factor γₖ = εₖ/4
5. ✓ Unified lattice statics

## Convergence

- **M1 vs M2 at T=0.5:** Differ by 0.16% in docc_bpk
- **M1 vs M2 at T=0.02:** Identical (converged)
- **Conclusion:** M=1 is converged for this system at low T

## Code Validation

- Internal code vs professor's corrected code: **VERIFIED ✓**
- Maximum difference: 1.76×10⁻⁵ (numerical precision)
- All observables agree across all temperatures

## Performance

| Config | Method | Time | Memory | Notes |
|--------|--------|------|--------|-------|
| M=1 | CPU | 5 min | 4 GB | Fast, converged |
| M=2 | GPU | 47 min | 16 GB | Sector-blocking required |
| M=2 | CPU | Failed | 16 GB | Numerical instability |
| M=2 Prof | N/A | >12 hrs | 64 GB | Infeasible (timeout) |

## Citation

If using this data, please cite:
- Corrected bond-scheme implementation (2026)
- Original BPK bond scheme formalism
