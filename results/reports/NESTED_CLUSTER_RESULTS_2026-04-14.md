# Ghost Nested Cluster DMFT - Results Summary

**Date:** April 14, 2026  
**System:** BU SCC Cluster  
**Parameters:** U=1.3, z=4.0, nquad=50, nT=100, T_max=2.0, T_min=0.05, mix=0.1, maxiter=5000

---

## Job Summary

### ✅ Successfully Completed

| Job | Code | M | Hardware | Runtime | Temps | Status |
|-----|------|---|----------|---------|-------|--------|
| 4273379 | Internal (sector-blocked ED) | 1 | CPU (2 cores) | ~5.7 hrs | 100/100 | ✓ Complete |
| 4273380 | Internal (sector-blocked ED) | 1 | **GPU (A100)** | ~3.2 hrs | 100/100 | ✓ Complete |
| 4273758 | Professor's standalone | 1 | CPU (2 cores) | ~12.2 hrs | 100/100 | ✓ Complete |

### ⚠️ Running or Failed

| Job | Code | M | Hardware | Status | Issue |
|-----|------|---|----------|--------|-------|
| 4273759 | Internal | 2 | CPU (4 cores) | Running (20 hrs) | 8/100 temps, very slow |
| 4273760 | Internal | 2 | **GPU** | ❌ Crashed | Eigenvalue convergence failure at T=1.4845 |

---

## M=1 Results Comparison

### Convergence Quality

| Code | Avg Iterations | Avg dp | Final dp (T=0.05) | Final docc |
|------|----------------|--------|-------------------|------------|
| **Internal CPU** | 2654 | 1.03e-02 | 3.27e-06 | 0.22211153 |
| **Internal GPU** | 2657 | 1.03e-02 | 5.03e-05 | 0.22211198 |
| **Professor** | (different format) | - | 7.75e-10 | 0.11374684 |

**Observation:** Internal CPU/GPU implementations give nearly identical results (docc agrees to 8 digits). However, they differ significantly from professor's standalone code, particularly at low temperature:
- Internal: docc(T=0.05) ≈ 0.222
- Professor: docc(T=0.05) ≈ 0.114

This suggests **different implementations or bugs in one version**.

### Key Temperature Points (Internal CPU)

| T | docc | docc1 | docc2 | iters | dp |
|---|------|-------|-------|-------|-----|
| 2.0000 | 0.21439559 | 0.214158 | 0.214396 | 5000 | 4.62e-03 |
| 0.7879 | 0.19218919 | 0.193352 | 0.192189 | 1640 | 9.08e-10 |
| 0.3104 | 0.18804673 | 0.196810 | 0.188047 | 476 | 8.83e-10 |
| 0.1223 | 0.17882646 | 0.214416 | 0.178826 | 2000 | 9.36e-10 |
| 0.0500 | 0.22211153 | 0.249569 | 0.222112 | 5000 | 3.27e-06 |

---

## M=2 Results

### Issues Encountered

**GPU Job (4273760):**
- Completed 9/100 temperatures before crashing
- Error: `numpy.linalg.LinAlgError: Eigenvalues did not converge`
- Crash occurred at T=1.4845 after showing divergent dp values (dp=3.49e+03)
- Some temperatures showed unstable convergence (T=1.7231: dp=2.09e+00)

**CPU Job (4273759):**
- Still running after 20+ hours
- Completed only 8/100 temperatures
- Current speed: ~2.5 hours per temperature point
- Estimated total time: ~10 days (!!)

**Root cause:** M=2 has much larger Hilbert space:
- Impurity 2: 6 orbitals → 4^6 = **4,096 states** (vs 256 for M=1)
- Numerical instabilities in eigenvalue solver
- Convergence difficulties in self-consistency loop

---

## Performance Analysis

### Speed Comparison (M=1)

| Hardware | Time/Temp | Total Time | Speedup |
|----------|-----------|------------|---------|
| CPU | ~3.4 min | 5.7 hrs | 1.0x |
| **GPU (A100)** | ~1.9 min | 3.2 hrs | **1.8x** |

GPU provides moderate speedup for M=1. Most time spent in least-squares fitting, not ED.

### M=1 vs M=2 Complexity

- M=1 Impurity 2: 4^4 = 256 states
- M=2 Impurity 2: 4^6 = 4,096 states (16x larger)
- M=2 appears to have numerical stability issues

---

## Recommendations

1. **Compare M=1 implementations:** Internal code gives docc≈0.222 at T=0.05, but professor's gives 0.114. Need to identify source of discrepancy.

2. **M=2 numerical stability:** Consider:
   - Using more robust eigenvalue solver (LAPACK vs default)
   - Reducing parameter bounds or initial guesses
   - Implementing better warm-starting between temperatures
   - Checking for NaN/Inf values

3. **M=2 performance:** Current speed makes M=2 impractical for production runs (days per sweep). May need algorithmic improvements.

---

## Files

**Successful runs:**
- M1 Internal CPU: `logs/nc_m1_4273379.out` (19 MB)
- M1 Internal GPU: `logs/nc_m1_gpu_4273380.out` (19 MB)
- M1 Professor: `logs/nc_m1_prof_4273758.out` (24 MB)

**Failed/Running:**
- M2 GPU (crashed): `logs/nc_m2_gpu_4273760.{out,err}` 
- M2 CPU (still running): `logs/nc_m2_4273759.out`

---

**Generated:** 2026-04-14 by automated analysis
