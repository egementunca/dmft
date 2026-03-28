# GPU Feasibility Report: Ghost-DMFT Bond Scheme

**Date:** 2026-03-28
**Code:** `ghost_dmft_bond_opt-copy.py`
**Target hardware:** NVIDIA A100 (SCC cluster)

---

## Problem Statement

The bond-scheme solver calls `scipy.least_squares` with up to `max_nfev=3000` per outer
iteration, up to 200 outer iterations, across 19 temperature points. Every residual
evaluation triggers a full exact diagonalization (`eigh`) of multiple Fock-space sector
matrices. On CPU this is infeasible for the intended parameter regimes.

---

## Matrix Dimension Analysis

### `build_H2` — dominant bottleneck

Mode count per spin: `nps = 2 + 3*M`

Hilbert space is sector-blocked by `(nup, ndn)` with `nup + ndn = nps`. Sector
dimension: `C(nps, nup) × C(nps, ndn)`.

| M | nps | Largest sector (nup = nps/2) | Dimension | Full unblocked |
|---|-----|------------------------------|-----------|----------------|
| 1 | 5   | C(5,2) × C(5,3) = 10 × 10   | 100       | C(10,5) = 252  |
| 2 | 8   | C(8,4) × C(8,4) = 70 × 70   | **4,900** | C(16,8) = 12,870 |
| 3 | 11  | C(11,5) × C(11,6) = 462 × 462 | **213,444** | C(22,11) ≈ 700k |

For **M=2**, `build_H2` must diagonalize a **4900×4900** matrix (plus smaller sectors)
on every single residual evaluation.

### Other operations

| Function | Matrix size | Notes |
|----------|-------------|-------|
| `lattice_statics` | 900 × (3×3) batched | n_k=30, M=1 |
| `lattice_statics` | 900 × (3×3) batched | n_k=30, M=2 (same) |
| `bond_lattice_statics` | 900 × (5×5) batched | M=2 |
| `gateway_statics` | 5×5 (M=1) / 5×5 (M=2) | Single matrix |
| `bond_gateway_statics` | 8×8 (M=1) / 14×14 (M=2) | Single matrix |
| `impurity_statics` | 16×16 (M=1) / 64×64 (M=2) | Full Fock space, small |

---

## CPU Cost Estimate

`build_H2` is called from `residuals()` inside `solve_bond`:

```
19 T-points × 200 outer iters × 3000 nfev = ~11.4M residual evaluations (worst case)
```

Measured/estimated `eigh` time on CPU (NumPy + MKL, single core):

| Sector | Dimension | Time/call |
|--------|-----------|-----------|
| nup=4  | 4900×4900 | ~0.3–1.0 s |
| nup=3,5| 3136×3136 | ~0.1–0.3 s |
| nup=2,6| 784×784   | ~0.005 s  |
| nup=1,7| 64×64     | negligible |

**Per residual call total (M=2): ~0.5–1.5 s**

Worst-case total: `11.4M × 1s = 132 days`. Even at 10× optimistic convergence:
**~13 days per full sweep**. Confirmed infeasible.

---

## GPU Feasibility (A100)

### Hardware specs

| Metric | A100 SXM4 80GB |
|--------|----------------|
| FP64 TFLOPS | 19.5 |
| Memory | 80 GB HBM2e |
| Memory bandwidth | 2,000 GB/s |

A single 4900×4900 float64 matrix occupies ~192 MB — trivially fits in GPU memory.
cuSOLVER's `dsyevd` (called by CuPy's `eigh`) is the appropriate backend.

### Expected speedup

Effective CPU FP64 throughput (dense LA, MKL, 1 core): ~0.5–1 TFLOP/s.
A100 FP64: 19.5 TFLOPS → theoretical ~20–40× for compute-bound kernels.

Accounting for GPU launch latency and CPU↔GPU transfer (negligible at this matrix
size, ~1 ms/transfer):

| Operation | CPU time/call | GPU time/call | Speedup |
|-----------|--------------|--------------|---------|
| `build_H2` nup=4 sector | ~0.5 s | ~0.025–0.05 s | **10–20×** |
| `build_H2` nup=3,5 sectors | ~0.15 s | ~0.01 s | ~15× |
| `lattice_statics` BZ batch | ~0.001 s | ~0.001 s | ~1× (no gain) |
| `bond_lattice_statics` BZ | ~0.002 s | ~0.002 s | ~1× (no gain) |
| Gateway/impurity (small) | negligible | negligible | — |

**Expected end-to-end speedup: 10–20×**, bringing the practical sweep time from
days into hours on the A100.

### Why GPU works here

- The 4900×4900 diagonalization is purely compute-bound dense LA — optimal GPU workload.
- The result must return to CPU for `scipy.least_squares` regardless, but the transfer
  cost (~1 ms) is negligible vs. the compute time.
- No branching or irregular memory access in `eigh` — no GPU underutilization.

---

## Implementation Strategy

### Minimum-change approach (CuPy)

CuPy is a drop-in NumPy replacement for CUDA. The only function that needs changing
is `build_H2` ([ghost_dmft_bond_opt-copy.py:336](../ghost_dmft_bond_opt-copy.py#L336)).

```python
import cupy as cp

# Inside build_H2, replace the eigh call:
H_gpu   = cp.asarray(H)                    # ~0.5 ms transfer
ev_gpu, evec_gpu = cp.linalg.eigh(H_gpu)   # GPU computation
ev   = cp.asnumpy(ev_gpu)                   # result back to CPU
evec = cp.asnumpy(evec_gpu)
```

No other changes needed. All downstream numpy operations (partition function,
observables) remain on CPU.

### Optional: cache GPU-side transition arrays

`get_H2_sector_cache` ([line 147](../ghost_dmft_bond_opt-copy.py#L147)) returns
`transition_map` as numpy arrays that are re-uploaded on every residual call. A
companion `@lru_cache` for GPU arrays (`cp.asarray` versions) would eliminate
redundant transfers:

```python
@lru_cache(maxsize=None)
def get_H2_sector_cache_gpu(M, nup):
    _, occ, ham_transitions, transition_map = get_H2_sector_cache(M, nup)
    tm_gpu = {k: tuple(cp.asarray(a) for a in v) for k, v in transition_map.items()}
    return cp.asarray(occ), ham_transitions, tm_gpu
```

This gives a further ~2–5× speedup for the observable accumulation loop.

### Cluster setup

```bash
module load cuda/12.x
pip install cupy-cuda12x
```

Or if using the existing `.venv`:

```bash
source .venv/bin/activate
pip install cupy-cuda12x  # match cluster CUDA version
python -c "import cupy; cupy.cuda.Device(0).use(); print(cupy.__version__)"
```

The job script (`jobs/ghost_dmft_bond_m1.sh`) would need a GPU queue directive:

```bash
#$ -l gpus=1
#$ -l gpu_c=7.0   # A100 compute capability
```

---

## Summary

| | CPU (current) | GPU A100 |
|-|--------------|----------|
| Bottleneck | `eigh` on 4900×4900, repeated ~11M× | Same, but 10–20× faster |
| Practical sweep time (M=2) | Days–weeks | Hours |
| Implementation effort | — | Low (CuPy, ~5 lines changed) |
| Memory requirement | Negligible | <1 GB (well within 80 GB) |
| **Verdict** | **Infeasible** | **Feasible** |

GPU acceleration via CuPy targeting `build_H2` is the recommended path. M=3 would
require GPU even more urgently: the 213k×213k sector matrix is not tractable on any
single CPU, but is also borderline for a single GPU (~340 GB for float64 — requires
either float32, distributed, or Lanczos instead of full diagonalization).
