# Job Templates

SGE templates for BU SCC runs.

## BHFM2 Canonical + Parity (Latest)

Use this as the single run guide for current BHFM2 studies:
- [BHFM2_SCC_INSTRUCTIONS.md](BHFM2_SCC_INSTRUCTIONS.md)

Related scripts:
- Prof baseline: `bhfm2_m2_sweep.sh`, `bhfm2_m2_doping.sh`
- Internal parity: `bhfm2_internal_m2_sweep.sh`, `bhfm2_internal_m2_doping.sh`

## Ghost Nested Cluster (April 2026, current)

BPK combination of single-site + two-site clusters with moment conditions.
Run professor's code first for baseline, then our optimized version.

```bash
cd $HOME/dmft && git pull && mkdir -p logs

# Step 1: Professor's standalone code (baseline)
qsub jobs/nested_cluster_m1_prof.sh     # M=1, ~6-12 hrs

# Step 2: Our optimized internal code (comparison)
qsub jobs/nested_cluster_m1.sh          # M=1, CPU
qsub jobs/nested_cluster_m1_gpu.sh      # M=1, GPU
qsub jobs/nested_cluster_m2.sh          # M=2, CPU
qsub jobs/nested_cluster_m2_gpu.sh      # M=2, GPU
```

**Parameters:**

| Job | mix | maxiter | nT | T range | nquad |
|-----|-----|---------|-----|---------|-------|
| M=1 (all) | 0.1 | 5000 | 100 | 2.0→0.05 | 50 |
| M=2 (all) | **0.03** | 5000 | 100 | 2.0→0.05 | 50 |

> **M=2 uses mix=0.03, not 0.1.** With mix=0.1, the M=2 iteration gets stuck
> in a 2-cycle at high T (dp oscillates ~0.03 for all 5000 iterations, never
> converging). The stiffer 16-parameter landscape requires smaller damping.
> Verified locally: mix=0.03 eliminates the 2-cycle (dp~9e-5 at T=2.0) and
> keeps M=2 on the physical branch through T=0.9 without collapse.

| Job | Code | M | GPU? | Wall time |
|-----|------|---|------|-----------|
| `nested_cluster_m1_prof.sh` | Professor's standalone | 1 | No | 12h |
| `nested_cluster_m1.sh` | Internal (sector-blocked ED) | 1 | No | 6h |
| `nested_cluster_m1_gpu.sh` | Internal (sector-blocked ED) | 1 | Yes | 4h |
| `nested_cluster_m2.sh` | Internal (sector-blocked ED) | 2 | No | 48h |
| `nested_cluster_m2_gpu.sh` | Internal (sector-blocked ED) | 2 | Yes | 36h |

GPU jobs use the CuPy / `_eigh` dispatch in `src/dmft/bond_ed.py`. See
[GPU_CLUSTER_README.md](GPU_CLUSTER_README.md) for one-time CuPy setup.

### Known iteration behavior (nested cluster)

The self-consistency has two fixed points:
- **Physical** (target): docc1 ≈ 0.21 at high T, decreasing toward 0.18 at low T.
- **Trivial**: docc1 ≈ 0.249569 (bath couplings W collapsed to non-interacting limit).

What to expect in output:
- **Healthy run**: docc values decrease smoothly from ~0.21 to ~0.18 as T drops;
  dp decreases monotonically or oscillates with shrinking amplitude.
- **Basin collapse**: docc1 jumps to ~0.249569 and locks there.
  dp may then diverge (→ 50+). A divergence guard in `solve_T` will break the
  iteration at dp > 50 to prevent the warm-start for the next T from being poisoned.
- **Slow convergence** (not a failure): With mix=0.03, convergence per T is slower
  (~1000–3000 iterations at low T) but reliable. maxiter=5000 is sufficient.

M=1 stays on the physical branch down to T ≈ 0.12 (verified, job 4273379).
M=2 with mix=0.03 stays on the physical branch through at least T=0.9 (local test).
Below T ≈ 0.12 (M=1), the trivial fixed point becomes competitive; professor's
code shows the same instability there — likely a physical multi-valued region.

## Dimer Ghost-DMFT (April 2026, superseded by nested cluster)

```bash
qsub jobs/dimer_study_m1.sh             # professor's, M=1
qsub jobs/dimer_study_m1_internal.sh    # internal, M=1
qsub jobs/dimer_study_m2.sh             # professor's, M=2, 24h
qsub jobs/dimer_study_m2_internal.sh    # internal, M=2, 24h
```

## Bond Scheme (March 2026)

See [RUN_BOND_INSTRUCTIONS.md](RUN_BOND_INSTRUCTIONS.md).

```bash
qsub jobs/bond_m1_internal.sh
qsub jobs/bond_m2_internal.sh           # CPU, 48 hrs
qsub jobs/bond_m2_internal_gpu.sh       # GPU, 4 hrs
```

## Phase Scan (Two-Ghost Variant B)

See [BU_SCC_PHASE_SCAN_RUNBOOK.md](../docs/notes/BU_SCC_PHASE_SCAN_RUNBOOK.md).

```bash
qsub jobs/phase_scan_m1_baseline.sh     # M=1, dense grid, 8h
qsub jobs/phase_scan_m2_quality.sh      # M=2, dense grid, 24h
```

## Setup

- Python venv: `source $HOME/dmft/.venv/bin/activate`
- Dependencies: `pip install -e '.[dev]'` (editable install with dev extras)
- For GPU jobs: also `pip install cupy-cuda12x` after `module load cuda/12.2`
