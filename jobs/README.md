# Job Templates

SGE templates for BU SCC runs.

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

Parameters (matching professor's): `--mix 0.1 --maxiter 5000 --nT 100 --T 2.0→0.05 --nquad 50`

| Job | Code | M | GPU? | Wall time |
|-----|------|---|------|-----------|
| `nested_cluster_m1_prof.sh` | Professor's standalone | 1 | No | 12h |
| `nested_cluster_m1.sh` | Internal (sector-blocked ED) | 1 | No | 6h |
| `nested_cluster_m1_gpu.sh` | Internal (sector-blocked ED) | 1 | Yes | 4h |
| `nested_cluster_m2.sh` | Internal (sector-blocked ED) | 2 | No | 48h |
| `nested_cluster_m2_gpu.sh` | Internal (sector-blocked ED) | 2 | Yes | 36h |

GPU jobs use the CuPy / `_eigh` dispatch in `src/dmft/bond_ed.py`. See
[GPU_CLUSTER_README.md](GPU_CLUSTER_README.md) for one-time CuPy setup.

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
