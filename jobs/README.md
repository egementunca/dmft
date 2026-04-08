# Job Templates

SGE templates for BU SCC runs.

## Ghost Nested Cluster (April 2026, current)

Latest scheme from the professor. BPK combination of single-site + two-site clusters with moment conditions.

```bash
cd $HOME/dmft && git pull && mkdir -p logs

# Internal optimized code (sector-blocked ED)
qsub jobs/nested_cluster_m1.sh    # M=1, ~2-4 hrs
qsub jobs/nested_cluster_m2.sh    # M=2, ~8-24 hrs
```

| Job | M | Wall time | Entry point |
|-----|---|-----------|-------------|
| `nested_cluster_m1.sh` | 1 | 4h | `scripts/run_nested_cluster.py` |
| `nested_cluster_m2.sh` | 2 | 24h | `scripts/run_nested_cluster.py` |

## Dimer Ghost-DMFT (April 2026, superseded)

Phase diagram scan over (U, n_dimer) grid. Superseded by nested cluster scheme.

```bash
# Professor's standalone code (baseline)
qsub jobs/dimer_study_m1.sh
qsub jobs/dimer_study_m2.sh

# Internal optimized code (comparison)
qsub jobs/dimer_study_m1_internal.sh
qsub jobs/dimer_study_m2_internal.sh
```

## Bond Scheme (March 2026)

See [RUN_BOND_INSTRUCTIONS.md](RUN_BOND_INSTRUCTIONS.md) for full details.

```bash
qsub jobs/bond_m1_internal.sh
qsub jobs/bond_m2_internal.sh        # CPU, 48 hrs
qsub jobs/bond_m2_internal_gpu.sh    # GPU, 4 hrs (see GPU_CLUSTER_README.md)
```

## Phase Scan

```bash
qsub jobs/phase_scan_m1_baseline.sh
qsub jobs/phase_scan_m2_quality.sh
```

## Setup

Before submitting, ensure:
- Python venv exists: `source $HOME/dmft/.venv/bin/activate`
- Dependencies installed: `pip install numpy scipy matplotlib`
