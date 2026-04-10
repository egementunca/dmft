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
qsub jobs/nested_cluster_m1.sh          # M=1, ~2-6 hrs
qsub jobs/nested_cluster_m2.sh          # M=2, ~24-48 hrs
```

Parameters (matching professor's): `--mix 0.1 --maxiter 5000 --nT 100 --T 2.0→0.05 --nquad 50`

| Job | Code | M | Wall time |
|-----|------|---|-----------|
| `nested_cluster_m1_prof.sh` | Professor's standalone | 1 | 12h |
| `nested_cluster_m1.sh` | Internal (sector-blocked ED) | 1 | 6h |
| `nested_cluster_m2.sh` | Internal (sector-blocked ED) | 2 | 48h |

## Dimer Ghost-DMFT (superseded)

```bash
qsub jobs/dimer_study_m1.sh             # professor's, M=1
qsub jobs/dimer_study_m1_internal.sh     # internal, M=1
```

## Bond Scheme (March 2026)

See [RUN_BOND_INSTRUCTIONS.md](RUN_BOND_INSTRUCTIONS.md).

```bash
qsub jobs/bond_m1_internal.sh
qsub jobs/bond_m2_internal.sh           # CPU, 48 hrs
qsub jobs/bond_m2_internal_gpu.sh       # GPU, 4 hrs
```

## Setup

- Python venv: `source $HOME/dmft/.venv/bin/activate`
- Dependencies: `pip install numpy scipy matplotlib`
