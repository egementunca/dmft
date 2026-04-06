# Job Templates

SGE templates for BU SCC runs.

## Dimer Ghost-DMFT Study (April 2026)

Phase diagram scan over (U, n_dimer) grid. Run professor's code first for baseline, then our optimized version for comparison.

```bash
cd $HOME/dmft && git pull && mkdir -p logs

# Step 1: Professor's standalone code (baseline)
qsub jobs/dimer_study_m1.sh             # M=1, ~2-4 hrs
qsub jobs/dimer_study_m2.sh             # M=2, ~8-16 hrs (professor's request)

# Step 2: Our optimized internal code (comparison)
qsub jobs/dimer_study_m1_internal.sh    # M=1, ~1-2 hrs (sector-blocked ED)
qsub jobs/dimer_study_m2_internal.sh    # M=2, ~4-8 hrs (sector-blocked ED)
```

Output files:
- Professor's: `study_M{M}_halffill_U{U}.npy`, `study_M{M}_doped_U{U}_n{n}.npy`, `study_M{M}_summary.npy`
- Internal: same names with `_internal` suffix

| Job | Code | M | Wall time | Entry point |
|-----|------|---|-----------|-------------|
| `dimer_study_m1.sh` | Professor's standalone | 1 | 6h | `run_ghost_dmft_study.py` |
| `dimer_study_m2.sh` | Professor's standalone | 2 | 24h | `run_ghost_dmft_study.py` |
| `dimer_study_m1_internal.sh` | Internal (optimized) | 1 | 6h | `scripts/run_dimer_study.py` |
| `dimer_study_m2_internal.sh` | Internal (optimized) | 2 | 24h | `scripts/run_dimer_study.py` |

## Bond Scheme (March 2026)

See [RUN_BOND_INSTRUCTIONS.md](RUN_BOND_INSTRUCTIONS.md) for full details.

```bash
qsub jobs/bond_m1_internal.sh
qsub jobs/bond_m1_prof_new.sh
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
- Update partition/account/module lines for your SCC setup if needed
