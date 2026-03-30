# Ghost DMFT Bond Scheme — Run Instructions

## Code Versions

There are two runnable implementations of the **corrected** bond scheme:

| Implementation | Entry point | Description |
|----------------|------------|-------------|
| **Internal** (recommended) | `scripts/run_bond_sweep.py` | Refactored into `src/dmft/`. Vectorized lattice, sector-blocked ED, operator caching. |
| **Professor's new** | `ghost_dmft_bond_new.py` | Professor's corrected standalone script. Full Fock space ED, loop-based lattice. |

Both implement the same physics (direct matching, alternating loop, dmu bisection, independent M per family). Results match to ~1e-9.

> **Legacy scripts** (`ghost_dmft_bond_opt_ORIGINAL.py`, `ghost_dmft_bond_opt-copy.py`, `*_gpu.py` variants) contain the old buggy code (BPK combination matching, wrong gamma_k, single M). Do not use for new runs.

## Parameters

- `M1g, M2g, Mbg` — g-ghost counts (M1h=M2h=Mbh=1 fixed)
- Default: `U=1.3, t=0.5, nk=30, 30 T points from 0.5 down to 0.02`

## Resource Requirements

| Config | Code | Cores | RAM | Wall time | GPU |
|--------|------|-------|-----|-----------|-----|
| M=1 internal | `run_bond_sweep.py` | 2 | 4 GB | 2 hrs | No |
| M=1 prof new | `ghost_dmft_bond_new.py` | 2 | 4 GB | 4 hrs | No |
| M=2 internal CPU | `run_bond_sweep.py` | 4 | 16 GB | 48 hrs | No |
| M=2 internal GPU | `run_bond_sweep.py` | 4 | 16 GB | 4 hrs | A100 |
| M=2 prof new | `ghost_dmft_bond_new.py` | — | ~35 GB | — | **Infeasible** |

M=2 with the professor's script requires ~35 GB RAM for dense 16384x16384 Fock-space matrices. Only the internal code (sector-blocked) can run M=2.

## Submit on SCC

```bash
cd $HOME/dmft
mkdir -p logs

# M=1: internal vs professor's new (comparison)
qsub jobs/bond_m1_internal.sh
qsub jobs/bond_m1_prof_new.sh

# M=2: internal only (professor's script can't handle M=2)
qsub jobs/bond_m2_internal.sh        # CPU, 48 hrs
qsub jobs/bond_m2_internal_gpu.sh    # GPU, 4 hrs (needs CuPy, see GPU_CLUSTER_README.md)
```

## Monitor

```bash
qstat -u $USER
tail -f logs/bond_m1_internal_<JOBID>.out
```

## Output

Internal code saves to `bond_M1g<X>M2g<Y>Mbg<Z>_U<U>[_tag].dat`.
Professor's script saves to `bond_M1g<X>M2g<Y>Mbg<Z>_U<U>[_tag].dat`.

Columns: `T  docc_ss  docc_bpk  docc1  docc2  hop`
