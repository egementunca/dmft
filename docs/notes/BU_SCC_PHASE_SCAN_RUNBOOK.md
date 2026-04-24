# BU SCC Runbook: Two-Ghost DMFT Phase Scan

This is a preparation guide for running your phase-scan workflow on BU SCC.
It is written as a practical template and can be adjusted once you pick exact
SCC partition/module details.

## Goals

- Run reproducible phase scans on cluster resources.
- Keep local and cluster outputs consistent.
- Avoid interactive/manual drift in solver environment.

## Assumptions

- Repository path on SCC: `$HOME/dmft` (adjust if needed).
- Python project uses `src/` layout.
- Phase scan entry point is `scripts/run_phase_scan.py`.
- Output path is under `diagnostics/phase_scan/runs/<run_name>/`.

## Recommended Repo Layout for Cluster Runs

- code: `src/dmft/`, `scripts/`
- configs: one YAML or CLI argument block per run (future improvement)
- outputs:
  - `diagnostics/phase_scan/runs/<run_name>/phase_scan.csv`
  - `diagnostics/phase_scan/runs/<run_name>/phase_boundaries.csv`
  - plot PNGs
- logs:
  - `diagnostics/phase_scan/runs/<run_name>/stdout.log`
  - `diagnostics/phase_scan/runs/<run_name>/stderr.log`

## One-Time Environment Setup (SCC)

Example bootstrap:

```bash
cd $HOME
git clone <your-private-repo-url> dmft
cd dmft

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

If SCC requires modules first, prepend:

```bash
module purge
module load python/3.x
```

## Single Interactive Sanity Run

```bash
cd $HOME/dmft
source .venv/bin/activate
export PYTHONPATH=src
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig

python3 scripts/run_phase_scan.py \
  --nu 6 --nt 3 --n-iw 128 --maxiter 60 \
  --outprefix diagnostics/phase_scan/runs/scc_smoke
```

## Batch Job Template (SGE)

BU SCC uses SGE (`qsub`), not SLURM. Ready-to-use templates exist in the repo:

- `jobs/phase_scan_m1_baseline.sh`
- `jobs/phase_scan_m2_quality.sh`

Example structure (quality job):

```bash
#!/bin/bash -l
#$ -N dmft_m2_quality
#$ -o logs/dmft_m2_quality_$JOB_ID.out
#$ -e logs/dmft_m2_quality_$JOB_ID.err
#$ -l h_rt=24:00:00
#$ -pe omp 4
#$ -l mem_per_core=4G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
source .venv/bin/activate
export PYTHONPATH=src
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

RUN_NAME=2026-03-20_m2_quality
OUT=diagnostics/phase_scan/runs/$RUN_NAME
mkdir -p "$OUT"

python3 scripts/run_phase_scan.py \
  --u-min 2.0 --u-max 3.4 --nu 30 \
  --t-min 0.02 --t-max 0.20 --nt 20 \
  --M 2 --n-iw 512 --mix 0.2 --tol 1e-6 --maxiter 180 \
  --require-converged --no-compat-mode \
  --outprefix "$OUT/main"
```

Submit:

```bash
qsub jobs/phase_scan_m2_quality.sh
```

## Monitoring and Collection

Monitor:

```bash
qstat -u $USER                 # running/queued jobs
qstat -j <JOBID>               # detailed job info
```

Inspect logs:

```bash
tail -n 100 logs/dmft_m2_quality_<JOBID>.out
tail -n 100 logs/dmft_m2_quality_<JOBID>.err
```

Pull results locally:

```bash
rsync -avz <user>@scc1.bu.edu:$HOME/dmft/diagnostics/phase_scan/runs/ ./diagnostics/phase_scan/runs/
```

## Quality-Tier Run Plan

Tier 0 (smoke):
- `M=1`, small grid, low `n_iw`
- objective: environment + file outputs

Tier 1 (baseline):
- `M=1`, dense grid
- objective: compare with professor sketch behavior

Tier 2 (quality target):
- `M=2`, dense grid, stronger convergence
- objective: publishable `Uc1/Uc2/Uc(T)` trends

Tier 3 (stability check):
- rerun a subset with slightly different mixing / `n_iw`
- objective: estimate boundary sensitivity

## Known Risks to Watch

- “Coexistence everywhere” from permissive classification logic in compat mode.
- Boundary lines pinned to scan range endpoints.
- Excessive runtime at `M=2` if grid is too dense initially.

## Future Improvements (when ready)

- Add checkpoint/restart per temperature.
- Add structured run config files (YAML/JSON) for exact provenance.
- Split `U` windows across job arrays for faster wall-clock throughput.
- Auto-generate a per-run summary markdown after each batch completion.
