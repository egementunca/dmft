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

RUN=2026-03-20_m2_quality
OUT=diagnostics/phase_scan/runs/$RUN/main
mkdir -p diagnostics/phase_scan/runs/$RUN

python3 scripts/run_phase_scan.py \
  --u-min 2.0 --u-max 3.4 --nu 30 \
  --t-min 0.02 --t-max 0.20 --nt 20 \
  --M 2 --n-iw 512 --mix 0.2 --tol 1e-6 --maxiter 180 \
  --require-converged --no-compat-mode \
  --use-branch-filters --z-metal-min 0.10 --z-ins-max 0.08 --v-ins-max 0.28 \
  --coexist-z-tol 1e-2 --coexist-docc-tol 2e-3 \
  --outprefix "$OUT"
