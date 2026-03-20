#!/bin/bash -l
#$ -N dmft_m1_base
#$ -o logs/dmft_m1_base_$JOB_ID.out
#$ -e logs/dmft_m1_base_$JOB_ID.err
#$ -l h_rt=08:00:00
#$ -pe omp 2
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

RUN=2026-03-20_m1_baseline
OUT=diagnostics/phase_scan/runs/$RUN/main
mkdir -p diagnostics/phase_scan/runs/$RUN

python3 scripts/run_phase_scan.py \
  --u-min 2.0 --u-max 3.4 --nu 30 \
  --t-min 0.02 --t-max 0.20 --nt 20 \
  --M 1 --n-iw 512 --mix 0.5 --tol 1e-8 --maxiter 100 \
  --outprefix "$OUT"
