#!/bin/bash
#SBATCH --job-name=dmft_m1_base
#SBATCH --output=logs/dmft_m1_base_%j.out
#SBATCH --error=logs/dmft_m1_base_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd $HOME/dmft
source .venv/bin/activate
export PYTHONPATH=src
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

RUN=2026-03-19_m1_baseline
OUT=diagnostics/phase_scan/runs/$RUN/main
mkdir -p diagnostics/phase_scan/runs/$RUN

python3 scripts/run_phase_scan.py \
  --u-min 2.0 --u-max 3.4 --nu 30 \
  --t-min 0.02 --t-max 0.20 --nt 20 \
  --M 1 --n-iw 512 --mix 0.5 --tol 1e-8 --maxiter 100 \
  --outprefix "$OUT"
