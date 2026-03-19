#!/bin/bash
#SBATCH --job-name=dmft_m2_quality
#SBATCH --output=logs/dmft_m2_quality_%j.out
#SBATCH --error=logs/dmft_m2_quality_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd $HOME/dmft
source .venv/bin/activate
export PYTHONPATH=src
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

RUN=2026-03-19_m2_quality
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
