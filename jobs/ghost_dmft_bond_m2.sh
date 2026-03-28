#!/bin/bash
#SBATCH --job-name=gdmft_bond_m2
#SBATCH --output=logs/gdmft_bond_m2_%j.out
#SBATCH --error=logs/gdmft_bond_m2_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

cd $HOME/dmft
source .venv/bin/activate

# Multi-threaded BLAS for 4900x4900 eigh (M=2 block-diag ED)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

# Professor's parameters: M=2, U=1.3, t=0.5
python3 ghost_dmft_bond_opt-copy.py \
  --M 2 --U 1.3 --t 0.5 \
  --mode both --nk 30 --verbose \
  --out ghost_dmft_square_M2_U1.3_t0.5_both.dat
