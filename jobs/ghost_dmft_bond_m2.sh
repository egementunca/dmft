#!/bin/bash -l
#$ -N gdmft_bond_m2
#$ -o logs/gdmft_bond_m2_$JOB_ID.out
#$ -e logs/gdmft_bond_m2_$JOB_ID.err
#$ -l h_rt=12:00:00
#$ -pe omp 4
#$ -l mem_per_core=2G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
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
