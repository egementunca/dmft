#!/bin/bash -l
#$ -N gdmft_m2_mine
#$ -o logs/gdmft_bond_m2_mine_$JOB_ID.out
#$ -e logs/gdmft_bond_m2_mine_$JOB_ID.err
#$ -l h_rt=48:00:00
#$ -pe omp 4
#$ -l mem_per_core=4G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
source .venv/bin/activate
export PYTHONPATH=src

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

# User's own codebase implementation (src/dmft/bond.py)
python3 scripts/run_bond_sweep.py \
  --M 2 --U 1.3 --t 0.5 \
  --mode both --nk 30 --verbose \
  --out ghost_dmft_square_M2_U1.3_t0.5_both_MINE.dat
