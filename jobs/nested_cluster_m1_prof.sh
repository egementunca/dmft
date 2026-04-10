#!/bin/bash -l
#$ -N nc_m1_prof
#$ -o logs/nc_m1_prof_$JOB_ID.out
#$ -e logs/nc_m1_prof_$JOB_ID.err
#$ -l h_rt=12:00:00
#$ -pe omp 2
#$ -l mem_per_core=2G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
source .venv/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

# Professor's standalone nested cluster code, M=1
python3 ghost_cluster_standalone.py \
  --M 1 --U 1.3 --nquad 50 \
  --nT 100 --T_max 2.0 --T_min 0.05 \
  --mix 0.1 --maxiter 5000 --tol 1e-9 \
  --sweep --verbose
