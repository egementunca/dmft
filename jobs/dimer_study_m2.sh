#!/bin/bash -l
#$ -N dimer_study_m2
#$ -o logs/dimer_study_m2_$JOB_ID.out
#$ -e logs/dimer_study_m2_$JOB_ID.err
#$ -l h_rt=24:00:00
#$ -pe omp 4
#$ -l mem_per_core=4G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
source .venv/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

# Dimer ghost-DMFT phase diagram study, M=2
# 9 U values x 9 fillings x 24 T points = 1944 DMFT solves
# Impurity dim = 4^6 = 4096 (with N,Sz sector blocking)
# Expected: 8-16 hours
python3 run_ghost_dmft_study.py --M 2
