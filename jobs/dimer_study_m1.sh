#!/bin/bash -l
#$ -N dimer_study_m1
#$ -o logs/dimer_study_m1_$JOB_ID.out
#$ -e logs/dimer_study_m1_$JOB_ID.err
#$ -l h_rt=06:00:00
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

# Dimer ghost-DMFT phase diagram study, M=1 (fast baseline)
# Expected: 2-4 hours
python3 run_ghost_dmft_study.py --M 1
