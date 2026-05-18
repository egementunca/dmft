#!/bin/bash -l
#$ -N bhfm2_m2_doping
#$ -o logs/bhfm2_m2_doping_$JOB_ID.out
#$ -e logs/bhfm2_m2_doping_$JOB_ID.err
#$ -l h_rt=36:00:00
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

# BHFM2 M=2 Ferrero-style doping sweep (professor's formulation)
# U/t=2.5 (U=1.25, t=0.5), fillings n=0.95/0.90/0.85, T=0.05-0.5
# 15 points total (3 fillings x 5 T), warm-started filling-by-filling
# Checkpoints per (T,n) point in BHFM2/ — safe to resume after interruption
python3 BHFM2/run_M2_doping.py
