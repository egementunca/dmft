#!/bin/bash -l
#$ -N bhfm2_m2_sweep
#$ -o logs/bhfm2_m2_sweep_$JOB_ID.out
#$ -e logs/bhfm2_m2_sweep_$JOB_ID.err
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

# BHFM2 M=2 minimal T-sweep (professor's formulation, no anti-bond)
# U=1.3, half-filling, T from 2.5 down to 0.2
# Checkpoints per T-point in BHFM2/ — safe to resume after interruption
python3 BHFM2/run_M2_sweep.py
