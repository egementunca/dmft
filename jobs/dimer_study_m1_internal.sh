#!/bin/bash -l
#$ -N dimer_m1_internal
#$ -o logs/dimer_m1_internal_$JOB_ID.out
#$ -e logs/dimer_m1_internal_$JOB_ID.err
#$ -l h_rt=06:00:00
#$ -pe omp 2
#$ -l mem_per_core=2G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
source .venv/bin/activate
export PYTHONPATH=src

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

# Optimized dimer ghost-DMFT study, M=1 (sector-blocked ED)
# Compare output against professor's run: jobs/dimer_study_m1.sh
python3 scripts/run_dimer_study.py --M 1
