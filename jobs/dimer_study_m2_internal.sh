#!/bin/bash -l
#$ -N dimer_m2_internal
#$ -o logs/dimer_m2_internal_$JOB_ID.out
#$ -e logs/dimer_m2_internal_$JOB_ID.err
#$ -l h_rt=24:00:00
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

# Optimized dimer ghost-DMFT study, M=2 (sector-blocked ED)
# Compare output against professor's run: jobs/dimer_study_m2.sh
python3 scripts/run_dimer_study.py --M 2
