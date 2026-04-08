#!/bin/bash -l
#$ -N nc_m2
#$ -o logs/nc_m2_$JOB_ID.out
#$ -e logs/nc_m2_$JOB_ID.err
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

# Ghost Nested Cluster, M=2, internal optimized code
python3 scripts/run_nested_cluster.py --M 2 --U 1.3 --nquad 50 --verbose
