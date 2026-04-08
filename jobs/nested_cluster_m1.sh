#!/bin/bash -l
#$ -N nc_m1
#$ -o logs/nc_m1_$JOB_ID.out
#$ -e logs/nc_m1_$JOB_ID.err
#$ -l h_rt=04:00:00
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

# Ghost Nested Cluster, M=1, internal optimized code
python3 scripts/run_nested_cluster.py --M 1 --U 1.3 --nquad 50 --verbose
