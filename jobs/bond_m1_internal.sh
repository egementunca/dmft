#!/bin/bash -l
#$ -N bond_m1_internal
#$ -o logs/bond_m1_internal_$JOB_ID.out
#$ -e logs/bond_m1_internal_$JOB_ID.err
#$ -l h_rt=02:00:00
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

# Internal codebase (src/dmft/bond.py), M1g=1 M2g=1 Mbg=1
python3 scripts/run_bond_sweep.py \
  --M1g 1 --M2g 1 --Mbg 1 \
  --U 1.3 --t 0.5 \
  --nT 30 --Tmin 0.02 --Tmax 0.5 \
  --nk 30 --verbose --no-gpu \
  --tag M1_U1.3_t0.5
