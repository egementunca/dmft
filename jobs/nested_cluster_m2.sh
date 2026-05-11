#!/bin/bash -l
#$ -N nc_m2
#$ -o logs/nc_m2_$JOB_ID.out
#$ -e logs/nc_m2_$JOB_ID.err
#$ -l h_rt=72:00:00
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

# Ghost Nested Cluster, M=2
# mix=0.03: M=2 has 16 bath params vs M=1's 8; mix=0.1 causes a 2-cycle at
# high T that never exits (verified in job 4273760 run). Smaller mix converges.
python3 scripts/run_nested_cluster.py \
  --M 2 --U 1.3 --nquad 50 \
  --nT 100 --T_max 2.0 --T_min 0.05 \
  --mix 0.03 --maxiter 5000 --tol 1e-9 \
  --sweep --verbose
