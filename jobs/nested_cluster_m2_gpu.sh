#!/bin/bash -l
#$ -N nc_m2_gpu
#$ -o logs/nc_m2_gpu_$JOB_ID.out
#$ -e logs/nc_m2_gpu_$JOB_ID.err
#$ -l h_rt=36:00:00
#$ -pe omp 4
#$ -l mem_per_core=4G
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
module load cuda/12.2
source .venv/bin/activate
export PYTHONPATH=src

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

python3 -c "import cupy; cupy.cuda.Device(0).use(); print('CuPy OK:', cupy.__version__)"

# Ghost Nested Cluster, M=2, GPU-accelerated
# Matches professor's parameters: mix=0.1, maxiter=5000, nT=100, T=2.0->0.05
python3 scripts/run_nested_cluster.py \
  --M 2 --U 1.3 --nquad 50 \
  --nT 100 --T_max 2.0 --T_min 0.05 \
  --mix 0.1 --maxiter 5000 --tol 1e-9 \
  --sweep --verbose
