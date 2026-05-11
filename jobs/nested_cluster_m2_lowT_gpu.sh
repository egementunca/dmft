#!/bin/bash -l
#$ -N nc_m2_lowT_gpu
#$ -o logs/nc_m2_lowT_gpu_$JOB_ID.out
#$ -e logs/nc_m2_lowT_gpu_$JOB_ID.err
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

# Ghost Nested Cluster, M=2, GPU-accelerated, low-T range only
# Job 5386983 produced good data T=2.0→T≈1.19; collapsed at T=1.10 (dp>50).
# This job picks up from T=1.20 with a cold start (x0), skipping the
# already-good high-T points. nT=86 keeps the same log-spaced density as
# the full 100-point 2.0→0.05 sweep.
# mix=0.03: required for M=2 (16 bath params); mix=0.1 causes 2-cycle at high T.
python3 scripts/run_nested_cluster.py \
  --M 2 --U 1.3 --nquad 50 \
  --nT 86 --T_max 1.20 --T_min 0.05 \
  --mix 0.03 --maxiter 5000 --tol 1e-9 \
  --sweep --verbose
