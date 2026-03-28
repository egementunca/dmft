#!/bin/bash -l
#$ -N bond_m2_internal_gpu
#$ -o logs/bond_m2_internal_gpu_$JOB_ID.out
#$ -e logs/bond_m2_internal_gpu_$JOB_ID.err
#$ -l h_rt=04:00:00
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

# Internal codebase (src/dmft/bond_ed.py) with GPU-accelerated build_H2
python3 scripts/run_bond_sweep.py \
  --M 2 --U 1.3 --t 0.5 \
  --mode both --nk 30 --verbose \
  --out ghost_dmft_square_M2_U1.3_t0.5_both_INTERNAL_GPU.dat
