#!/bin/bash -l
#$ -N bhfm2_int_m2_sweep
#$ -o logs/bhfm2_int_m2_sweep_$JOB_ID.out
#$ -e logs/bhfm2_int_m2_sweep_$JOB_ID.err
#$ -l h_rt=24:00:00
#$ -pe omp 4
#$ -l mem_per_core=4G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd "$HOME/dmft"
module load python3/3.10.12
source .venv/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

# Internal parity runner (mirrors BHFM2 run_M2_sweep.py target grid)
python3 scripts/run_bhfm2_minimal.py \
  --mode sweep \
  --M 2 --Mb 1 \
  --U 1.3 --t 0.5 --z 0.5 \
  --Nk 16 --n-moments 8 \
  --outdir results/bhfm2_minimal/sweep_u1p3 \
  --results-file Tsweep_M2_U1.3.pkl \
  --params-file Tsweep_M2_U1.3_params.pkl

