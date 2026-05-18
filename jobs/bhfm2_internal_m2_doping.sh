#!/bin/bash -l
#$ -N bhfm2_int_m2_doping
#$ -o logs/bhfm2_int_m2_doping_$JOB_ID.out
#$ -e logs/bhfm2_int_m2_doping_$JOB_ID.err
#$ -l h_rt=36:00:00
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

# Internal parity runner (mirrors BHFM2 run_M2_doping.py Ferrero grid)
python3 scripts/run_bhfm2_minimal.py \
  --mode doping \
  --M 2 --Mb 1 \
  --U 1.25 --t 0.5 --z 0.5 \
  --Nk 16 --n-moments 8 \
  --fillings 0.95,0.90,0.85 \
  --temps 0.5,0.3,0.2,0.1,0.05 \
  --max-chunks 10 \
  --outdir results/bhfm2_minimal/doping_u1p25 \
  --results-file doping_M2_U1.25.pkl \
  --params-file doping_M2_U1.25_params.pkl

