#!/bin/bash -l
#$ -N bond_m1_prof_new
#$ -o logs/bond_m1_prof_new_$JOB_ID.out
#$ -e logs/bond_m1_prof_new_$JOB_ID.err
#$ -l h_rt=04:00:00
#$ -pe omp 2
#$ -l mem_per_core=2G
#$ -P compcircuits
#$ -j n

set -euo pipefail

cd $HOME/dmft
module load python3/3.10.12
source .venv/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mplconfig
mkdir -p "$MPLCONFIGDIR" logs

# Professor's CORRECTED script (ghost_dmft_bond_new.py), M1g=1 M2g=1 Mbg=1
# Expected: ~30 min - 2 hrs (full Fock space, non-vectorized lattice)
# Results should match internal code (src/dmft/) to ~1e-9
python3 ghost_dmft_bond_new.py \
  --M1g 1 --M2g 1 --Mbg 1 \
  --M1h 1 --M2h 1 --Mbh 1 \
  --U 1.3 --t 0.5 \
  --nT 30 --Tmin 0.02 --Tmax 0.5 \
  --nk 30 --verbose \
  --tag M1_U1.3_t0.5_prof_new
