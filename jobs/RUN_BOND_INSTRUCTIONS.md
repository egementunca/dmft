# Ghost DMFT Bond Scheme — Run Instructions

## Script

[`ghost_dmft_bond_opt-copy.py`](../ghost_dmft_bond_opt-copy.py) — Professor's standalone Ghost-DMFT with bond scheme on a square lattice.

**Parameters from professor:** `M=2, U=1.3, t=0.5`

## Resource Requirements

| Resource | Value |
|----------|-------|
| Cores | 4 (for BLAS threading on 4900×4900 ED blocks) |
| RAM | 8 GB |
| Wall time | 6–8 hrs expected, 12 hrs allocated |
| Jobs | 1 |

## Submit on SCC

```bash
cd $HOME/dmft
mkdir -p logs
sbatch jobs/ghost_dmft_bond_m2.sh
```

## Monitor

```bash
squeue -u $USER
tail -f logs/gdmft_bond_m2_<JOBID>.out
```

## Output

Results saved to `ghost_dmft_square_M2_U1.3_t0.5_both.dat` in the working directory.

Columns: `T  T/D  beta  docc_ss  docc_bpk  docc_2  docc_1  hop  dmu  res`

## Quick Local Test (M=1, fast)

```bash
python3 ghost_dmft_bond_opt-copy.py --M 1 --U 1.3 --t 0.5 --mode both
```

This runs in ~5–15 min and validates the script works before submitting the M=2 job.
