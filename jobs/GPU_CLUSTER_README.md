# GPU Job Guide — Ghost-DMFT Bond Scheme (A100)

## Why GPU

The bond solver's `impurity2_statics` diagonalizes (nup, ndn) sectors of the
two-site interacting cluster. For M=2 the largest sector is ~1225x1225.
GPU acceleration via CuPy/cuSOLVER speeds up sectors with dimension >= 256.

| Backend | ~1225x1225 eigh | Expected M=2 sweep (30 T pts) |
|---------|-----------------|-------------------------------|
| CPU (NumPy/MKL, 4 cores) | ~10-50 ms | 2-8 hrs |
| A100 GPU (CuPy/cuSOLVER) | ~3-10 ms | 1-3 hrs |

Only `impurity2_statics` sector diagonalizations are sent to GPU.
All other operations (BZ sums, gateway matrices) stay on CPU.

---

## One-time setup: install CuPy into the project venv

Check which CUDA version is available on the node you will run on:

```bash
module load cuda/12.2
nvcc --version
```

Then install the matching CuPy wheel:

```bash
module load python3/3.10.12
module load cuda/12.2
source $HOME/dmft/.venv/bin/activate
pip install cupy-cuda12x
```

Verify:

```bash
python3 -c "import cupy; cupy.cuda.Device(0).use(); print(cupy.__version__)"
```

If the SCC cuda module version differs (e.g. `cuda/11.8`), install
`cupy-cuda11x` instead and update the `module load cuda/...` line in
the job script.

---

## Submitting the GPU job

```bash
cd $HOME/dmft
qsub jobs/bond_m2_internal_gpu.sh
```

The script:
- Requests 1 GPU (`-l gpus=1`) with compute capability >= 8.0 (`-l gpu_c=8.0`),
  which targets A100 (cc 8.0) and H100 (cc 9.0).
- Runs a CuPy smoke-test before the main script to catch setup errors early.
- Log: `logs/bond_m2_internal_gpu_<JOBID>.out`

---

## Forcing CPU fallback

Pass `--no-gpu` to disable GPU dispatch:

```bash
python3 scripts/run_bond_sweep.py --M1g 2 --M2g 2 --Mbg 1 --U 1.3 --no-gpu
```

CuPy does not need to be installed for `--no-gpu` mode.

---

## Current job scripts

| Script | Code | M | GPU? | Wall time |
|--------|------|---|------|-----------|
| `bond_m1_internal.sh` | `scripts/run_bond_sweep.py` | 1 | No | 2h |
| `bond_m1_prof_new.sh` | `ghost_dmft_bond_new.py` | 1 | No | 4h |
| `bond_m2_internal.sh` | `scripts/run_bond_sweep.py` | 2 | No | 48h |
| `bond_m2_internal_gpu.sh` | `scripts/run_bond_sweep.py` | 2 | Yes | 4h |
| `nested_cluster_m1_gpu.sh` | `scripts/run_nested_cluster.py` | 1 | Yes | 4h |
| `nested_cluster_m2_gpu.sh` | `scripts/run_nested_cluster.py` | 2 | Yes | 36h |

The nested-cluster GPU jobs use the same CuPy / `_eigh` dispatch in `src/dmft/bond_ed.py`.
Sectors below dim 256 fall back to NumPy automatically.

### Legacy (old buggy code, do not use for new runs)

| Script | Code | Notes |
|--------|------|-------|
| `bond_m[12]_prof_orig.sh` | `ghost_dmft_bond_opt_ORIGINAL.py` | BPK matching, wrong gamma_k |
| `bond_m[12]_prof_patched.sh` | `ghost_dmft_bond_opt-copy.py` | Same bugs + temp fix |
| `bond_m2_prof_*_gpu.sh` | `*_gpu.py` variants | GPU versions of legacy code |

---

## Checking job status and output

```bash
qstat -u $USER                                   # running/queued jobs
qstat -j <JOBID>                                 # detailed job info
tail -f logs/bond_m2_internal_gpu_<JOBID>.out    # live output
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: cupy` | Run the pip install step above |
| `CUDARuntimeError: cudaErrorNoDevice` | Node has no GPU; resubmit or check `-l gpus=1` |
| CUDA/CuPy version mismatch at import | Match `cupy-cudaXXx` wheel to `module load cuda/XX.Y` |
| `gpu_c` requirement not satisfied | A100 is cc 8.0; lower to `7.0` to allow V100 nodes too |
| Job killed (out of time) | Increase `h_rt` in the job script |
