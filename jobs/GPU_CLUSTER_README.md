# GPU Job Guide — Ghost-DMFT Bond Scheme (A100)

## Why GPU

The bond solver calls `build_H2` inside every residual evaluation of
`scipy.least_squares`. For M=2 this diagonalizes a **4900×4900** matrix
(the half-filling sector) plus smaller sectors on every function call —
up to thousands of times per temperature point.

| Backend | 4900×4900 eigh | Expected sweep time (M=2) |
|---------|---------------|--------------------------|
| CPU (NumPy/MKL, 4 cores) | ~0.3–1 s | days |
| A100 GPU (CuPy/cuSOLVER) | ~0.03–0.05 s | hours |

Only `build_H2` sector diagonalizations are sent to GPU (dimensions ≥ 256).
All other operations (BZ sums, small gateway/impurity matrices) stay on CPU.

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
`bond_m2_prof_patched_gpu.sh`.

---

## Submitting the GPU job

```bash
cd $HOME/dmft
qsub jobs/bond_m2_prof_patched_gpu.sh
```

The script:
- Requests 1 GPU (`-l gpus=1`) with compute capability ≥ 8.0 (`-l gpu_c=8.0`),
  which targets A100 (cc 8.0) and H100 (cc 9.0).
- Runs a CuPy smoke-test before the main script to catch setup errors early.
- Writes output to `ghost_dmft_square_M2_U1.3_t0.5_both_GPU.dat`.
- Log: `logs/bond_m2_prof_patched_gpu_<JOBID>.out`

---

## Forcing CPU fallback

Pass `--no-gpu` to run the GPU script on CPU (useful for debugging or
when no GPU node is available):

```bash
python3 ghost_dmft_bond_opt_gpu.py --M 2 --U 1.3 --mode both --no-gpu
```

CuPy does not need to be installed for `--no-gpu` mode.

---

## Script inventory

| Script | Code | M | GPU? |
|--------|------|---|------|
| `bond_m1_prof_orig.sh` | `ghost_dmft_bond_opt_ORIGINAL.py` | 1 | No |
| `bond_m1_prof_patched.sh` | `ghost_dmft_bond_opt-copy.py` | 1 | No |
| `bond_m1_internal.sh` | `scripts/run_bond_sweep.py` | 1 | No |
| `bond_m2_prof_orig.sh` | `ghost_dmft_bond_opt_ORIGINAL.py` | 2 | No |
| `bond_m2_prof_patched.sh` | `ghost_dmft_bond_opt-copy.py` | 2 | No |
| `bond_m2_internal.sh` | `scripts/run_bond_sweep.py` | 2 | No |
| **`bond_m2_prof_orig_gpu.sh`** | **`ghost_dmft_bond_opt_ORIGINAL_gpu.py`** | **2** | **Yes** |
| **`bond_m2_prof_patched_gpu.sh`** | **`ghost_dmft_bond_opt_gpu.py`** | **2** | **Yes** |
| **`bond_m2_internal_gpu.sh`** | **`scripts/run_bond_sweep.py`** (via `bond_ed._init_gpu`) | **2** | **Yes** |
| `phase_scan_m1_baseline.sh` | `scripts/run_phase_scan.py` | 1 | No |
| `phase_scan_m2_quality.sh` | `scripts/run_phase_scan.py` | 2 | No |

---

## Checking job status and output

```bash
qstat -u $USER                              # running/queued jobs
qstat -j <JOBID>                            # detailed job info
tail -f logs/bond_m2_prof_patched_gpu_<JOBID>.out   # live output
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: cupy` | Run the pip install step above |
| `CUDARuntimeError: cudaErrorNoDevice` | Node has no GPU; resubmit or check `-l gpus=1` |
| CUDA/CuPy version mismatch at import | Match `cupy-cudaXXx` wheel to `module load cuda/XX.Y` |
| `gpu_c` requirement not satisfied | A100 is cc 8.0; lower to `7.0` to allow V100 nodes too |
| Job killed (out of time) | 4h limit is conservative for M=2; increase `h_rt` if needed |
