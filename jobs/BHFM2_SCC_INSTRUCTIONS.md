# BHFM2 SCC Latest Runbook

Single source of truth for BHFM2 cluster runs.

- **Prof canonical code**: `BHFM2/*`
- **Internal parity code**: `src/dmft/bhfm2_minimal.py` + `scripts/run_bhfm2_minimal.py`
- **Date**: 2026-05-18

## 1) Target Runs

Two studies, both at `M=2, Mb=1`:

1. Half-filling temperature sweep
- `U=1.3`, `t=0.5`, `z=0.5`
- `T = 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2`

2. Ferrero-style doping sweep
- `U/t=2.5` (`U=1.25`, `t=0.5`), `z=0.5`
- fillings `n = 0.95, 0.90, 0.85`
- `T = 0.5, 0.3, 0.2, 0.1, 0.05`

## 2) Job Scripts

### Prof canonical (recommended baseline)
- `jobs/bhfm2_m2_sweep.sh`
- `jobs/bhfm2_m2_doping.sh`

### Internal parity (same grids/physics targets)
- `jobs/bhfm2_internal_m2_sweep.sh`
- `jobs/bhfm2_internal_m2_doping.sh`

## 3) Preflight (login node)

```bash
cd $HOME/dmft
module load python3/3.10.12
source .venv/bin/activate

# Prof environment/timing check
cd BHFM2
python3 check_setup.py
cd ..

# Internal parity wiring check
python3 scripts/run_bhfm2_minimal.py --mode check --M 2 --Mb 1 --U 1.3 --z 0.5 --Nk 16 --n-moments 8
```

## 4) Submit

### A) Prof-only baseline (do this first)
```bash
cd $HOME/dmft && git pull && mkdir -p logs
qsub jobs/bhfm2_m2_sweep.sh
qsub jobs/bhfm2_m2_doping.sh
```

### B) Internal-only parity
```bash
cd $HOME/dmft && git pull && mkdir -p logs
qsub jobs/bhfm2_internal_m2_sweep.sh
qsub jobs/bhfm2_internal_m2_doping.sh
```

### C) A/B comparison (if queue allocation allows)
```bash
cd $HOME/dmft && git pull && mkdir -p logs
qsub jobs/bhfm2_m2_sweep.sh
qsub jobs/bhfm2_m2_doping.sh
qsub jobs/bhfm2_internal_m2_sweep.sh
qsub jobs/bhfm2_internal_m2_doping.sh
```

## 5) Monitor

```bash
qstat -u $USER

# Prof logs
tail -f logs/bhfm2_m2_sweep_<JOBID>.out
tail -f logs/bhfm2_m2_doping_<JOBID>.out

# Internal logs
tail -f logs/bhfm2_int_m2_sweep_<JOBID>.out
tail -f logs/bhfm2_int_m2_doping_<JOBID>.out
```

## 6) Outputs

### Prof canonical outputs
- `BHFM2/Tsweep_M2_U1.3.pkl`
- `BHFM2/doping_M2_U1.25.pkl`
- checkpoints in `BHFM2/M2_T*_ckpt.pkl`, `BHFM2/M2_doped_T*_n*_ckpt.pkl`

### Internal parity outputs
- `results/bhfm2_minimal/sweep_u1p3/Tsweep_M2_U1.3.pkl`
- `results/bhfm2_minimal/doping_u1p25/doping_M2_U1.25.pkl`
- checkpoints under each run dir in `checkpoints/`

## 7) Resume Behavior

All four runners are restart-safe.

- Re-submitting the same script will skip completed points.
- Partially solved points resume from per-point checkpoints.
- No manual edit needed unless files are corrupted.

## 8) Quick Quality Gates

For each solved point, prefer:
- residual norm `||r|| < 1e-3`
- `n_avg` close to target (typically within `1e-3`)

If many points stall above `1e-2`, stop and inspect warm starts/checkpoints before submitting more jobs.

## 9) Postprocess (Prof outputs)

```bash
python3 BHFM2/plot_fermi_arcs.py BHFM2/doping_M2_U1.25.pkl
```

## 10) Cluster Agent Prompt (Use After `git pull`)

Copy this prompt to your cluster agent:

```text
You are operating on BU SCC in $HOME/dmft.
Goal: run BHFM2 canonical sweeps first, then internal parity sweeps, and report job IDs + health.

Do exactly:
1) cd $HOME/dmft && git pull && mkdir -p logs
2) module load python3/3.10.12
3) source .venv/bin/activate
4) cd BHFM2 && python3 check_setup.py && cd ..
5) python3 scripts/run_bhfm2_minimal.py --mode check --M 2 --Mb 1 --U 1.3 --z 0.5 --Nk 16 --n-moments 8
6) Submit prof jobs:
   qsub jobs/bhfm2_m2_sweep.sh
   qsub jobs/bhfm2_m2_doping.sh
7) Wait 20 seconds, then run qstat -u $USER and capture job IDs/states.
8) Tail first 40 lines for each submitted job log (if file exists) and report:
   - whether runner started
   - first residual/chunk lines
   - any import/runtime errors
9) If prof jobs start cleanly, submit internal parity jobs:
   qsub jobs/bhfm2_internal_m2_sweep.sh
   qsub jobs/bhfm2_internal_m2_doping.sh
10) Again run qstat -u $USER and report all 4 job IDs and states.
11) Output a concise status table:
   job_name | job_id | state | log_path | started_ok | first_issue

Do not modify code. Do not delete checkpoints. Stop after reporting status.
```
