# Job Templates

SLURM templates for BU SCC phase-scan runs.

## Files

- `phase_scan_m1_baseline.sh`
  - fast baseline (`M=1`) for sketch compatibility checks
- `phase_scan_m2_quality.sh`
  - higher-quality run (`M=2`) with stricter validity

## Usage

```bash
cd $HOME/dmft
mkdir -p logs
sbatch jobs/phase_scan_m1_baseline.sh
sbatch jobs/phase_scan_m2_quality.sh
```

Before submitting, update partition/account/module lines for your SCC setup.
