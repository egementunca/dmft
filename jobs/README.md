# Job Templates

SGE templates for BU SCC runs.

## Bond Scheme (corrected, March 2026)

See [RUN_BOND_INSTRUCTIONS.md](RUN_BOND_INSTRUCTIONS.md) for full details.

```bash
cd $HOME/dmft && mkdir -p logs

# M=1: internal vs professor's new (comparison)
qsub jobs/bond_m1_internal.sh
qsub jobs/bond_m1_prof_new.sh

# M=2: internal only
qsub jobs/bond_m2_internal.sh        # CPU, 48 hrs
qsub jobs/bond_m2_internal_gpu.sh    # GPU, 4 hrs (see GPU_CLUSTER_README.md)
```

## Phase Scan

```bash
qsub jobs/phase_scan_m1_baseline.sh
qsub jobs/phase_scan_m2_quality.sh
```

## Usage

Before submitting, update partition/account/module lines for your SCC setup.
