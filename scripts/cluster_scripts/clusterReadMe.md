# IDW-PINN Cluster Scripts

Scripts for running IDW-PINN training jobs on the Rocks cluster.

## Quick Start

```bash
# 1. Create a new case
./create_idw_case.sh MyCaseName

# 2. Copy data into the case folder
cp /path/to/your_data.csv ~/projects/pinnRuns/MyCaseName/

# 3. Edit config.yaml to point to your data
cd ~/projects/pinnRuns/MyCaseName
nano config.yaml  # set input_file: "your_data.csv"

# 4. Submit jobs
./submit_idw_pinn.sh config.yaml 5 fixed  # 5 runs with fixed seeds
```

## Case Folder Structure

Running `create_idw_case.sh` generates:

```
~/projects/pinnRuns/MyCaseName/
├── config.yaml              # Training configuration (edit this)
├── submit_idw_pinn.sh       # Job submission script
├── your_data.csv            # Your data file (copy here manually)
└── default_scripts/
    ├── runCase.sh           # SGE job template
    └── idw_cluster_train.py # Training script
```

## Configuration

Edit `config.yaml` to configure your run. Key settings:

```yaml
data:
  input_file: "your_data.csv"    # Path to your CSV or MAT file
  n_obs: 500                      # Number of observation points
  n_f: 10000                      # Number of collocation points

physics:
  diff_coeff_init: 0.1            # Initial D guess
  diff_coeff_true: 0.2            # Ground truth (if known, for validation)

training:
  epochs: 10000                   # Adam epochs
  lbfgs_epochs: 5000              # L-BFGS epochs

network:
  layers: [3, 64, 64, 64, 64, 1]  # Network architecture
```

## Scripts Reference

### create_idw_case.sh

Creates a new case directory with all necessary files.

```bash
./create_idw_case.sh [case_name] [options]

Options:
  -s, --source DIR    Source scripts directory (default: ~/projects/IDW/scripts/cluster_scripts)
  -c, --config FILE   Config template to copy (default: ~/projects/IDW/configs/default_2d_inverse.yaml)
  -d, --dest DIR      Parent directory for cases (default: ~/projects/pinnRuns)
  -r, --runs N        Number of runs (default: 1)
  -m, --seed-mode     Seed mode: fixed, random, sequential (default: fixed)
```

### submit_idw_pinn.sh

Submits training jobs to the SGE scheduler.

```bash
./submit_idw_pinn.sh <config_file> [num_runs] [seed_mode] [base_seed]

Arguments:
  config_file   Path to YAML config (default: config.yaml)
  num_runs      Number of independent runs (default: 1)
  seed_mode     How to generate seeds (default: fixed)
                - fixed: Use predetermined seed list (42, 55, 71, ...)
                - random: Generate random seeds
                - sequential: Seeds starting from base_seed
  base_seed     Starting seed for sequential mode (default: 42)
```

## Data Format

### CSV Format (experimental data)

```csv
x,y,t,intensity
0,0,0,0.95
1,0,0,0.87
...
```

- `x, y`: Spatial coordinates (pixels or physical units)
- `t`: Time (frames or physical units)
- `intensity`: Measured concentration/intensity values

### MAT Format (synthetic data)

MATLAB file containing:
- `usol`: Solution array (nx × ny × nt)
- `x`, `y`, `t`: Coordinate vectors

## Workflow Example

```bash
# Create case for experimental run
./create_idw_case.sh 20260128_Exp1

# Navigate to case
cd ~/projects/pinnRuns/20260128_Exp1

# Copy experimental data
cp ~/data/my_experiment.csv ./experiment_data.csv

# Edit config to use your data and adjust parameters
nano config.yaml
# Set: input_file: "experiment_data.csv"
# Adjust: n_obs, epochs, etc.

# Submit 10 runs with different seeds
./submit_idw_pinn.sh config.yaml 10 fixed

# Monitor jobs
qstat

# Results will be in runs/ subdirectory
ls runs/
```

## Output Structure

After jobs complete:

```
~/projects/pinnRuns/MyCaseName/
├── config.yaml
├── submit_idw_pinn.sh
├── submission_log_YYYYMMDD_HHMMSS.txt
└── runs/
    ├── idw_run1_s42/
    │   ├── config.yaml
    │   ├── training.log
    │   ├── results/
    │   └── ...
    ├── idw_run2_s55/
    └── ...
```

