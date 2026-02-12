#!/bin/bash
# SGE job script for IDW-PINN training
# This script is customized per job by submit_idw_pinn.sh

#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N JOBNAME
#$ -pe mpich 1
#$ -P WolframGroup

# Error handling
set -e

# Source environment
source ~/.bashrc

# Set TensorFlow/Python environment variables
export TF_CPP_MIN_LOG_LEVEL=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Memory management
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0
ulimit -c unlimited

# Activate conda environment
conda activate pinn

# Debug info
echo "=========================================="
echo "IDW-PINN Job Starting"
echo "=========================================="
echo "Job name: $JOB_NAME"
echo "Job ID: $JOB_ID"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $PWD"
echo "Seed: $SEED"
echo "Config: $CONFIG_PATH"
echo ""

echo "Environment:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo ""

# System info
echo "System resources:"
echo "  CPUs: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

echo "=========================================="
echo ""

# Set PYTHONPATH to your project root
export PYTHONPATH="/state/partition1/home/$USER/projects/IDW:${PYTHONPATH}"

# Get case folder info
CASE_FOLDER=$PWD
CASE_NAME=$(basename $CASE_FOLDER)

# Setup temp directory on compute node
TMP_DIR="/tmp/$USER/$CASE_NAME"
mkdir -p $TMP_DIR

# Copy files to temp
echo "Copying files to compute node temp: $TMP_DIR"
rsync -a $CASE_FOLDER/ $TMP_DIR/
cd $TMP_DIR

# Start memory monitor in background
(
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S'): Memory: $(free -m | grep Mem | awk '{print $3}')MB" >> memory_monitor.log
        sleep 30
    done
) &
MONITOR_PID=$!

# Run training
echo "Starting IDW-PINN training..."
echo "Command: python -m src.idw_pinn.train --config $CONFIG_PATH --seed $SEED"
echo ""

{
    python -m src.idw_pinn.train \
        --config "$CONFIG_PATH" \
        --seed "$SEED" \
        2>&1 | tee training.log
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    echo "Training completed with exit code: $EXIT_CODE"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Exit code $EXIT_CODE" >> execution.log
}

# Kill memory monitor
kill $MONITOR_PID 2>/dev/null || true

# Sync results back
echo ""
echo "Syncing results back to $CASE_FOLDER"
cd ..
rsync -uavz --exclude="*.core" $TMP_DIR/ $CASE_FOLDER/

# Cleanup
echo "Cleaning up temp directory"
rm -rf $TMP_DIR

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="

exit $EXIT_CODE
