#!/bin/bash
# Main submission script for IDW-PINN cluster jobs
# Usage: ./submit_idw_pinn.sh <config_file> [num_runs] [seed_mode]
set -e
# Configuration
CONFIG_FILE=${1:-"config.yaml"}
NUM_RUNS=${2:-1}
SEED_MODE=${3:-"fixed"}  # Options: "random", "fixed", "sequential"
BASE_SEED=${4:-42}
WORKDIR=$PWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "IDW-PINN Cluster Submission"
echo "=========================="
echo "Config file: $CONFIG_FILE"
echo "Working directory: $WORKDIR"
echo "Number of runs: $NUM_RUNS"
echo "Seed mode: $SEED_MODE"
echo ""
# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi
# Verify job template exists
if [ ! -f "$SCRIPT_DIR/default_scripts/runCase.sh" ]; then
    echo "Error: Job template not found: $SCRIPT_DIR/default_scripts/runCase.sh"
    exit 1
fi
# Generate seeds based on mode
declare -a SEEDS
case $SEED_MODE in
    "random")
        echo "Generating random seeds..."
        for i in $(seq 1 $NUM_RUNS); do
            SEEDS[$i]=$RANDOM
        done
        ;;
    "sequential")
        echo "Generating sequential seeds starting from $BASE_SEED..."
        for i in $(seq 1 $NUM_RUNS); do
            SEEDS[$i]=$((BASE_SEED + i - 1))
        done
        ;;
    "fixed")
        echo "Using predetermined seed list..."
        FIXED_SEEDS=(42 55 71 89 107 127 149 173 199 227 251 277 307 337 367 397 431 463 499 541)
        for i in $(seq 1 $NUM_RUNS); do
            if [ $i -le ${#FIXED_SEEDS[@]} ]; then
                SEEDS[$i]=${FIXED_SEEDS[$((i-1))]}
            else
                SEEDS[$i]=$RANDOM
            fi
        done
        ;;
    *)
        echo "Error: Unknown seed mode '$SEED_MODE'"
        echo "Valid modes: random, fixed, sequential"
        exit 1
        ;;
esac
echo "Seeds for this run: ${SEEDS[@]}"
echo ""
# Create submission log
SUBMIT_LOG="$WORKDIR/submission_log_$(date +%Y%m%d_%H%M%S).txt"
echo "IDW-PINN Cluster Submission - $(date)" > $SUBMIT_LOG
echo "Config: $CONFIG_FILE" >> $SUBMIT_LOG
echo "Seed mode: $SEED_MODE" >> $SUBMIT_LOG
echo "Number of runs: $NUM_RUNS" >> $SUBMIT_LOG
echo "Seeds: ${SEEDS[@]}" >> $SUBMIT_LOG
echo "========================================" >> $SUBMIT_LOG
# Submit jobs
for i in $(seq 1 $NUM_RUNS); do
    SEED=${SEEDS[$i]}
    RUN_NAME="idw_run${i}_s${SEED}"
    RUN_DIR="$WORKDIR/runs/$RUN_NAME"
    
    echo "Submitting run $i: $RUN_NAME (seed=$SEED)"
    echo "Run $i: $RUN_NAME (seed=$SEED)" >> $SUBMIT_LOG
    
    # Create run directory
    mkdir -p "$RUN_DIR"
    
    # Copy config to run directory
    cp "$CONFIG_FILE" "$RUN_DIR/config.yaml"
    
    # Copy job script template and customize
    cp "$SCRIPT_DIR/default_scripts/runCase.sh" "$RUN_DIR/runCase.sh"
    cd "$RUN_DIR"
    
    # Update job name
    sed -i "s/JOBNAME/$RUN_NAME/g" runCase.sh
    
    # Submit job with seed as argument
    JOB_ID=$(qsub -v SEED=$SEED,CONFIG_PATH=config.yaml runCase.sh | awk '{print $3}')
    echo "  Job ID: $JOB_ID"
    echo "  Job ID: $JOB_ID" >> $SUBMIT_LOG
    
    cd "$WORKDIR"
    
    # Small delay to avoid overwhelming scheduler
    sleep 1
done
echo ""
echo "All $NUM_RUNS jobs submitted successfully!"
echo "Results will be in: $WORKDIR/runs/"
echo "Submission log: $SUBMIT_LOG"