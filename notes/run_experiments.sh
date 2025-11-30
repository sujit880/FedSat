#!/bin/bash
# Advanced batch script to run federated learning experiments
# Usage: bash run_experiments.sh [OPTIONS]

# Initialize conda for bash shell (if not already initialized)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

# Activate conda environment (only if conda is available)
if command -v conda &> /dev/null; then
    conda activate py3_12 2>/dev/null || echo "Warning: Could not activate 'py3_12' environment"
fi

# Parse command line arguments
EXPERIMENT_SET="baseline"  # default
DRY_RUN=""
CONFIG_FILE="configs/experiments.yaml"
EXPERIMENTS=""
GRID_SEARCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --set)
            EXPERIMENT_SET="$2"
            shift 2
            ;;
        --all)
            EXPERIMENT_SET="all"
            shift
            ;;
        --experiments)
            # Collect all experiment names until next flag
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXPERIMENTS="$EXPERIMENTS $1"
                shift
            done
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --grid)
            GRID_SEARCH="$2"
            shift 2
            ;;
        --list)
            python run_yaml_experiments.py --list --config "$CONFIG_FILE"
            exit 0
            ;;
        --help)
            echo "Usage: bash run_experiments.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --set NAME           Run experiment set (default: baseline)"
            echo "  --all                Run all experiments"
            echo "  --experiments NAME...  Run specific experiments"
            echo "  --grid NAME          Run grid search"
            echo "  --dry-run            Show what would run without executing"
            echo "  --config FILE        Use custom config file (default: configs/experiments.yaml)"
            echo "  --list               List all available experiments and sets"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_experiments.sh --set baseline_comparison"
            echo "  bash run_experiments.sh --all"
            echo "  bash run_experiments.sh --experiments quick_test baseline_comparison"
            echo "  bash run_experiments.sh --grid hyperparameter_search"
            echo "  bash run_experiments.sh --dry-run --all"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=============================="
echo "Starting Experiments"
echo "=============================="
echo "Config file: $CONFIG_FILE"

# Run experiments based on mode
if [ -n "$GRID_SEARCH" ]; then
    echo "Mode: Grid Search ($GRID_SEARCH)"
    python run_yaml_experiments.py "$GRID_SEARCH" --config "$CONFIG_FILE" $DRY_RUN
    EXIT_CODE=$?
elif [ -n "$EXPERIMENTS" ]; then
    echo "Mode: Specific Experiments ($EXPERIMENTS)"
    EXIT_CODE=0
    for exp in $EXPERIMENTS; do
        echo ""
        echo "Running experiment: $exp"
        python run_yaml_experiments.py "$exp" --config "$CONFIG_FILE" $DRY_RUN
        if [ $? -ne 0 ]; then
            EXIT_CODE=1
        fi
    done
elif [ "$EXPERIMENT_SET" == "all" ]; then
    echo "Mode: All Experiments"
    python run_yaml_experiments.py --all --config "$CONFIG_FILE" $DRY_RUN
    EXIT_CODE=$?
else
    echo "Mode: Experiment Set ($EXPERIMENT_SET)"
    python run_yaml_experiments.py "$EXPERIMENT_SET" --config "$CONFIG_FILE" $DRY_RUN
    EXIT_CODE=$?
fi

echo ""
echo "=============================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Batch completed successfully!"
else
    echo "✗ Batch completed with errors"
fi
echo "=============================="

exit $EXIT_CODE
