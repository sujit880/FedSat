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

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default config file
CONFIG_FILE="configs/experiments.yaml"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if Python is available
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.6+."
        exit 1
    fi
    print_success "Python found: $(python --version)"
}

# Function to check if PyYAML is installed
check_pyyaml() {
    if ! python -c "import yaml" &> /dev/null; then
        print_warning "PyYAML not found. Installing..."
        pip install pyyaml
        print_success "PyYAML installed"
    else
        print_success "PyYAML is installed"
    fi
}

# Function to check if config file exists
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    print_success "Config file found: $CONFIG_FILE"
}

# Function to show usage
show_usage() {
    cat << EOF
${GREEN}FedSat YAML Experiment Runner${NC}

Usage: $0 [COMMAND] [OPTIONS]

${YELLOW}Commands:${NC}
  list                    List all available experiments
  run SUITE [OPTIONS]     Run a specific experiment suite
  quick                   Run quick_test (shortcut)
  baseline                Run baseline_comparison (shortcut)
  adaptive                Run adaptive_methods (shortcut)
  imbalance               Run class_imbalance (shortcut)
  ablation                Run ablation_study (shortcut)
  full                    Run full_comparison (shortcut)
  paper                   Run main_results for paper (shortcut)
  help                    Show this help message

${YELLOW}Options:${NC}
  --dry-run              Print commands without executing
  --config FILE          Use custom config file (default: configs/experiments.yaml)
  
${YELLOW}Examples:${NC}
  # List all available experiments
  $0 list
  
  # Run quick test
  $0 quick
  
  # Run baseline comparison with dry-run
  $0 baseline --dry-run
  
  # Run full comparison (long running)
  $0 full
  
  # Run paper experiments
  $0 paper
  
  # Run custom suite
  $0 run hyperparameter_search
  
  # Use custom config
  $0 run quick_test --config my_experiments.yaml

${YELLOW}Available Experiment Suites:${NC}
  Standard Experiments:
    • quick_test           - Fast validation (10 rounds, 1 seed)
    • baseline_comparison  - Compare FL baseline methods
    • adaptive_methods     - Adaptive optimization study
    • class_imbalance      - Class imbalance handling
    • ablation_study       - Ablation on FedSat components
    • full_comparison      - Complete evaluation
    • hyperparameter_search- Grid search
    • scalability          - Scalability study
    
  Paper Experiments:
    • main_results         - Main comparison table
    • ablation_table       - Ablation study table
    • loss_comparison      - Loss function comparison
    • non_iid_robustness   - Non-IID robustness study

${YELLOW}Notes:${NC}
  • FedSat requires: --trainer=fedavg --agg=fedsat --loss=CALC
  • Use --dry-run to preview commands before running
  • Long experiments should be run with nohup or screen

EOF
}

# Parse arguments
COMMAND=""
SUITE=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        list|help)
            COMMAND="$1"
            shift
            ;;
        run)
            COMMAND="run"
            SUITE="$2"
            shift 2
            ;;
        quick|baseline|adaptive|imbalance|ablation|full|paper)
            COMMAND="run"
            case $1 in
                quick)    SUITE="quick_test" ;;
                baseline) SUITE="baseline_comparison" ;;
                adaptive) SUITE="adaptive_methods" ;;
                imbalance) SUITE="class_imbalance" ;;
                ablation) SUITE="ablation_study" ;;
                full)     SUITE="full_comparison" ;;
                paper)    SUITE="main_results" ;;
            esac
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
case "$COMMAND" in
    help|"")
        show_usage
        exit 0
        ;;
    
    list)
        print_info "Checking dependencies..."
        check_python
        check_pyyaml
        check_config
        echo ""
        print_info "Listing all available configurations..."
        python run_yaml_experiments.py --list --config "$CONFIG_FILE"
        ;;
    
    run)
        if [ -z "$SUITE" ]; then
            print_error "No experiment suite specified"
            show_usage
            exit 1
        fi
        
        print_info "Checking dependencies..."
        check_python
        check_pyyaml
        check_config
        
        echo ""
        print_info "Running experiment suite: ${GREEN}$SUITE${NC}"
        
        if [ -n "$DRY_RUN" ]; then
            print_warning "DRY RUN MODE - No experiments will be executed"
        fi
        
        echo ""
        
        # Build command
        CMD="python run_yaml_experiments.py $SUITE --config $CONFIG_FILE $DRY_RUN"
        
        # Execute
        if [ -n "$DRY_RUN" ]; then
            eval $CMD
        else
            # Ask for confirmation for long-running experiments
            case "$SUITE" in
                full_comparison|hyperparameter_search|main_results)
                    print_warning "This is a long-running experiment suite!"
                    read -p "Continue? (y/n): " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        print_info "Aborted by user"
                        exit 0
                    fi
                    ;;
            esac
            
            # Run the experiment
            eval $CMD
            
            if [ $? -eq 0 ]; then
                print_success "Experiment completed successfully!"
            else
                print_error "Experiment failed!"
                exit 1
            fi
        fi
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
