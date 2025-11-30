#!/bin/bash
# FedSat Quick Reference Card
# Paste this in your terminal for quick access to common commands

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════╗
║                    FEDSAT QUICK REFERENCE CARD                           ║
╚══════════════════════════════════════════════════════════════════════════╝

🚀 QUICK START
───────────────────────────────────────────────────────────────────────────
  ./quick_start.sh                    # Interactive menu (EASIEST!)
  ./run_yaml_experiments.sh quick     # Quick validation test

📊 COMMON EXPERIMENTS
───────────────────────────────────────────────────────────────────────────
  ./run_yaml_experiments.sh quick     # Fast test (~10 min)
  ./run_yaml_experiments.sh baseline  # Baseline comparison (6-12h)
  ./run_yaml_experiments.sh adaptive  # Adaptive methods (8-16h)
  ./run_yaml_experiments.sh ablation  # Ablation study (6-12h)
  ./run_yaml_experiments.sh paper     # Main results (1-2 days)
  ./run_yaml_experiments.sh full      # Full comparison (3-7 DAYS!)

🔍 PREVIEW & LISTING
───────────────────────────────────────────────────────────────────────────
  ./run_yaml_experiments.sh list                # List all experiments
  ./run_yaml_experiments.sh baseline --dry-run  # Preview commands
  ./run_yaml_experiments.sh help                # Show help

📁 DATASET GENERATION
───────────────────────────────────────────────────────────────────────────
  ./generate_datasets.sh              # Interactive dataset generator
  
  # OR manually:
  python generate_clients_dataset.py --dataset cifar10 \
    --type noiid_lbldir --clients 100 --beta 0.3

⚙️ MANUAL EXECUTION
───────────────────────────────────────────────────────────────────────────
  # FedSat (CORRECT)
  python main.py --trainer=fedavg --agg=fedsat --loss=CALC \
    --dataset=cifar10 --dataset_type=noiid_lbldir_b0_3_k100 \
    --num_rounds=200 --num_epochs=5 --batch_size=64

  # FedAvg baseline
  python main.py --trainer=fedavg --loss=CE \
    --dataset=cifar10 --dataset_type=noiid_lbldir_b0_3_k100 \
    --num_rounds=200 --num_epochs=5 --batch_size=64

🌙 BACKGROUND EXECUTION
───────────────────────────────────────────────────────────────────────────
  # Run in background with nohup
  nohup ./run_yaml_experiments.sh full > full.log 2>&1 &
  
  # Monitor progress
  tail -f full.log
  
  # Check status
  ps aux | grep run_yaml_experiments

📝 PAPER EXPERIMENTS
───────────────────────────────────────────────────────────────────────────
  ./run_yaml_experiments.sh paper                    # Main results table
  ./run_yaml_experiments.sh run ablation_table       # Ablation table
  ./run_yaml_experiments.sh run loss_comparison      # Loss comparison
  ./run_yaml_experiments.sh run non_iid_robustness   # Non-IID study

🔧 CUSTOMIZATION
───────────────────────────────────────────────────────────────────────────
  # Edit configuration
  vim configs/experiments.yaml
  
  # Run custom experiment
  ./run_yaml_experiments.sh run my_experiment

💾 KEY PARAMETERS FOR FEDSAT
───────────────────────────────────────────────────────────────────────────
  --trainer=fedavg        # Use FedAvg trainer (NOT fedsat!)
  --agg=fedsat            # FedSat aggregation method
  --loss=CALC             # CALC or CACS loss (REQUIRED for fedsat agg)
  
  ❌ WRONG: --trainer=fedavg --agg=fedsat --loss=CE   (will ERROR!)
  ✅ RIGHT: --trainer=fedavg --agg=fedsat --loss=CALC

🗂️ RESULTS LOCATION
───────────────────────────────────────────────────────────────────────────
  RESULTS/json_dump/      # Raw JSON results
  RESULTS/results/        # Processed results
  RESULTS/figures/        # Generated plots

🆘 TROUBLESHOOTING
───────────────────────────────────────────────────────────────────────────
  chmod +x *.sh           # Make scripts executable
  pip install pyyaml      # Install PyYAML
  ./generate_datasets.sh  # Generate missing datasets
  nvidia-smi              # Check GPU status

📚 DOCUMENTATION
───────────────────────────────────────────────────────────────────────────
  RUNNING_EXPERIMENTS.md  # Complete running guide
  SCRIPTS_REFERENCE.md    # Shell scripts reference
  YAML_CONFIG_GUIDE.md    # YAML configuration guide
  FEDSAT_CONFIG.md        # FedSat configuration details

╔══════════════════════════════════════════════════════════════════════════╗
║  RECOMMENDED FIRST STEPS:                                                ║
║  1. ./generate_datasets.sh          # Generate datasets                 ║
║  2. ./run_yaml_experiments.sh quick # Validate setup                    ║
║  3. ./quick_start.sh                # Interactive experiments            ║
╚══════════════════════════════════════════════════════════════════════════╝

EOF
