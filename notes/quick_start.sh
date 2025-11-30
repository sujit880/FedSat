#!/bin/bash
# Quick Start Script for FedSat Experiments
# Provides easy access to common experimental tasks

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                   ║${NC}"
echo -e "${GREEN}║          FedSat Experiment Quick Start           ║${NC}"
echo -e "${GREEN}║                                                   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# Menu
PS3="Select an option (1-11): "
options=(
    "Quick Test (10 rounds, 1 seed)"
    "Baseline Comparison (FedAvg, FedProx, SCAFFOLD, MOON)"
    "Adaptive Methods (FedAdagrad, FedYogi, FedAdam)"
    "Class Imbalance Study"
    "Ablation Study (FedSat components)"
    "Full Comparison (All methods)"
    "Paper: Main Results Table"
    "Paper: Ablation Table"
    "Paper: Loss Comparison"
    "List All Available Experiments"
    "Exit"
)

select opt in "${options[@]}"
do
    case $opt in
        "Quick Test (10 rounds, 1 seed)")
            echo -e "\n${BLUE}Running Quick Test...${NC}\n"
            ./run_yaml_experiments.sh quick
            break
            ;;
        "Baseline Comparison (FedAvg, FedProx, SCAFFOLD, MOON)")
            echo -e "\n${YELLOW}⚠ This may take several hours${NC}"
            read -p "Continue? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "\n${BLUE}Running Baseline Comparison...${NC}\n"
                ./run_yaml_experiments.sh baseline
            fi
            break
            ;;
        "Adaptive Methods (FedAdagrad, FedYogi, FedAdam)")
            echo -e "\n${YELLOW}⚠ This may take several hours${NC}"
            read -p "Continue? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "\n${BLUE}Running Adaptive Methods...${NC}\n"
                ./run_yaml_experiments.sh adaptive
            fi
            break
            ;;
        "Class Imbalance Study")
            echo -e "\n${YELLOW}⚠ This may take several hours${NC}"
            read -p "Continue? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "\n${BLUE}Running Class Imbalance Study...${NC}\n"
                ./run_yaml_experiments.sh imbalance
            fi
            break
            ;;
        "Ablation Study (FedSat components)")
            echo -e "\n${YELLOW}⚠ This may take several hours${NC}"
            read -p "Continue? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "\n${BLUE}Running Ablation Study...${NC}\n"
                ./run_yaml_experiments.sh ablation
            fi
            break
            ;;
        "Full Comparison (All methods)")
            echo -e "\n${YELLOW}⚠ This will take 3-7 DAYS to complete!${NC}"
            read -p "Are you sure? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "\n${BLUE}Running Full Comparison...${NC}"
                echo -e "${YELLOW}Tip: Run with nohup in background:${NC}"
                echo -e "  nohup ./run_yaml_experiments.sh full > full_exp.log 2>&1 &"
                echo ""
                read -p "Run in foreground anyway? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    ./run_yaml_experiments.sh full
                fi
            fi
            break
            ;;
        "Paper: Main Results Table")
            echo -e "\n${YELLOW}⚠ This will take 1-2 days${NC}"
            read -p "Continue? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "\n${BLUE}Running Main Results...${NC}\n"
                ./run_yaml_experiments.sh paper
            fi
            break
            ;;
        "Paper: Ablation Table")
            echo -e "\n${BLUE}Running Ablation Table...${NC}\n"
            ./run_yaml_experiments.sh run ablation_table
            break
            ;;
        "Paper: Loss Comparison")
            echo -e "\n${BLUE}Running Loss Comparison...${NC}\n"
            ./run_yaml_experiments.sh run loss_comparison
            break
            ;;
        "List All Available Experiments")
            echo -e "\n${BLUE}Listing All Experiments...${NC}\n"
            ./run_yaml_experiments.sh list
            break
            ;;
        "Exit")
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option $REPLY"
            ;;
    esac
done
