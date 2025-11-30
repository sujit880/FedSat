#!/bin/bash
# Dataset Generation Helper for FedSat Experiments
# This script helps generate datasets needed for experiments

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                   ║${NC}"
echo -e "${GREEN}║          FedSat Dataset Generator                 ║${NC}"
echo -e "${GREEN}║                                                   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# Function to generate dataset
generate_dataset() {
    local dataset=$1
    local beta=$2
    local num_clients=$3
    
    echo -e "${BLUE}Generating $dataset with beta=$beta, clients=$num_clients${NC}"
    
    python generate_clients_dataset.py \
        --dataset "$dataset" \
        --type noiid_lbldir \
        --clients "$num_clients" \
        --beta "$beta"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset generated successfully${NC}"
    else
        echo -e "${RED}✗ Dataset generation failed${NC}"
        return 1
    fi
}

# Menu
echo -e "${YELLOW}Select dataset generation option:${NC}\n"
PS3="Select an option (1-7): "
options=(
    "Generate all datasets for experiments (RECOMMENDED)"
    "CIFAR-10 (beta=0.3, 100 clients)"
    "CIFAR-100 (beta=0.3, 100 clients)"
    "Fashion-MNIST (beta=0.3, 100 clients)"
    "EMNIST (beta=0.3, 100 clients)"
    "Custom dataset configuration"
    "Exit"
)

select opt in "${options[@]}"
do
    case $opt in
        "Generate all datasets for experiments (RECOMMENDED)")
            echo -e "\n${YELLOW}This will generate datasets for all experiments${NC}"
            echo -e "Datasets to generate:"
            echo "  • CIFAR-10:  beta=0.5, 0.3, 0.1 (100 clients)"
            echo "  • CIFAR-100: beta=0.5, 0.3, 0.1 (100 clients)"
            echo "  • FMNIST:    beta=0.5, 0.3, 0.1 (100 clients)"
            echo ""
            read -p "Continue? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # CIFAR-10
                echo -e "\n${BLUE}═══ CIFAR-10 ═══${NC}"
                generate_dataset "cifar10" 0.5 100
                generate_dataset "cifar10" 0.3 100
                generate_dataset "cifar10" 0.1 100
                
                # CIFAR-100
                echo -e "\n${BLUE}═══ CIFAR-100 ═══${NC}"
                generate_dataset "cifar100" 0.5 100
                generate_dataset "cifar100" 0.3 100
                generate_dataset "cifar100" 0.1 100
                
                # Fashion-MNIST
                echo -e "\n${BLUE}═══ Fashion-MNIST ═══${NC}"
                generate_dataset "fmnist" 0.5 100
                generate_dataset "fmnist" 0.3 100
                generate_dataset "fmnist" 0.1 100
                
                echo -e "\n${GREEN}✓ All datasets generated successfully!${NC}"
            fi
            break
            ;;
        "CIFAR-10 (beta=0.3, 100 clients)")
            generate_dataset "cifar10" 0.3 100
            break
            ;;
        "CIFAR-100 (beta=0.3, 100 clients)")
            generate_dataset "cifar100" 0.3 100
            break
            ;;
        "Fashion-MNIST (beta=0.3, 100 clients)")
            generate_dataset "fmnist" 0.3 100
            break
            ;;
        "EMNIST (beta=0.3, 100 clients)")
            generate_dataset "emnist" 0.3 100
            break
            ;;
        "Custom dataset configuration")
            echo ""
            read -p "Dataset (cifar10/cifar100/fmnist/emnist): " dataset
            read -p "Beta (e.g., 0.3): " beta
            read -p "Number of clients (e.g., 100): " num_clients
            generate_dataset "$dataset" "$beta" "$num_clients"
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
