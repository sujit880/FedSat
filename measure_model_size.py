import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flearn.models.model import get_model_by_name


def count_parameters(model):
    """Count the total number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Calculate the model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def measure_all_models(dataset='cifar'):
    """Measure size and parameters for all models in MODEL_FACTORIES"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {dataset}\n")
    print("=" * 80)
    
    # List of models to measure (from MODEL_FACTORIES in get_model_by_name)
    models_to_test = [
        'mlp',
        'resnet18',
        'tresnet18p',
        'tresnet34p',
        'tresnet50p',
        'tresnet101p',
        'tresnet18',
        'tresnet34',
        'tresnet50',
        'tresnet101',
        'resnet8',
        'lenet5',
    ]
    
    results = []
    
    for model_name in models_to_test:
        try:
            print(f"\nMeasuring {model_name}...")
            
            # Create model
            model = get_model_by_name(dataset, device, model_name)
            
            # Count parameters
            num_params = count_parameters(model)
            
            # Get model size in MB
            size_mb = get_model_size_mb(model)
            
            results.append({
                'model': model_name,
                'parameters': num_params,
                'size_mb': size_mb
            })
            
            print(f"  ✓ {model_name}: {num_params:,} parameters, {size_mb:.2f} MB")
            
            # Clear memory
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ✗ Error measuring {model_name}: {str(e)}")
            results.append({
                'model': model_name,
                'parameters': 'Error',
                'size_mb': 'Error',
                'error': str(e)
            })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("\nSUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model Name':<20} {'Parameters':>20} {'Size (MB)':>15}")
    print("-" * 80)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['model']:<20} {result['parameters']:>20,} {result['size_mb']:>15.2f}")
        else:
            print(f"{result['model']:<20} {'Error':>20} {'Error':>15}")
    
    print("=" * 80)
    
    # Save results to file
    output_file = f'model_sizes_{dataset}.txt'
    with open(output_file, 'w') as f:
        f.write(f"Model Size Measurements for {dataset}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model Name':<20} {'Parameters':>20} {'Size (MB)':>15}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            if 'error' not in result:
                f.write(f"{result['model']:<20} {result['parameters']:>20,} {result['size_mb']:>15.2f}\n")
            else:
                f.write(f"{result['model']:<20} {'Error':>20} {'Error':>15}\n")
                f.write(f"  Error: {result['error']}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    # You can change the dataset here if needed
    dataset = 'cifar'  # Use 'cifar' for CIFAR-10
    
    # Allow command-line argument for dataset
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    
    results = measure_all_models(dataset)
