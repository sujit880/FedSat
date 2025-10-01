import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict
import random

def get_representative_subset(test_loader, num_samples=1000, n_class=10):
    # Collect all data and labels
    all_inputs = []
    all_targets = []

    for inputs, targets in test_loader:
        all_inputs.append(inputs)
        all_targets.append(targets)

    # Concatenate into single tensors
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    total_samples = all_targets.shape[0]
    assert total_samples >= num_samples, "Requested more samples than available."

    # Get indices grouped by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(all_targets):
        class_indices[int(label.item())].append(idx)

    # Stratified sampling: get proportional count per class
    selected_indices = []
    for cls in range(n_class):
        cls_idxs = class_indices[cls]
        proportion = len(cls_idxs) / total_samples
        k = int(round(proportion * num_samples))
        sampled = random.sample(cls_idxs, min(k, len(cls_idxs)))
        selected_indices.extend(sampled)

    # If slightly under/over due to rounding, fix the size
    if len(selected_indices) > num_samples:
        selected_indices = random.sample(selected_indices, num_samples)
    elif len(selected_indices) < num_samples:
        remaining = list(set(range(total_samples)) - set(selected_indices))
        selected_indices.extend(random.sample(remaining, num_samples - len(selected_indices)))

    # Create a Subset and DataLoader
    subset_dataset = torch.utils.data.TensorDataset(all_inputs[selected_indices], all_targets[selected_indices])
    subset_loader = DataLoader(subset_dataset, batch_size=int(num_samples/2), shuffle=False)

    return subset_loader, len(selected_indices)
