import torch
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
from typing import List
from copy import deepcopy
from collections import OrderedDict
from flearn.clients.fedks import FedKSeedClient
from flearn.utils.trainer_utils import get_optimizer_by_name
from flearn.config.trainer_params import FLAIR_ARGS
from flearn.utils.constants import CLASSES, SAMPLE_PER_CLASS
from flearn.trainers.server import BaseServer
from flearn.models.generative import CLS
import random
import torch.nn.functional as F

class FedKSeedServer(BaseServer):
    def __init__(self, params):
        super().__init__(params)
        self.seed_pool_size = 20
        self.seed_pool = list(range(self.seed_pool_size))
        self.scalar_gradient_accumulator = {seed: 0.0 for seed in self.seed_pool}

        self.clients: list[FedKSeedClient] = self.clients
        self.num_classes = CLASSES[self.dataset]  
        self.sample_per_class = SAMPLE_PER_CLASS[self.dataset]
        self.feature_dim = self.client_model.get_feature_dim()

        self.global_classifier = CLS(device=self.device, input_dim=self.feature_dim, num_classes=self.num_classes).to(self.device)
        for val in self.global_classifier.parameters():
            val.requires_grad = False
        self.global_cls_dict =OrderedDict({k: v.clone().detach() for k, v in self.global_classifier.named_parameters()})
        print(f'Global Classifier: {self.global_cls_dict.keys()}')

        FLAIR_ARGS["classifier"] = deepcopy(self.global_classifier)
        FLAIR_ARGS["num_classes"] = self.num_classes
        FLAIR_ARGS["lamda_falg"] = 1.0

        for (k, v) in self.client_model.state_dict().items():
            if k.startswith('fc.') or k.startswith('resnet.fc.') or k.startswith('linear.') or k.startswith('resnet.linear.'):
                v.requires_grad = False
        
        # Set seed pool for all clients
        FLAIR_ARGS["seed_pool"] = self.seed_pool
        for client in self.clients:
            client.init_client_specific_params(**FLAIR_ARGS)            
            client.model = deepcopy(self.client_model)
            client.use_custom_classifier = True # This needs to be set true for poper testing and validation
    
    def train(self):
        """Train using FedKSeed approach"""
        print(f"Training with {self.clients_per_round} workers using FedKSeed")
        
        for i in trange(self.num_rounds, desc=self.desc):
            # Evaluate model
            self.eval(i, self.set_client_model_test)
            
            # Select clients for this round
            selected_clients: list[FedKSeedClient] = self.select_clients(i, num_clients=min(self.clients_per_round, len(self.clients)))
            
            # Collect solutions from clients
            all_scalar_grads = []
            csolns = []
            
            for c in tqdm(selected_clients, desc="Training Clients", leave=False):
                c.round = i
                scalar_grads, soln_cls = c.solve_inner_kseed(
                    global_classifier=deepcopy(self.global_classifier),
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )
                all_scalar_grads.append((c.num_samples, scalar_grads))
                csolns.append(soln_cls)
            
            # Aggregate classifier parameters
            self.global_cls_dict = self.aggregate_cls(csolns)
            self.global_classifier.load_state_dict(self.global_cls_dict)
            
            # Aggregate scalar gradients
            self.aggregate_scalar_gradients(all_scalar_grads)
            
            # Update global model using accumulated scalar gradients
            self.update_global_model_with_scalar_grads()
        
        self.eval_end()

    def set_client_model_test(self, client: FedKSeedClient):
        # pass
        client.classifier.load_state_dict(self.global_cls_dict)
        client.model.load_state_dict(self.client_model.state_dict())
    
    def aggregate_scalar_gradients(self, all_scalar_grads):
        """Aggregate scalar gradients from clients"""
        total_samples = sum(samples for samples, _ in all_scalar_grads)
        
        # Reset accumulator
        for seed in self.seed_pool:
            self.scalar_gradient_accumulator[seed] = 0.0
        
        # Weighted aggregation
        for samples, scalar_grads in all_scalar_grads:
            weight = samples / total_samples
            for seed, grad in scalar_grads.items():
                self.scalar_gradient_accumulator[seed] += weight * grad

    @torch.no_grad()
    def aggregate_cls(self, wsolns):  # Weighted average using PyTorch
        total_weight = 0.0
        # Assume wsolns is a list of tuples (w, soln), where soln is a list of PyTorch tensors

        base = [torch.zeros_like(val) for val in self.global_cls_dict.values()]

        for w, client_state_dict in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(client_state_dict.values()):
                base[i] += w * v

        # Divide each aggregated tensor by the total weight to compute the average
        averaged_soln = [v / total_weight for v in base]
        self.global_cls_dict = OrderedDict(zip(self.global_cls_dict.keys(), averaged_soln))
        return self.global_cls_dict
    
    def update_global_model_with_scalar_grads(self):
        """Update global model using accumulated scalar gradients"""
        # Store original parameters
        original_params = OrderedDict({k: v.clone() for k, v in self.client_model.named_parameters()})
        
        # Apply updates for each seed
        for seed, scalar_grad in self.scalar_gradient_accumulator.items():
            if abs(scalar_grad) < 1e-8:  # Skip negligible gradients
                continue
                
            # Set seed for reproducible perturbation
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Generate perturbation
            with torch.no_grad():
                for k, param in self.client_model.named_parameters():
                    if param.requires_grad:
                        # Generate perturbation and apply scaled update
                        perturbation = torch.randn_like(param) * 1e-3 * torch.norm(param)
                        param.add_(-self.learning_rate * scalar_grad * perturbation)


class ParameterGeneratorModel(torch.nn.Module):
    def __init__(self, feature_extractor_dim, classifier_dim, latent_dim=256):
        super().__init__()
        # Encoder for classifier parameters
        self.classifier_encoder = torch.nn.Sequential(
            torch.nn.Linear(classifier_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, latent_dim)
        )
        
        # Decoder for feature extractor parameters
        self.feature_decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, feature_extractor_dim)
        )
        
        # UNet-based diffusion model components
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, latent_dim)
        )
        
        self.diffusion_model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, latent_dim)
        )
        
    def forward(self, classifier_params, timestep, noise=None):
        # Encode classifier parameters
        classifier_embedding = self.classifier_encoder(classifier_params)
        
        # Time embedding
        time_embedding = self.time_embed(timestep.unsqueeze(1).float())
        
        # Combine embeddings
        combined = torch.cat([classifier_embedding, time_embedding], dim=1)
        
        # Predict noise (or directly the latent)
        noise_pred = self.diffusion_model(combined)
        
        return noise_pred
    
    def generate_feature_extractor(self, classifier_params, num_diffusion_steps=100):
        # Start from random noise
        latent = torch.randn(1, self.diffusion_model[0].in_features // 2, device=classifier_params.device)
        
        # Diffusion sampling process
        for i in reversed(range(num_diffusion_steps)):
            timestep = torch.tensor([i / num_diffusion_steps], device=classifier_params.device)
            
            # Predict noise
            noise_pred = self.forward(classifier_params, timestep, latent)
            
            # Update latent (simplified diffusion sampling)
            alpha = 1 - (i / num_diffusion_steps) * 0.02  # Simplified schedule
            latent = (latent - (1 - alpha) * noise_pred) / alpha
            
            # Add some noise for stochasticity
            if i > 0:
                noise_scale = (i / num_diffusion_steps) * 0.1
                latent = latent + noise_scale * torch.randn_like(latent)
        
        # Decode the latent to get feature extractor parameters
        feature_extractor_params = self.feature_decoder(latent)
        
        return feature_extractor_params
