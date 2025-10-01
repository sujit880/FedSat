from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch

class MOONClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_client_specific_params(
        self,
        mu: float,
        tau: int,
        prev_model: torch.nn.Module,
        **kwargs,
    ) -> None:
        self.mu = mu
        self.tau = tau
        self.prev_model = prev_model
        for param in self.prev_model.parameters():
            param.requires_grad = False
        self.prev_model.eval()


    def solve_inner(self, num_epochs=1, batch_size=10):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes_read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """

        bytes_w = graph_size(self.model)
        train_sample_size = 0
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
    
    def solve_inner_moon_t(self, global_model:torch.nn.Module=None, num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        '''
        if global_model is None:
            print(f'Require global model, training stoped')
            raise RuntimeError
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        self.model.train()
        global_model.eval()
        for epoch in range(num_epochs): # for epoch in tqdm(range(num_epochs), desc='Epoch: ', leave=False, ncols=120):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if len(labels)<=1: continue
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level # Adding noise to input for DP 
                # Compute features for current, global, and previous models
                z_curr = self.model.get_representation_features(inputs)
                # print(f"Feature shape {z_curr.shape}")
                with torch.no_grad():
                    z_global = global_model.get_representation_features(inputs)
                    z_prev = self.prev_model.get_representation_features(inputs)

                # Compute loss components
                logits = self.model.classifier(z_curr)
                loss_sup = self.criterion(logits, labels)

                sim_curr_global = torch.nn.functional.cosine_similarity(z_curr, z_global, dim=-1)
                sim_prev_curr = torch.nn.functional.cosine_similarity(z_curr, z_prev, dim=-1)
                # denominator = torch.exp(sim_prev_curr / self.tau) + torch.exp(sim_curr_global / self.tau)
                # loss_con = -torch.log(torch.exp(sim_curr_global / self.tau) / (denominator + 1e-8))

                logits1 = sim_curr_global.reshape(-1,1)
                logits1 = torch.cat((logits1, sim_prev_curr.reshape(-1,1)), dim=1)
                logits1 /= self.tau
                labels1 = torch.zeros(inputs.size(0)).cuda().long()
                loss_con = self.mu * self.criterion(logits1, labels1)
                
            
                # Combine losses and backpropagate
                loss = loss_sup + loss_con #self.mu * loss_con.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(f'total_loss: {loss.item()}, cls_loss: {loss_sup.item()}, con_loss: {loss_con.item()}')
                train_sample_size += len(labels)

        # self.prev_model.load_state_dict(self.model.state_dict())
        self.prev_model.load_state_dict(global_model.state_dict())
        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
