import torch
import numpy as np
from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
from flearn.utils.trainer_utils import get_optimizer_by_name
from flearn.models.generative import LCLS
from typing import Tuple, Union, List, Optional
from flearn.models.model import TorchResNet


class FLUTEClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: Optional[TorchResNet] = self.model

    def init_client_specific_params(
        self,
        num_classes: int,
        rep_round: int,
        lambda1: float,
        lambda2: float,
        lambda3: float,
        classifier: LCLS,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.rep_round = rep_round
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.classifier: LCLS = classifier
        self.optimizer: torch.optim.Optimizer = get_optimizer_by_name(
                            optm=self.optm, 
                            parameters=list(self.model.parameters())+ list(self.classifier.parameters()), 
                            lr=self.lr, 
                            weight_decay=self.weight_decay, 
                            momentum = self.momentum, # For SGD only
                        )

        # Adding client data labels
        data_labels = torch.zeros((self.num_classes, 1), device=self.device)
        for y in np.unique(self.trainloader.dataset.dataset.targets):
            data_labels[y][0] = 1
        self.data_labels = data_labels

    def solve_inner(self, num_epochs=1, batch_size=10):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes_read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """
        self.model.train()
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def solve_inner_flute(self, num_epochs=1, batch_size=10):
        """Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """

        self.model.train()

        bytes_w = graph_size(self.model)
        train_sample_size = 0
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:

                if len(inputs) < 1:
                    continue

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)

                weight = self.classifier.cls.weight

                loss += self.lambda1 * torch.norm(
                    torch.matmul(weight, weight.t())
                    / torch.norm(torch.matmul(weight, weight.t()))
                    - 1
                    / torch.sqrt(torch.tensor(self.num_classes - 1).to(self.device))
                    * torch.mul(
                        (
                            torch.eye(self.num_classes).to(self.device)
                            - 1
                            / self.num_classes
                            * torch.ones((self.num_classes, self.num_classes)).to(
                                self.device
                            )
                        ),
                        torch.matmul(
                            self.data_labels,
                            self.data_labels.t(),
                        ).to(self.device),
                    )
                )
                loss += self.lambda2 * torch.norm(weight) ** 2

                if epoch >= num_epochs - self.rep_round:
                    loss += self.lambda3 * torch.norm(features) ** 2

                self.optimizer.zero_grad()
                loss.backward()

                if epoch < num_epochs - self.rep_round:
                    self.model.zero_grad()

                self.optimizer.step()

                train_sample_size += len(labels)

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def finetune_flute(self, num_epochs=1, batch_size=10):
        for layer in self.model.parameters():
            layer.requires_grad = False

        self.solve_inner(num_epochs, batch_size)

        for layer in self.model.parameters():
            layer.requires_grad = True
    
    def flute_localUpdate(self, num_epochs=1, batch_size=10):
        self.model.train()

        bytes_w = graph_size(self.model)
        train_sample_size = 0
        bias_p = []
        weight_p = []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        for name, p in self.classifier.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=self.lr, momentum=self.momentum
        )
        head_eps = num_epochs - self.rep_round
        epoch_loss = []
        num_updates = 0
        
        for iter in range(num_epochs):
            done = False
            # if (iter < head_eps):
            #     for name, param in self.model.named_parameters():
            #         param.requires_grad = False
            # if (iter >= head_eps):
            #     for name, param in self.model.named_parameters():
            #         param.requires_grad = True
            if (iter >= self.rep_round):
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
            batch_loss = []

            for inputs, labels in self.trainloader:

                if len(inputs) < 1:
                    continue

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level # Adding noise to input for DP 

                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()

                if iter >= head_eps:
                    for name, param in self.classifier.named_parameters():
                        # Compute Frobenius norm of param * param^T
                        param_gram = torch.matmul(param, param.t())  # [C x C]
                        f_norm = torch.norm(param_gram, p='fro') + 1e-8  # add epsilon to avoid div-by-zero

                        # Identity matrix and uniform matrix
                        eye = torch.eye(self.num_classes, device=self.device)
                        ones = torch.ones((self.num_classes, self.num_classes), device=self.device)

                        # Label similarity matrix
                        label_sim = torch.matmul(self.data_labels, self.data_labels.T).to(self.device)

                        # Construct target matrix
                        scaling = 1.0 / torch.sqrt(torch.tensor(self.num_classes - 1.0, device=self.device))
                        target = scaling * torch.mul(eye - (1.0 / self.num_classes) * ones, label_sim)

                        # Frobenius norm difference (regularization term)
                        r1 = 0.25 * torch.norm(param_gram / f_norm - target, p='fro')
                        r2 = 0.0025 * torch.norm(param, p='fro') ** 2
                    r3 = 0.0005 * torch.norm(features, p='fro') ** 2
                    loss += r1 + r2 + r3
                elif iter < head_eps:
                    for name, param in self.classifier.named_parameters():
                        # Compute Frobenius norm of param * param^T
                        param_gram = torch.matmul(param, param.t())  # [C x C]
                        f_norm = torch.norm(param_gram, p='fro') + 1e-8  # add epsilon to avoid div-by-zero

                        # Identity matrix and uniform matrix
                        eye = torch.eye(self.num_classes, device=self.device)
                        ones = torch.ones((self.num_classes, self.num_classes), device=self.device)

                        # Label similarity matrix
                        label_sim = torch.matmul(self.data_labels, self.data_labels.T).to(self.device)

                        # Construct target matrix
                        scaling = 1.0 / torch.sqrt(torch.tensor(self.num_classes - 1.0, device=self.device))
                        target = scaling * torch.mul(eye - (1.0 / self.num_classes) * ones, label_sim)

                        # Frobenius norm difference (regularization term)
                        r1 = 0.25 * torch.norm(param_gram / f_norm - target, p='fro')
                        r2 = 0.0025 * torch.norm(param, p='fro') ** 2
                    loss += r1 + r2

                loss.backward()
                self.optimizer.step()
                train_sample_size += len(labels)


                num_updates += 1
                batch_loss.append(loss.item())

                # if num_updates == self.args.local_updates:
                #     done = True
                #     break
            if len(batch_loss)>0:    
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # if done:
            #     break
        for _, param in self.model.named_parameters():
            param.requires_grad = True

        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)