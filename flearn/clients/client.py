import torch
import numpy as np
from thop import profile
from typing import Any
from rich.console import Console
import copy
from collections import OrderedDict
from flearn.data.dataset import CLASSES as CLASSES
from flearn.data.data_utils import get_dataloader, get_dataset_stats
from flearn.utils.torch_utils import process_grad
from flearn.utils.torch_utils import graph_size
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.optimizer.pggd import PerGodGradientDescent
from flearn.optimizer.scaffoldopt import ScaffoldOptimizer
from flearn.utils.losses import get_loss_fun
from flearn.utils.trainer_utils import trainable_params


class BaseClient(object):

    def __init__(
        self,
        user_id,
        device,
        lr: float,
        weight_decay: float,
        momentum: float,
        loss: str,
        batch_size: int,
        dataset: str,
        valset_ratio: float,
        logger: Console,
        gpu: int,
        dataset_type: str,
        n_class: int,
        optm=None,
        group=None,
        model: torch.nn.Module = None,
        num_workers: int = 0,
        img_float_val_range: tuple[int | float, int | float] = (0, 1),
    ):
        self.id = user_id  # integer
        self.lr = lr
        self.loss = loss
        self.group = group
        self.num_samples = None
        self.test_samples = None
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.logger = logger
        self.model = copy.deepcopy(model)
        self.optm = optm
        self.c_local = None  # Required for scaffold
        self.sensitivity = None  # Required for Elastic Aggregation
        self.device = device
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.use_custom_classifier: bool = False
        self.classifier: torch.nn.Module = None # if self.use_custom_classifier is True self.classifier needs to have custom classifier module
        # Initialize required atributes to calculate loss

        # TODO: Add to utils.losses
        if loss == "CL":
            self.label_distrib = get_dataset_stats(
                self.dataset,
                dataset_type=dataset_type,
                n_class=n_class,
                client_id=self.id,
            ).to(self.device)
            CLCrossEntropyLoss = get_loss_fun(self.loss)
            self.criterion = CLCrossEntropyLoss(
                label_distrib=self.label_distrib, tau=0.5 #defaulte tau was 0.5, 0.1
            )
        elif loss == "FL":
            FocalLossFn = get_loss_fun(self.loss)
            self.criterion = FocalLossFn(alpha=1.0, gamma=2.0)
        elif loss == "LS":
            LSLossFn = get_loss_fun(self.loss)
            self.criterion = LSLossFn(smoothing=0.1)
        elif loss == "CB":
            CBLossFn = get_loss_fun(self.loss)
            self.criterion = CBLossFn(beta=0.9999)
        elif self.loss == "CAPA":
            K = CLASSES[self.dataset]
            # safe initial weights: uniform off-diagonal for W, zeros for U
            W0 = (torch.ones(K, K, device=self.device) - torch.eye(K, device=self.device)) / (K - 1)
            U0 = torch.zeros(K, device=self.device)
            # swap your criterion
            CAPALoss = get_loss_fun(self.loss)
            self.criterion = CAPALoss(W0, U0, lam=0.5, mu=0.01, tau=2.0, margin=0.2, kappa=6.0, use_gate=True) # margin=0.1

            # soft confusion trackers (float, with small prior)
            prior = 0.5
            self.conf_N = torch.full((K, K), prior, device=self.device)
            self.pred_q = torch.full((K,),     prior, device=self.device)
            self.label_y = torch.full((K,),    prior, device=self.device)
        elif self.loss == "DB":
            K = CLASSES[self.dataset]
            # swap your criterion
            DBCCLoss = get_loss_fun(self.loss)
            self.criterion = DBCCLoss(
                K=K,
                alpha=1.0,         # logit-adjust strength (like CB parameter beta)
                ema_m=0.995,       # EMA momentum for priors & confusion
                device=self.device
            )
        elif self.loss == "CACS":
            K = CLASSES[self.dataset]
            CACSLoss = get_loss_fun(self.loss)
            # Dictionary of optimal parameters for each dataset
            CACS_PARAMS = {
                "cifar10":   dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "cifar":   dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "svhn":      dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "mnist":     dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "fmnist":     dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                # "emnist":   dict(m0=0.05, alpha=0.3, ema_m=0.85, warmup_steps=2000, prior_beta=0.9, conf_beta=0.7, lmu=0.9, cmu=0.4),
                # "emnist":   dict(ema_m=0.95, warmup_steps=2000, prior_beta=0.95, conf_beta=0.5, lmu=0.9, cmu=0.9),
                "emnist":   dict(ema_m=0.95, warmup_steps=2000, prior_beta=0.95, conf_beta=0.85, lmu=0.9, cmu=0.9),
                # "emnist":   dict(ema_m=0.9, warmup_steps=1000, prior_beta=0.99, conf_beta=0.9, lmu=0.5, cmu=0.5),
                "femnist":   dict(m0=0.05, alpha=0.3, ema_m=0.85, warmup_steps=2000, prior_beta=0.8, conf_beta=0.3, lmu=0.9, cmu=0.4),
                # "cifar100":  dict(m0=0.1, alpha=0.5, ema_m=0.75, warmup_steps=2000, prior_beta=0.85, conf_beta=0.85, lmu=0.9, cmu=0.4),
                # "cifar100":  dict(m0=0.1, alpha=0.3, ema_m=0.85, warmup_steps=2000, prior_beta=0.85, conf_beta=0.85, lmu=0.9, cmu=0.4),
                # "cifar100":  dict(m0=0.1, alpha=0.3, ema_m=0.85, warmup_steps=2000, prior_beta=0.85, conf_beta=0.85, lmu=0.9, cmu=0.9), #best1                
                # "cifar100":  dict(m0=0.1, alpha=0.1, ema_m=0.85, warmup_steps=2000, prior_beta=0.85, conf_beta=0.85, lmu=0.9, cmu=0.41),
                # "cifar100":  dict(m0=0.1, alpha=1.0, ema_m=0.85, warmup_steps=2000, prior_beta=0.85, conf_beta=0.85, lmu=0.9, cmu=0.9),   #best1
                "cifar100":  dict(m0=0.2, alpha=1.0, ema_m=0.85, warmup_steps=2000, prior_beta=0.85, conf_beta=0.85, lmu=0.9, cmu=0.9), 
                # Add more datasets as needed
            }
            # Use default if dataset not found
            params = CACS_PARAMS.get(self.dataset, dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05))
            self.criterion = CACSLoss(
                K=K,
                device=self.device,
                **params
            )
        elif self.loss == "CALC":
            # CACS with Label Calibration: pass label distribution and similar dataset-specific params
            K = CLASSES[self.dataset]
            # label distribution (per-client/global depending on get_dataset_stats implementation)
            self.label_distrib = get_dataset_stats(
                self.dataset,
                dataset_type=dataset_type,
                n_class=n_class,
                client_id=self.id,
            ).to(self.device)
            CACSLC = get_loss_fun(self.loss)
            # reuse the same params dictionary as CACS where available
            CACS_PARAMS_LC = {
                "cifar10":   dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "cifar":   dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "svhn":      dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "mnist":     dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "fmnist":     dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05),
                "emnist":   dict(ema_m=0.95, warmup_steps=2000, prior_beta=0.95, conf_beta=0.85, lmu=0.9, cmu=0.9),
                "femnist":   dict(m0=0.05, alpha=0.3, ema_m=0.85, warmup_steps=2000, prior_beta=0.8, conf_beta=0.3, lmu=0.9, cmu=0.4),
                "cifar100":  dict(m0=0.2, alpha=1.0, ema_m=0.85, warmup_steps=2000, prior_beta=0.85, conf_beta=0.85, lmu=0.9, cmu=0.9),
            }
            params_lc = CACS_PARAMS_LC.get(self.dataset, dict(m0=0.1, alpha=0.5, ema_m=0.95, warmup_steps=1000, prior_beta=0.95, conf_beta=0.5, lmu=0.95, cmu=0.05))
            # default tau for label calibration
            tau_default = 0.1
            self.criterion = CACSLC(
                K=K,
                label_distrib=self.label_distrib,
                tau=tau_default,
                device=self.device,
                **params_lc,
            )
        elif self.loss == "LCCA":
            # Label-Calibrated CE with CACS-style confusion adaptivity
            K = CLASSES[self.dataset]
            self.label_distrib = get_dataset_stats(
                self.dataset,
                dataset_type=dataset_type,
                n_class=n_class,
                client_id=self.id,
            ).to(self.device)
            LCCE = get_loss_fun(self.loss)
            # sensible defaults
            tau_default = 0.1
            lambda_conf_default = 0.5
            ema_m_default = 0.995
            self.criterion = LCCE(
                K=K,
                label_distrib=self.label_distrib,
                tau=tau_default,
                lambda_conf=lambda_conf_default,
                ema_m=ema_m_default,
                device=self.device,
            )
        else:
            self.criterion = get_loss_fun(self.loss)()
        # Setting optimizer

        # TODO: Add to utils.optmizer
        if optm == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=weight_decay, momentum=momentum
            )
        elif optm == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay, momentum=momentum)
            
        self.trainloader, self.valloader, self.num_samples, self.test_samples = (
            get_dataloader(
                dataset,
                user_id,
                dataset_type,
                n_class,
                batch_size,
                valset_ratio,
                num_workers,
                img_float_val_range,
            )
        )
        self.iter_trainloader = iter(self.trainloader)

        self.noisy = False
        self.noise_level = 0.0
     

    def setattributes(self, key, value):  # Required additing attributes can be set
        setattr(self, key, value)

    def set_model_params(self, model_state_dict: OrderedDict) -> None:
        self.model.load_state_dict(model_state_dict, strict=False)

    def get_model_params(
        self,
        toDetach: bool = True,
        toCPU: bool = False,
        getNumpy: bool = False,
        customModel: torch.nn.Module | None = None,
    ) -> OrderedDict[str, Any]:

        model = customModel
        if model is None:
            model = self.model

        params: OrderedDict[str, Any] = model.state_dict()
        for key, val in params.items():
            val = val.clone()
            if toDetach:
                val = val.detach()
            if toCPU:
                val = val.cpu()
            if getNumpy:
                val = val.numpy()
        return params
    
    def get_classifier_params(
        self,
        toDetach: bool = True,
        toCPU: bool = False,
        getNumpy: bool = False,
        customModel: torch.nn.Module | None = None,
    ) -> OrderedDict[str, Any]:

        model: torch.nn.Module = customModel
        if model is None:
            model = self.model

        params: OrderedDict[str, torch.Tensor] = OrderedDict(
                                                (key, val.clone()) for key, val in model.state_dict().items() 
                                                if key.startswith('fc.') or key.startswith('resnet.fc.')
                                            )
        for key, val in params.items():
            if toDetach:
                val = val.detach()
            if toCPU:
                val = val.cpu()
            if getNumpy:
                val = val.numpy()
        return params

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    def get_grads(self, data=None):
        """get model gradient"""
        self.optimizer.zero_grad()
        if data == None:
            inputs, labels = self.get_data_batch()
        else:
            inputs, labels = data
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        grads = process_grad([param.grad.cpu() for param in self.model.parameters()])
        num_samples = len(labels)
        return num_samples, grads

    def get_grads_t(self, data=None):
        """get model gradient"""
        self.optimizer.zero_grad()
        if data == None:
            inputs, labels = self.get_data_batch()
        else:
            inputs, labels = data
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        grads = [param.grad.clone() for param in self.model.parameters()]
        self.optimizer.zero_grad()
        num_samples = len(labels)
        return (num_samples, grads)

    def solve_grad(self):
        """get model gradient with cost"""
        inputs, labels = self.get_data_batch()
        flops, params_size = profile(self.model, inputs=(inputs,))
        # Print the results
        # print(f"FLOPs: {flops / 1e9} G")
        # print(f"Number of parameters: {params_size / 1e6} M")
        bytes_w = graph_size(self.model)
        num_samples, grads = self.get_grads([inputs, labels])
        comp = flops * num_samples
        bytes_r = graph_size(self.model)
        return ((num_samples, grads), (bytes_w, comp, bytes_r))

    def solve_iters(self, num_iters=1, batch_size=10):
        """Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        """

        bytes_w = graph_size(self.model)
        for _ in range(num_iters):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if len(labels)<=1: continue
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        soln = self.get_params()
        comp = 0
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def grad_sensitivity(self):
        if hasattr(self, 'sensitivity') is False or self.sensitivity is None:
            self.sensitivity = torch.zeros(
                len(list(self.model.parameters())), device=self.device
            )
            if hasattr(self, 'elastic_mu') is False:
                self.elastic_mu = 0.95
        self.model.eval()
        for x, y in self.valloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            grads_norm = [
                torch.norm(layer_grad[0]) ** 2
                for layer_grad in torch.autograd.grad(
                    loss, self.model.parameters()
                )
            ]
            for i in range(len(grads_norm)):
                self.sensitivity[i] = (
                    self.elastic_mu * self.sensitivity[i]
                    + (1 - self.elastic_mu) * grads_norm[i].abs()
                )
        self.model.train()
        return self.sensitivity

    def check_attribute(self, attribute: str):
        if hasattr(self, attribute):
            print(f"Attribute {attribute} is present in the class.")
            return True
        else:
            print(f"Attribute {attribute} is not present in the class.")
            return False

    def train_error_and_loss(self, modelInCPU: bool = False):
        tot_correct, loss, train_sample = 0, 0.0, 0

        if modelInCPU:
            self.model = self.model.to(self.device)

        for inputs, labels in self.trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
            else: outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            if self.loss == "DBCC":
                feats = self.model.get_representation_features(inputs)
                logits = self.model.classifier(feats)
                if hasattr(self.model, "resnet"):
                    if hasattr(self.model.resnet, "fc"):
                        class_w = self.model.resnet.fc.weight  
                elif hasattr(self.model, "fc"):
                    class_w = self.model.fc.weight
                elif hasattr(self.model, "linear"):
                    class_w = self.model.linear.weight                      
                loss += self.criterion(logits, labels, feats, class_w).item()
            else:  
                loss += self.criterion(outputs, labels).item()
            train_sample += len(labels)

        if modelInCPU:
            self.model = self.model.cpu()

        return tot_correct, loss, train_sample

    def test(self, modelInCPU: bool = False) -> tuple[int, int]:
        """tests the current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        """
        self.model.eval()
        if modelInCPU:
            self.model = self.model.to(self.device)

        tot_correct, loss, test_sample = 0, 0.0, 0
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
            else: outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            if self.loss == "DBCC":
                feats = self.model.get_representation_features(inputs)
                logits = self.model.classifier(feats)
                if hasattr(self.model, "resnet"):
                    if hasattr(self.model.resnet, "fc"):
                        class_w = self.model.resnet.fc.weight  
                elif hasattr(self.model, "fc"):
                    class_w = self.model.fc.weight
                elif hasattr(self.model, "linear"):
                    class_w = self.model.linear.weight                       
                loss += self.criterion(logits, labels, feats, class_w).item()
            else:  
                loss += self.criterion(outputs, labels).item()
            test_sample += len(labels)

        self.model.train()
        if modelInCPU:
            self.model = self.model.cpu()
        return tot_correct, test_sample
    
    def test_model(self, client, modelInCPU: bool = False) -> tuple[int, int]:
        """tests the current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        """
        # print(f'Calling Local test')
        client.model.eval()
        if modelInCPU:
            client.model = client.model.to(self.device)

        tot_correct, loss, test_sample = 0, 0.0, 0
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = client.model.get_representation_features(inputs)
                outputs = client.classifier(features)
            else: outputs = client.model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            if self.loss == "DBCC":
                feats = self.model.get_representation_features(inputs)
                logits = self.model.classifier(feats)
                if hasattr(self.model, "resnet"):
                    if hasattr(self.model.resnet, "fc"):
                        class_w = self.model.resnet.fc.weight  
                elif hasattr(self.model, "fc"):
                    class_w = self.model.fc.weight
                elif hasattr(self.model, "linear"):
                    class_w = self.model.linear.weight                       
                loss += self.criterion(logits, labels, feats, class_w).item()
            else:  
                loss += self.criterion(outputs, labels).item()
            test_sample += len(labels)

        client.model.train()
        if modelInCPU:
            client.model = client.model.cpu()

        return tot_correct, test_sample

    def test_stats(self, record_stats=None, modelInCPU: bool = False):
        if record_stats == None:
            record_stats = {}
        tot_correct, loss, test_sample, pred = (
            0,
            0.0,
            0,
            np.zeros(CLASSES[self.dataset]),
        )
        
        self.model.eval()
        if modelInCPU:
            self.model = self.model.to(self.device)

        targ, matched = copy.deepcopy(pred), copy.deepcopy(pred)
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
            else: outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            if self.loss == "DBCC":
                feats = self.model.get_representation_features(inputs)
                logits = self.model.classifier(feats)
                if hasattr(self.model, "resnet"):
                    if hasattr(self.model.resnet, "fc"):
                        class_w = self.model.resnet.fc.weight  
                elif hasattr(self.model, "fc"):
                    class_w = self.model.fc.weight
                elif hasattr(self.model, "linear"):
                    class_w = self.model.linear.weight                      
                loss += self.criterion(logits, labels, feats, class_w).item()
            else:  
                loss += self.criterion(outputs, labels).item()
            # print(f"Predicted: {predicted}, Target: {labels}") #Print
            # print(record_stats)
            if "pred" not in record_stats:
                record_stats["pred"] = {}
                record_stats["target"] = {}
                record_stats["match"] = {}
            for x, y in zip(predicted, labels):
                X = x.item()
                Y = y.item()
                pred[int(x)] += 1
                targ[int(y)] += 1
                if X not in record_stats["pred"]:
                    # print(f'predicted new class {X}')
                    record_stats["pred"][X] = 0
                record_stats["pred"][X] += 1
                if Y not in record_stats["target"]:
                    # print(f'predicted new class {Y}')
                    record_stats["target"][Y] = 0
                record_stats["target"][Y] += 1
                if X == Y:
                    matched[int(Y)] += 1
                    if Y not in record_stats["match"]:
                        # print(f'new matced in worker {self.id}: {X}, {Y}')
                        record_stats["match"][Y] = 0
                    record_stats["match"][Y] += 1
            test_sample += len(labels)
        if test_sample>0.0:
            acc = (tot_correct / test_sample)
        else:
            print(f'Client:{self.id} Doesn\'t have sufficient test sample') 
            acc = 0.0
        # pred[CLASSES[self.dataset]] = acc #reduced size of pred no space for storing acc

        self.model.train()
        if modelInCPU:
            self.model = self.model.cpu()

        return loss, acc, matched, record_stats, pred, targ
    
    def test_and_cost_matrix_stats_t(self, modelInCPU: bool = False):
        self.model.eval()

        if modelInCPU:
            self.model = self.model.to(self.device)

        tot_correct, test_sample = 0.0, 0.0
        cost_matrix = torch.zeros(
            CLASSES[self.dataset], CLASSES[self.dataset], device=self.device
        )
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
            else: outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            for x, y in zip(labels, predicted):
                cost_matrix[x][y] = cost_matrix[x][y] + 1
                # print(f"x: {x}, y: {y}")
                if x == y:
                    tot_correct += 1
            test_sample += len(labels)
            # print(cost_matrix, tot_correct, test_sample)
            # raise RuntimeError
        acc = (tot_correct / test_sample) if test_sample>0.0 else 0.0
        cost_acc = torch.diag(cost_matrix).sum() / cost_matrix.sum()
        # print(f"acc: {acc}, cost_acc: {cost_acc} , total_sample: {test_sample}, cost_sample: {cost_matrix.sum()}, total_correct: {tot_correct}, cost_correct: {torch.diag(cost_matrix).sum()}")

        if modelInCPU:
            self.model = self.model.cpu()

        self.model.train()
        return cost_matrix

