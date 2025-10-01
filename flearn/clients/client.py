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
                label_distrib=self.label_distrib, tau=0.1 #defaulte tau was 0.5
            )
        elif loss == "MSL":
            self.label_distrib = get_dataset_stats(
                self.dataset,
                dataset_type=dataset_type,
                n_class=n_class,
                client_id=self.id,
            ).to(self.device)
            K = CLASSES[self.dataset]
            C0 = torch.zeros(K, K, device=self.device)
            MSLCrossEntropyLoss = get_loss_fun(self.loss)
            self.criterion = MSLCrossEntropyLoss(
                label_distrib=self.label_distrib, conf_N=C0, tau=0.5
            ) #v2
            # self.criterion = MSLCrossEntropyLoss(
            #     num_classes=K, device=self.device
            # ) #v3
        elif loss == "FL":
            FocalLossFn = get_loss_fun(self.loss)
            self.criterion = FocalLossFn(alpha=1.0, gamma=2.0)
        elif loss == "LS":
            LSLossFn = get_loss_fun(self.loss)
            self.criterion = LSLossFn(smoothing=0.1)

        elif loss == "CB":
            CBLossFn = get_loss_fun(self.loss)
            self.criterion = CBLossFn(beta=0.9999)
        elif loss == "CS":
            CrossSensitiveLoss = get_loss_fun(self.loss)
            self.criterion = CrossSensitiveLoss(
                torch.ones(
                    CLASSES[self.dataset], CLASSES[self.dataset], device=self.device
                )
            )
        elif loss == "CSN":
            CrossSensitiveLoss = get_loss_fun(self.loss)
            self.criterion = CrossSensitiveLoss(
                cost_matrix=torch.ones(
                    CLASSES[self.dataset], CLASSES[self.dataset], device=self.device
                ),
            )
        elif loss == "PSL" or loss=="PSL1":
            K = CLASSES[self.dataset]
            C0 = torch.zeros(K, K)
            MisclassificationAwarePairwiseLoss = get_loss_fun(self.loss)
            self.criterion = MisclassificationAwarePairwiseLoss(
                    cost_matrix=C0,
                    lam=0.5,          # fraction of plain CE (raise if training is noisy)
                    alpha=0.5,        # blend expected-cost vs pairwise
                    margin=0.1,       # soft margin; 0.0–0.5 are common
                    normalize_pairwise=True,
                    focal_gamma=0.0,  # set to 1.0 to focus hard examples in exp-cost term
                    reduction="mean"
                ).to(device)
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
        elif self.loss == "MCAPA":
            K = CLASSES[self.dataset]
            # swap your criterion
            MCAPALoss = get_loss_fun(self.loss)
            self.criterion = MCAPALoss(device=self.device, K=K, lam=0.75, mu=0.025, mtau=0.5, tau=2.0, margin=0.05, use_gate=True) # margin=0.05
        elif self.loss == "MCA":
            K = CLASSES[self.dataset]
            # swap your criterion
            MCALoss = get_loss_fun(self.loss)
            self.criterion = MCALoss(device=self.device, K=K, lam=0.9, mu=0.05, mtau=0.5, tau=2.0, margin=0.5, use_gate=True) # margin=0.05
        elif self.loss == "DBCC":
            K = CLASSES[self.dataset]
            # swap your criterion
            DBCCLoss = get_loss_fun(self.loss)
            self.criterion = DBCCLoss(
                K=K,
                alpha=1.0,         # logit-adjust strength (like CB parameter beta)
                lambda_cc=0.1,     # weight for contrastive term
                topM=5,            # number of confusing negatives per class
                ema_m=0.995,       # EMA momentum for priors & confusion
                warmup_epochs=2,   # epochs before enabling confusion term
                rare_temp=0.07,    # temp for rare classes (sharper)
                head_temp=0.10,    # temp for head classes (softer)
                device=self.device
            )
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
        elif self.loss == "CALB":
            K = CLASSES[self.dataset]
            CALBLoss = get_loss_fun(self.loss)
            self.criterion = CALBLoss(
                K=K,
                scale=0.5,          # strength of confusion-based margins
                ema_m=0.995,        # EMA smoothing for confusion/prior
                la_alpha=0.0,       # set >0.0 to also apply LA
                warmup_steps=10000,  # wait ~1 epoch before enabling margins
                row_norm="l1",
                symmetrize=True,
                device=self.device
            )
        elif self.loss == "CACS":
            K = CLASSES[self.dataset]
            CACSLoss = get_loss_fun(self.loss)
            self.criterion = CACSLoss(
                K=K,
                m0=0.1,            # base cost
                alpha=0.5,         # confusion scaling
                ema_m=0.995,       # smooth the confusion
                warmup_steps=1000, # ~1 epoch # v1,2 = 10000, v3=100
                prior_beta=0.5,    # set 0.5–1.0 to also apply Bayes logit adjust #v1 0.5 v2=0.3, p_0= 0.0
                device=self.device
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
        if optm == "PGD":
            self.optimizer = PerturbedGradientDescent(
                self.model.parameters(), learning_rate=self.lr
            )
        if optm == "PGGD":
            self.optimizer = PerGodGradientDescent(
                self.model.parameters(), learning_rate=self.lr
            )
        if optm == "SCAFFOLD":
            self.optimizer = ScaffoldOptimizer(
                self.model.parameters(), lr=self.lr, weight_decay=1e-4
            )
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

    def test_sensitivity_stats(self, record_stats=None):

        self.model.eval()
        if record_stats == None:
            record_stats = {}
        tot_correct, losses, test_sample, pred = (
            0,
            0.0,
            0,
            np.zeros(CLASSES[self.dataset]),
        )
        targ, matched = copy.deepcopy(pred), copy.deepcopy(pred)
        layer_num = len(trainable_params(self.model))
        sensitivity = torch.zeros(layer_num, device=self.device)
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
            else: outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            grads_norm = [
                torch.norm(layer_grad[0]) ** 2
                for layer_grad in torch.autograd.grad(
                    loss, trainable_params(self.model)
                )
            ]
            for i in range(len(grads_norm)):
                sensitivity[i] = (
                    self.mu * sensitivity[i] + (1 - self.mu) * grads_norm[i].abs()
                )
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            losses += loss.item()
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
        acc = (tot_correct / test_sample) if test_sample>0.0 else 0.0
        self.model.train()
        return (
            losses,
            acc,
            matched,
            record_stats,
            pred,
            targ,
            (test_sample, np.array(sensitivity.cpu())),
        )

    def test_sensitivity_stats_t(self, record_stats=None):
        self.model.eval()
        if record_stats == None:
            record_stats = {}
        tot_correct, losses, test_sample, pred = (
            0,
            0.0,
            0,
            np.zeros(CLASSES[self.dataset]),
        )
        targ, matched = copy.deepcopy(pred), copy.deepcopy(pred)
        layer_num = len(trainable_params(self.model))
        sensitivity = torch.zeros(layer_num, device=self.device)
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
            else: outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            grads_norm = [
                torch.norm(layer_grad[0]) ** 2
                for layer_grad in torch.autograd.grad(
                    loss, trainable_params(self.model)
                )
            ]
            for i in range(len(grads_norm)):
                sensitivity[i] = (
                    self.miu * sensitivity[i] + (1 - self.miu) * grads_norm[i].abs()
                )
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            losses += loss.item()
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
        acc = (tot_correct / test_sample) if test_sample>0.0 else 0.0
        self.model.train()
        return (
            losses,
            acc,
            matched,
            record_stats,
            pred,
            targ,
            (test_sample, np.array(sensitivity.cpu())),
        )

    def test_and_cost_sensitive_stats_t(self, global_coef_matrix=None):
        self.model.eval()
        tot_correct, losses, test_sample = 0, 0.0, 0
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if self.use_custom_classifier:
                features = self.model.get_representation_features(inputs)
                outputs = self.classifier(features)
            else: outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            grads = list(torch.autograd.grad(loss, self.model.parameters()))
            for named_coeffs, grad in zip(self.coeff_matrix.items(), grads):
                name, coeffs = named_coeffs
                grad_norm = torch.tensor(
                    [torch.norm(x) ** 2 for x in grad], device=self.device
                )
                self.coeff_matrix[name] = (
                    self.miu * coeffs + (1 - self.miu) * grad_norm.abs()
                ).to(coeffs.device)
            _, predicted = torch.max(outputs, 1)
            cost_matrix = torch.ones(
                self.cost_matrix.shape, device=self.cost_matrix.device
            )
            for x, y in zip(labels, predicted):
                cost_matrix[x][y] = cost_matrix[x][y] + 1
                if x == y:
                    tot_correct += 1
            test_sample += len(labels)
        acc = (tot_correct / test_sample) if test_sample>0.0 else 0.0
        self.model.train()
        # for k , v in self.coeff_matrix.items():
        #     print(f"\nCoefficient:\n{k}: {v.shape}")
        return (test_sample, self.coeff_matrix.values()), cost_matrix

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
