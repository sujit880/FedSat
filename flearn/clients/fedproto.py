from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size
import torch
import copy


class ProtoClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.loss_mse = torch.nn.MSELoss()

    def init_client_specific_params(
        self,
        lamda: float,
        num_classes: int,
        **kwargs,
    ) -> None:
        self.lamda = lamda
        self.num_classes = num_classes

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
    
    def solve_inner_fedproto_t(self, global_protos, num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in the training process
            2: bytes_write: number of bytes transmitted
        '''
        # print("0 mem:", self.get_free_vram_gb(self.device))
        bytes_w = graph_size(self.model)
        train_sample_size = 0
        proto_sums = {}
        proto_counts = {}
        agg_protos_label = {}
        self.model.train()
        for epoch in range(num_epochs): 
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.noisy:
                    # print("adding noise")
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level # Adding noise to input for DP 
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                features = self.model.get_representation_features(inputs)
                loss = self.criterion(outputs, labels)
                
                if len(global_protos) == 0:
                    loss_p = 0*loss
                else:
                    proto_new = copy.deepcopy(features.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss_p = self.loss_mse(proto_new, features)

                loss += loss_p * self.lamda

                loss.backward()
                self.optimizer.step()

                train_sample_size += len(labels)

                # Calculate mean proto
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in proto_sums:
                        proto_sums[label] += features[i].detach()
                        proto_counts[label] += 1
                    else:
                        proto_sums[label] = features[i].clone().detach()
                        proto_counts[label] = 1

        # After all epochs, compute the mean of the prototypes for each label
        for label, proto_sum in proto_sums.items():
            agg_protos_label[label] = proto_sum / proto_counts[label]
        soln = self.get_model_params()
        comp = num_epochs * (train_sample_size// batch_size) * batch_size
        bytes_r = graph_size(self.model)
        return (self.num_samples, soln), (bytes_w, comp, bytes_r), agg_protos_label
    
    def test_model_(self, model:torch.nn.Module, modelInCPU: bool = False) -> tuple[int, int]:
        """tests the current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        """
        # print(f'Calling Local test')
        if modelInCPU:
            model = model.to(self.device)

        tot_correct, loss, test_sample = 0, 0.0, 0
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            loss += self.criterion(outputs, labels).item()
            test_sample += len(labels)

        if modelInCPU:
            model = model.cpu()

        return tot_correct, test_sample
    
    def predict_(self, x, prototypes):
        """
        Predict the label for a batch of instances x by measuring the L2 distance to each class prototype.
        sssss
        x: A batch of instances (a tensor where each row is an instance).
        prototypes: A dictionary where keys are class labels and values are the corresponding prototype vectors.
        """
        # Sort the prototypes by label values (keys) to ensure the order corresponds to class labels.
        sorted_prototypes = sorted(prototypes.items(), key=lambda item: item[0])

        # Get the sorted prototypes (list of tuples (label, prototype))
        sorted_labels, sorted_vectors = zip(*sorted_prototypes)
        # Ensure that all prototype vectors are tensors (if they are not already)
        sorted_vectors = [torch.tensor(prototype, dtype=torch.float32, device=self.device) if not isinstance(prototype, torch.Tensor) else prototype for prototype in sorted_vectors]

        # Convert the prototype vectors to a tensor
        sorted_vectors = torch.stack(sorted_vectors)

        # Prepare a tensor to store the distances
        distances = torch.zeros((x.size(0), len(sorted_vectors)), dtype=torch.float32, device=self.device)

        # For each prototype, compute the L2 distance to the feature vectors of the batch
        for idx, prototype in enumerate(sorted_vectors):
            # Expand the prototype to the batch size and compute the L2 distance
            prototype = prototype.to(self.device)
            prototype_expanded = prototype.expand_as(x)
            # print(f'{x.device}, {prototype_expanded.device}, {prototype.device}')
            distances[:, idx] = torch.norm(x - prototype_expanded, dim=1)
    
        # Get the index of the minimum distance for each instance
        predicted_labels = distances.argmin(dim=1)
        # print(f'{distances.device}, {predicted_labels.device}')

        return predicted_labels, distances

    def test_proto(self, model:torch.nn.Module, prototypes, modelInCPU: bool = False) -> tuple[int, int]:
        """tests the current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        """
        # print(f'Calling Local test')
        if modelInCPU:
            model = model.to(self.device)

        tot_correct, loss, test_sample = 0, 0.0, 0
        for inputs, labels in self.valloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if len(labels)<=1: continue
            if len(prototypes)==self.num_classes:
                features = self.model.get_representation_features(inputs)
                predicted, _ = self.predict(features, prototypes)
                # print(f'\nLabels: {labels} \nPredicted: {predicted}')
            else:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            test_sample += len(labels)

        if modelInCPU:
            model = model.cpu()

        return tot_correct, test_sample
    
    def predict(self, x, prototypes):
        """
        Predict the label for a batch of instances x by measuring the L2 distance to each class prototype.
        
        x: A batch of instances (a tensor where each row is an instance).
        prototypes: A dictionary where keys are class labels and values are the corresponding prototype vectors.
        """
        # Prepare a tensor to store the distances
        distances = torch.zeros((x.size(0), len(prototypes)), dtype=torch.float32, device=self.device)

        # For each prototype, compute the L2 distance to the feature vectors of the batch
        for idx, (label, prototype) in enumerate(prototypes.items()):
            # Expand the prototype to the batch size and compute the L2 distance
            prototype = prototype = torch.tensor(prototype, dtype=torch.float32, device=self.device) if not isinstance(prototype, torch.Tensor) else prototype.to(self.device)
            prototype_expanded = prototype.expand_as(x)
            distances[:, label] = torch.norm(x - prototype_expanded, dim=1)

        # Get the index of the minimum distance for each instance
        predicted = distances.argmin(dim=1)

        return  predicted, distances