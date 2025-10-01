import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision import transforms
from collections import OrderedDict
from transformers import BertModel
import subprocess
from flearn.data.dataset import CLASSES


func = lambda x: x.detach().clone()


class elu(nn.Module):
    def __init__(self) -> None:
        super(elu, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, x, 0.2 * (torch.exp(x) - 1))


class linear(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(linear, self).__init__()
        self.w = nn.Parameter(
            torch.randn(out_c, in_c) * torch.sqrt(torch.tensor(2 / in_c))
        )
        self.b = nn.Parameter(torch.randn(out_c))

    def forward(self, x):
        return F.linear(x, self.w, self.b)


class MLP_MNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_MNIST, self).__init__()
        self.fc1 = linear(28 * 28, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x

    def set_params(self, model_params=None):
        if model_params is not None:
            with torch.no_grad():
                for param, value in zip(self.parameters(), model_params):
                    # print(type(value))
                    if isinstance(value, np.ndarray):
                        param.copy_(torch.from_numpy(value))
                    elif isinstance(value, torch.Tensor):
                        param.copy_(value)
                    else:
                        print("Variable is neither a numpy.ndarray nor a torch.Tensor")
                        # print("check loaded model:  ->" , model_params)
                        self.load_state_dict(model_params)
                        break

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return func(x)


ARGS = {
    "mnist": (1, 256, 10),
    "emnist": (1, 256, 62),
    "fmnist": (1, 256, 10),
    "cifar": (3, 400, 10),
    "cifar10": (3, 400, 10),
    "cifar100": (3, 400, 100),
    "areview": (768, 2),  # BERT-base hidden size is 768, 2 classes
}


class LeNet5(nn.Module):
    def __init__(self, dataset) -> None:
        super(LeNet5, self).__init__()
        self.feature_dim = 84
        self.net = nn.Sequential(
            nn.Conv2d(ARGS[dataset][0], 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ARGS[dataset][1], 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(84, ARGS[dataset][2])

    def forward(self, x):
        x = self.net(x)
        return self.fc(x)

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        x = self.net[:10](x)
        return x
    
    def get_feature_dim(self):
        return self.feature_dim
    
    def classifier(self, x):
        """Apply the final classification layer"""
        return self.fc(x)
   

class BERTClassifier(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.feature_dim = ARGS[dataset][0]  # BERT hidden size
        try:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        except OSError as e:
            if "Unauthorized" in str(e) or "401" in str(e):
                print("Authorization error when downloading bert-base-uncased from Hugging Face.")
                print("Please login using: huggingface-cli login")
                subprocess.run(["huggingface-cli", "login"])
                # Try again after login
                self.bert = BertModel.from_pretrained("bert-base-uncased")
            else:
                raise
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(ARGS[dataset][0], ARGS[dataset][1])  # num_classes

    def forward(self, x):
        # x is a dict: input_ids, attention_mask (and maybe token_type_ids)
        bert_output = self.bert(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x.get('token_type_ids', None),
        )
        pooled_output = bert_output.pooler_output  # [batch, hidden_dim]
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        with torch.no_grad():
            bert_output = self.bert(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                token_type_ids=x.get('token_type_ids', None),
            )
            return bert_output.pooler_output  # feature vector

    def get_feature_dim(self):
        return self.feature_dim

    def classifier(self, x):
        return self.fc(x)


class TextCNN(nn.Module):
    def __init__(self, dataset):
        super(TextCNN, self).__init__()
        # ARGS[dataset] = (embedding_dim, num_classes)
        self.embedding_dim = ARGS[dataset][0]
        self.num_classes = ARGS[dataset][1]
        self.feature_dim = 100  # chosen size of concatenated conv outputs
        
        # Parameters for TextCNN
        vocab_size = ARGS[dataset][2]  # vocab size needed for embedding layer
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.conv1 = nn.Conv2d(1, 100, (3, self.embedding_dim))  # kernel size 3
        self.conv2 = nn.Conv2d(1, 100, (4, self.embedding_dim))  # kernel size 4
        self.conv3 = nn.Conv2d(1, 100, (5, self.embedding_dim))  # kernel size 5

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, self.num_classes)  # 100*3 filters

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch, 1, seq_len, embed_dim]

        x1 = F.relu(self.conv1(x)).squeeze(3)  # [batch, 100, seq_len-3+1]
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch, 100]

        x2 = F.relu(self.conv2(x)).squeeze(3)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)

        x3 = F.relu(self.conv3(x)).squeeze(3)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        x = torch.cat((x1, x2, x3), 1)  # [batch, 300]
        x = self.dropout(x)
        return self.fc(x)

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        with torch.no_grad():
            x = self.embedding(x).unsqueeze(1)
            x1 = F.relu(self.conv1(x)).squeeze(3)
            x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
            x2 = F.relu(self.conv2(x)).squeeze(3)
            x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
            x3 = F.relu(self.conv3(x)).squeeze(3)
            x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
            feat = torch.cat((x1, x2, x3), 1)
            return feat

    def get_feature_dim(self):
        return 300

    def classifier(self, x):
        return self.fc(x)


class MLP_CIFAR10(nn.Module):
    def __init__(self) -> None:
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = linear(32 * 32 * 3, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return func(x)


class MLP_CIFAR100(nn.Module):
    def __init__(self) -> None:
        super(MLP_CIFAR100, self).__init__()
        self.fc1 = linear(32 * 32 * 3, 512)
        self.fc2 = linear(512, 256)
        self.fc3 = linear(256, 100)
        self.flatten = nn.Flatten()
        self.activation = elu()
        self.classifier = None

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return func(x)


class MNIST_SOLVER(nn.Module):
    def __init__(self) -> None:
        super(MNIST_SOLVER, self).__init__()
        self.fc1 = linear(28 * 28, 128)
        self.fc2 = linear(128, 128)
        self.fc3 = linear(128, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]


class TorchResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrain=True, num_classes=None):
        """
        Initialize ResNet model with flexible architecture and pretraining options
        
        Args:
            model_name (str): Name of the ResNet architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrain (bool): Whether to use pretrained weights
            num_classes (int, optional): Number of classes for the final layer
        """
        super(TorchResNet, self).__init__()
        
        # Dictionary mapping model names to their corresponding functions
        self.model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        # Dictionary mapping model names to their feature dimensions
        self.feature_dims = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048
        }
        
        if model_name not in self.model_dict:
            raise ValueError(f"Unsupported model name. Choose from: {list(self.model_dict.keys())}")
            
        # Initialize the model with or without pretrained weights
        if pretrain:
            weights = f"IMAGENET1K_V1"
            self.resnet = self.model_dict[model_name](weights=weights)
        else:
            self.resnet = self.model_dict[model_name](weights=None)
            
        # Modify the last fully connected layer if num_classes is specified
        self.feature_dim = self.resnet.fc.in_features
        # self.resnet.fc = nn.Identity()
        if num_classes is not None:
            # print(f'Got number of classes')
            self.resnet.fc = nn.Linear(self.feature_dim, num_classes)
        else: 
            print(f'Required number of classes not set!')
            raise RuntimeError
        self.model_name = model_name

    def forward(self, x):
        return self.resnet(x)
    
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_params_t(self):
        """
        Get parameters of the feature extraction layers (everything except the final classifier)
        Returns:
            Iterator of parameters from the feature extraction layers
        """
        # Get all layers except the final fully connected layer
        representation_params = []
        for name, param in self.resnet.named_parameters():
            if not name.startswith('fc.'):
                representation_params.append(param)
        return representation_params
    
    def get_classifier_params_t(self):
        """
        Get parameters of the classifier layer (final fully connected layer)
        Returns:
            Iterator of parameters from the classifier layer
        """
        # Get only the final fully connected layer parameters
        classifier_params = []
        for name, param in self.resnet.named_parameters():
            if name.startswith('fc.'):
                classifier_params.append(param)
        return classifier_params
    
        
    def get_representation_features(self, x):
        """Extract features before the final classification layer"""
        # Pass through all layers up to but not including the fully connected (fc) layer
        for name, layer in self.resnet.named_children():
            if name != 'fc':  # Skip the classifier layer
                x = layer(x)
        return torch.flatten(x, 1)
    
    # def get_representation_features(self, x):
    #     """Extract features before the final classification layer"""
    #     # Use the ResNet forward method to pass through the layers up to the classifier
    #     # We avoid the last fully connected (fc) layer
    #     x = self.resnet.conv1(x)     # First conv layer
    #     x = self.resnet.bn1(x)       # Batch norm after first conv
    #     x = self.resnet.relu(x)      # ReLU activation
    #     x = self.resnet.maxpool(x)   # Maxpooling layer
        
    #     # Pass through the rest of the layers (until the last fully connected layer)
    #     x = self.resnet.layer1(x)
    #     x = self.resnet.layer2(x)
    #     x = self.resnet.layer3(x)
    #     x = self.resnet.layer4(x)
        
    #     # Flatten the output to match the input dimension of the fully connected layer
    #     x = torch.flatten(x, 1)
        
    #     return x


    
    def classifier(self, x):
        return self.resnet.fc(x)
    
    def get_feature_dim(self):
        """Return the feature dimension based on the model architecture"""
        return self.feature_dim


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


Resnet8_avg_pool = {"mnist": 8, "cifar10": 9, "cifar": 9, "cifar100": 9, "fmnist": 8, "har": None, "areview": None,}


class ResNet8(nn.Module):
    def __init__(self, block, num_blocks, dataset, num_classes=10, input_channels=3):
        super(ResNet8, self).__init__()
        self.dataset = dataset
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.linear = nn.Linear(128 * block.expansion, num_classes)
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.PReLU(),
            self.layer1,
            self.layer2
        )
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        # print(f'\nshape1: {out.shape}')
        out = F.avg_pool2d(out, Resnet8_avg_pool[self.dataset])
        out = out.view(out.size(0), -1)
        # print(f'\nshape2: {out.shape}')
        out = self.linear(out)
        return out

    
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, Resnet8_avg_pool[self.dataset])
        out = out.view(out.size(0), -1)
        return out
    
    
    def get_representation_params_t(self):
        return self.features.parameters()
    
    def get_classifier_params_t(self):
        return self.linear.parameters()

    def classifier(self, x):
        return self.linear(x)
    
    def get_feature_dim(self):
        return 128

# Define ResNet18 model for CIFAR-10
class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, input_channels=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature_dim = 512

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return func(out)    
    
    def classifier(self, x):
        """Apply the final classification layer"""
        return self.fc(x)
    
    def get_feature_dim(self):
        """Return the feature dimension based on the model architecture"""
        return self.feature_dim


class PeFLL_EmbedNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        embed_dim: int,
        embed_y: int,
        embed_num_kernels: int,
    ):
        super(PeFLL_EmbedNetwork, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embed_y = embed_y
        self.embed_num_kernels = embed_num_kernels

        in_channels = self.input_channels + self.embed_y * self.num_classes

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, self.embed_num_kernels, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                self.embed_num_kernels,
                2 * self.embed_num_kernels,
                5,
            ),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * self.embed_num_kernels * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, self.embed_dim),
        )
        self.resize = transforms.Resize((32, 32))

    def forward(self, x, y):
        if self.embed_y:
            h, w = x.shape[2], x.shape[3]
            if h < 32 or w < 32:
                x = self.resize(x)
            y = F.one_hot(y, self.num_classes)
            y = y.view(y.shape[0], y.shape[1], 1, 1)
            c = torch.zeros(
                (x.shape[0], y.shape[1], x.shape[2], x.shape[3]), device=x.device
            )
            c += y
            x = torch.cat((x, c), dim=1)
        return self.model(x)

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]


class PeFLL_HyperNetwork(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        hyper_hidden_dim: int,
        hyper_num_hidden_layers: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hyper_hidden_dim = hyper_hidden_dim
        self.hyper_num_hidden_layers = hyper_num_hidden_layers

        mlp_layers = [nn.Linear(self.embed_dim, self.hyper_hidden_dim)]
        for _ in range(self.hyper_num_hidden_layers):
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Linear(self.hyper_hidden_dim, self.hyper_hidden_dim))
        self.mlp = nn.Sequential(*mlp_layers)

        parameters, self.params_name = [], []
        for key, param in backbone.named_parameters():
            parameters.append(param)
            self.params_name.append(key)
        self.params_shape = {
            name: backbone.state_dict()[name].shape for name in self.params_name
        }
        self.params_generator = nn.ParameterDict()
        for name, param in zip(self.params_name, parameters):
            self.params_generator[name.replace(".", "-")] = nn.Linear(
                self.hyper_hidden_dim, param.numel()
            )

    def forward(self, embedding):
        features = self.mlp(embedding)
        return OrderedDict(
            (
                name,
                self.params_generator[name.replace(".", "-")](features).reshape(
                    self.params_shape[name]
                ),
            )
            for name in self.params_name
        )

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]


class DistHN_HyperNetwork(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        hyper_hidden_dim: int,
        hyper_num_hidden_layers: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hyper_hidden_dim = hyper_hidden_dim
        self.hyper_num_hidden_layers = hyper_num_hidden_layers

        mlp_layers = [nn.Linear(self.input_dim, self.hyper_hidden_dim)]
        for _ in range(self.hyper_num_hidden_layers):
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Linear(self.hyper_hidden_dim, self.hyper_hidden_dim))
        self.mlp = nn.Sequential(*mlp_layers)

        parameters, self.params_name = [], []
        for key, param in backbone.named_parameters():
            parameters.append(param)
            self.params_name.append(key)
        self.params_shape = {
            name: backbone.state_dict()[name].shape for name in self.params_name
        }
        self.params_generator = nn.ParameterDict()
        for name, param in zip(self.params_name, parameters):
            self.params_generator[name.replace(".", "-")] = nn.Linear(
                self.hyper_hidden_dim, param.numel()
            )

    def forward(self, embedding):
        features = self.mlp(embedding)
        return OrderedDict(
            (
                name,
                self.params_generator[name.replace(".", "-")](features).reshape(
                    self.params_shape[name]
                ),
            )
            for name in self.params_name
        )

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]


def MnistResNet8():
    print("mnist")
    return ResNet8(BasicBlock, [1, 1], input_channels=1, num_classes=10)

class SimplexModel(nn.Module):
    def __init__(self, args):
        super(SimplexModel, self).__init__()
        self.args = args
        device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Fix the way we access model and dataset names from args
        dataset_name = args.get("dataset").get("name")  # Default to 'mnist' if not specified
        model_name = args.get("model").get("name")  # Default to resnet18 if not specified
        self.endpoints = args.get("floco", {}).get("endpoints", 3)
        seed = args.get("common", {}).get("seed", 42)
        
        base_model = get_model_by_name(dataset_name, device, model_name)
        self.net = base_model
        self.feature_dim = base_model.get_feature_dim()
        self.fc = SimplexLinear(
            endpoints=self.endpoints,
            in_features=self.feature_dim,
            out_features=CLASSES[dataset_name],
            bias=True,
            seed=seed,
        )
        self.subregion_parameters = None

    def forward(self, x):
        endpoints = self.endpoints
        if self.subregion_parameters is None:
            if self.training:
                sample = np.random.exponential(scale=1.0, size=endpoints)
                self.fc.alphas = sample / sample.sum()
            else:
                self.fc.alphas = tuple([1 / endpoints for _ in range(endpoints)])
        else:
            if self.training:
                self.fc.alphas = self._sample_L1_ball(*self.subregion_parameters)
            else:
                self.fc.alphas = self.subregion_parameters[0]
        x = self.net.get_representation_features(x)
        return self.fc(x)
    
    def _sample_L1_ball(self, center, radius):
        u = np.random.uniform(-1, 1, len(center))
        u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
        return center + np.random.uniform(0, radius) * u 

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        return self.net.get_representation_features(x)

    def get_feature_dim(self):
        return self.feature_dim

    def classifier(self, x):
        return self.fc(x)

class SimplexLinear(nn.Linear):
    def __init__(self, endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoints = endpoints
        self.alphas = tuple([1 / endpoints for _ in range(endpoints)])
        self._weights = nn.ParameterList(
            [self._initialize_weight(self.weight, seed + i) for i in range(endpoints)]
        )
    
    def _initialize_weight(self, init_weight: torch.Tensor, seed: int) -> torch.nn.Parameter:
        weight = torch.nn.Parameter(torch.zeros_like(init_weight))
        torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(weight)
        return weight

    @property
    def weight(self) -> torch.nn.Parameter:
        return sum(alpha * weight for alpha, weight in zip(self.alphas, self._weights))
    

class SimplexcModel(nn.Module):
    def __init__(self, dataset_name, model_name, device, endpoints=20, seed=42):
        super(SimplexcModel, self).__init__()        
        self.endpoints = endpoints
        
        base_model = get_model_by_name(dataset_name, device, model_name)
        self.net = base_model
        self.feature_dim = base_model.get_feature_dim()
        self.fc = SimplexLinear(
            endpoints=self.endpoints,
            in_features=self.feature_dim,
            out_features=CLASSES[dataset_name],
            bias=True,
            seed=seed,
        )

    def forward(self, x):
        endpoints = self.endpoints
        if self.training:
            sample = np.random.exponential(scale=1.0, size=endpoints)
            self.fc.alphas = sample / sample.sum()
        else:
            self.fc.alphas = tuple([1 / endpoints for _ in range(endpoints)])

        x = self.net.get_representation_features(x)
        return self.fc(x)

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]

    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

    def get_representation_features(self, x):
        return self.net.get_representation_features(x)

    def get_feature_dim(self):
        return self.feature_dim

    def classifier(self, x):
        return self.fc(x)


MODEL_DICT = {
    "mnist": MLP_MNIST,
    "cifar": MLP_CIFAR10,
    "cifar10": MLP_CIFAR10,
    "cifar100": MLP_CIFAR100,
    "nmnist": MLP_MNIST,
}
# CLASSES = {
#     "mnist": 10,
#     "nmnist": 10,
#     "cifar10": 10,
#     "cifar": 10,
#     "cifar100": 100,
#     "emnist_digits": 26,
#     "emnist_balanced": 47,
# }


def get_model(dataset, device):
    return MODEL_DICT[dataset]().to(device)


CHANNELS = {
    "mnist": 1,
    "cifar": 3,
    "cifar10": 3,
    "cifar100": 3,
    "fmnist": 1,
    "areview": 1,
    "har": 1,
}

MLP = {
    "mnist": MLP_MNIST(),
    "cifar": MLP_CIFAR10(),
    "cifar10": MLP_CIFAR10(),
    "cifar100": MLP_CIFAR100(),
    "fmnist": None,
    "areview": None,
    "har": None,
}


def get_model_by_name(dataset, device, model):
    # print(f"\nIn model.py to build: {model} model")
    MODEL_DICT = {"mlp": MLP[dataset], 
                "resnet18": ResNet18(BasicBlock, [2, 2, 2, 2], input_channels=CHANNELS[dataset], num_classes=CLASSES[dataset]),
                'tresnet18p': TorchResNet(model_name='resnet18', pretrain=True, num_classes=CLASSES[dataset]),
                'tresnet34p': TorchResNet(model_name='resnet34', pretrain=True, num_classes=CLASSES[dataset]),
                'tresnet50p': TorchResNet(model_name='resnet50', pretrain=True, num_classes=CLASSES[dataset]),
                'tresnet101p': TorchResNet(model_name='resnet101', pretrain=True, num_classes=CLASSES[dataset]),
                'tresnet18': TorchResNet(model_name='resnet18', pretrain=False, num_classes=CLASSES[dataset]),
                'tresnet34': TorchResNet(model_name='resnet34', pretrain=False, num_classes=CLASSES[dataset]),
                'tresnet50': TorchResNet(model_name='resnet50', pretrain=False, num_classes=CLASSES[dataset]),
                'tresnet101': TorchResNet(model_name='resnet101', pretrain=False, num_classes=CLASSES[dataset]),
                "resnet8": ResNet8(BasicBlock, [1, 1], dataset=dataset, input_channels=CHANNELS[dataset], num_classes=CLASSES[dataset]),
                "lenet5": LeNet5(dataset=dataset),
                # Add BERT and TextCNN here, assuming you have defined these classes:
                "bert": BERTClassifier(dataset=dataset),
                # "textcnn": TextCNN(vocab_size=VOCAB_SIZE[dataset], embed_dim=EMBED_DIM[dataset], num_classes=CLASSES[dataset]),
    }
    c_model = MODEL_DICT[model].to(device)
    # print(c_model)
    return c_model

