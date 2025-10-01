import torch
import torch.nn as nn
import torch.nn.functional as F

class KCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, num_classes=10):
        super(KCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.kl_weight = 1e-2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        
        # Classifier
        self.classifier = nn.Linear(input_dim, num_classes)

        # Label-wise Latent space predictor
        self.label_to_latent = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Tanh()  # To bound the output
        )

    def generate_from_label(self, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float().to(self.label_to_latent[0].weight.dtype)
        z = self.label_to_latent(c_onehot)
        return self.decode(z, c)
    
    def encode(self, x, c):
        x_c = torch.cat([x, F.one_hot(c, num_classes=self.num_classes).float()], dim=1)
        x = self.encoder(x_c)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        z_c = torch.cat([z, F.one_hot(c, num_classes=self.num_classes).float()], dim=1)
        return self.decoder(z_c)
    
    def forward(self, x, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float().to(self.label_to_latent[0].weight.dtype)
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        pred_z = self.label_to_latent(c_onehot)
        recon_x = self.decode(z, c)
        class_pred = self.classifier(recon_x)
        return recon_x, mu, log_var, z, pred_z, class_pred

    def distillation_loss(self, recon_x, target_features):
        # Use MSE loss to distill knowledge from ResNet features to CVAE-generated features
        return F.mse_loss(recon_x, target_features)

    def loss_function(self, recon_x, x, mu, logvar, z, pred_z, c, class_pred, target_features=None):
        # Reconstruction loss (using MSE to retain feature similarity)
        recon_loss = F.mse_loss(recon_x, x)
        z_loss = F.mse_loss(pred_z, z.clone().detach())
        # KL Divergence
        KLD = None
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Classification loss
        classification_loss = F.cross_entropy(class_pred, c)
        
        # Distillation loss (aligns generated features with ResNet mean features)
        distillation_loss = 0.0
        if target_features is not None:
            distillation_loss = self.distillation_loss(recon_x, target_features)
        
        # Total loss: Incorporate distillation loss
        total_loss = (
            recon_loss * 0.25 +  # Reduced weight for reconstruction loss
            z_loss * 0.5 + # Train latent space predictor
            classification_loss * 0.9 +  # Increased weight for classification loss
            distillation_loss * 0.5  # Weight for the distillation loss
        )
        
        return total_loss, recon_loss, KLD, classification_loss, distillation_loss

    def update_kl_weight(self):
        if self.kl_weight >= 0.1:
            return 0.1
        else:
            self.kl_weight += 1e-2 * 2
            return self.kl_weight


class ICVAE(nn.Module):
    def __init__(self, device, input_dim, latent_dim=128, num_classes=10):
        super(ICVAE, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.kl_weight = 1e-2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),  # Input with class information
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Encoded representation of input
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Latent space (default `mu` and `log_var` from encoder)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        
        # # Class-specific latent prediction layers (corrected input dimension to num_classes)
        # self.class_mu_layer = nn.Sequential(
        #         nn.Linear(num_classes, 256),  # class vector -> latent_mu
        #         nn.Linear(256, latent_dim),
        # )
        # self.class_log_var_layer = nn.Sequential(
        #         nn.Linear(num_classes, 256),  # class vector -> latent_logvar
        #         nn.Linear(256, latent_dim),
        # )
        self.class_z = nn.Sequential(
                nn.Linear(num_classes, 256),  # class vector -> class_z
                nn.Linear(256, latent_dim),
        )
        self.class_latent = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        
        # Classifier (for auxiliary task of class prediction)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def get_c_mu(self, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        # Predict class-specific `mu` values based on class vector `c`
        # c is one-hot encoded with shape [batch_size, num_classes]
        # c = c.to(self.class_mu_layer.weight.dtype)
        return self.class_mu_layer(c_onehot)  # Output shape: [batch_size, latent_dim]

    def get_c_logvar(self, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        # Predict class-specific `log_var` values based on class vector `c`
        # c is one-hot encoded with shape [batch_size, num_classes]
        # c = c.to(self.class_log_var_layer.weight.dtype)
        return self.class_log_var_layer(c_onehot)  # Output shape: [batch_size, latent_dim]
    
    def get_c_z(self, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        # Predict class-specific `latant vector` values based on class vector `c`
        # c is one-hot encoded with shape [batch_size, num_classes]
        return self.class_z(c_onehot)  # Output shape: [batch_size, latent_dim]
    
    def get_c_latent_dim(self,c):
        batch_size = c.size(0)  # Get the batch size from the class labels tensor
        x = torch.rand(batch_size, self.input_dim).to(self.device)  # Random tensor with the correct batch size    
        x_c = torch.cat([x, F.one_hot(c, num_classes=self.num_classes).float()], dim=1)
        latent_dim = self.class_latent(x_c)        
        return latent_dim

    def reparameterize(self, mu, log_var):
        # Sample from the class-specific Gaussian distribution using the reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_from_label_1(self, c):      
        # Pass class one-hot encoding through the class-specific layers to get class-specific `mu` and `log_var`
        mus = self.get_c_mu(c)  # Class-specific `mu` for each class
        log_vars = self.get_c_logvar(c)  # Class-specific `log_var` for each class
        mus = mus + 0.5 * torch.randn_like(mus).to(c.device)
        # Now sample `z` from the class-specific distribution for each example and add noise for diversity
        z = self.reparameterize(mus, log_vars) 
        
        # Decode the latent variable `z` with class information
        return self.decode(z, c)
    
    def generate_from_label_2(self, c):      
        # Pass class one-hot encoding through the class-specific layers to get class-specific `latent vector`
        z = self.get_c_z(c)  # Class-specific `mu` for each class
        z = z*0.5 + 0.5*torch.randn_like(z).to(c.device)        
        # Decode the latent variable `z` with class information
        return self.decode(z, c)
    
    def generate_from_label(self, c):      
        # Pass class one-hot encoding through the class-specific layers to get class-specific `latent vector`
        latent_dim = self.get_c_latent_dim(c)  
        latent_dim = latent_dim + 0.1*torch.randn_like(latent_dim).to(c.device)  
        mu = self.fc_mu(latent_dim)
        log_var = self.fc_log_var(latent_dim)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, c)

    def decode(self, z, c):
        # Concatenate the latent variable `z` with class one-hot information
        z_c = torch.cat([z, F.one_hot(c, num_classes=self.num_classes).float()], dim=1)
        return self.decoder(z_c)

    def encode_1(self, x, c):
        # Concatenate input `x` with class one-hot encoding
        x_c = torch.cat([x, F.one_hot(c, num_classes=self.num_classes).float()], dim=1)
        x = self.encoder(x_c)
        
        # Get the `mu` and `log_var` from the encoder
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var
    
    def forward_1(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, c)
        return recon_x, mu, log_var, z
    
    def encode(self, x, c):
        # Concatenate input `x` with class one-hot encoding
        x_c = torch.cat([x, F.one_hot(c, num_classes=self.num_classes).float()], dim=1)
        x = self.encoder(x_c)
        
        # Get the `mu` and `log_var` from the encoder
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var, x
    
    def forward(self, x, c):
        mu, log_var, x = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, c)
        return recon_x, mu, log_var, x
   
class Proto(nn.Module):
    def __init__(self, device, out_dim, latent_dim=128, num_classes=10):
        super(Proto, self).__init__()
        self.device = device
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.decoder = nn.Sequential(
            nn.Linear(num_classes, 256),
            # nn.BatchNorm1d(256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            # nn.BatchNorm1d(512),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
        
        self.classifier = nn.Linear(out_dim, num_classes)

    def generate_from_label(self, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float().to(self.device)
        return self.decoder(c_onehot)
    
    def forward(self, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float().to(self.device)
        features = self.decoder(c_onehot)
        logits = self.classifier(features)
        return features, logits

class CVAEH(nn.Module):
    def __init__(self, device, input_dim, latent_dim=128, num_classes=10, hidden_dims=[512, 256]):
        super(CVAEH, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        print(f"Using Hidden Dimensions")
        # If hidden_dims not provided, use default scaled values based on input_dim
        if hidden_dims is None:
            hidden_dims = [
                max(128, input_dim * 2),   # First hidden layer (encoder)
                max(64, input_dim)         # Second hidden layer (encoder output / decoder input)
            ]
        
        self.hidden_dims = hidden_dims

        # Encoder: input_dim + num_classes → hidden_dims[0] → hidden_dims[1]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
        )

        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder: latent_dim + num_classes → hidden_dims[1] → hidden_dims[0] → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        x_c = torch.cat([x, c_onehot], dim=1)
        x_enc = self.encoder(x_c)
        mu = self.fc_mu(x_enc)
        log_var = self.fc_log_var(x_enc)
        return mu, log_var

    def decode(self, z, c):
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        z_c = torch.cat([z, c_onehot], dim=1)
        return self.decoder(z_c)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, c)
        return recon_x, mu, log_var, z

    def target_z(self, c):
        z = torch.randn(c.size(0), self.latent_dim).to(self.device)
        x = torch.randn(c.size(0), self.input_dim).to(self.device)
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var) + 0.1 * z
        return z

    def generate_from_label(self, c):
        z = self.target_z(c)
        return self.decode(z, c)
    
class CVAE_(nn.Module):
    def __init__(self, device, input_dim, latent_dim=128, num_classes=10):
        super(CVAE, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder: The input size is input_dim + num_classes because class information is concatenated with input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),  # Concatenate input with class information
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Encoded representation of input
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Latent space: Fully connected layers to get mu and log_var for variational inference
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        
        # Decoder: The input size is latent_dim + num_classes because class information is concatenated with latent variable
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),  # Concatenate latent and class info
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),  # Output reconstructed input
        )
        
    def reparameterize(self, mu, log_var):
        # Reparameterization trick to sample from Gaussian distribution
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, c):
        # Concatenate input `x` with one-hot class encoding `c`
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()  # One-hot class labels
        x_c = torch.cat([x, c_onehot], dim=1)  # Concatenate along the feature dimension
        
        # Pass through encoder layers
        x_enc = self.encoder(x_c)
        
        # Get `mu` and `log_var` for the latent space
        mu = self.fc_mu(x_enc)  # Output of shape [batch_size, latent_dim]
        log_var = self.fc_log_var(x_enc)  # Output of shape [batch_size, latent_dim]
        return mu, log_var
    
    def decode(self, z, c):
        # Concatenate the latent variable `z` with class information (one-hot vector)
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        z_c = torch.cat([z, c_onehot], dim=1)  # Concatenate along the feature dimension
        return self.decoder(z_c)
    
    def forward(self, x, c):
        # Encode and sample z using the reparameterization trick
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        
        # Decode z to reconstruct the input
        recon_x = self.decode(z, c)
        return recon_x, mu, log_var, z
    
    def target_z(self, c):
        # z = torch.randn(c.size(0), self.latent_dim).to(self.device)  # Sample from standard normal
        x = torch.randn(c.size(0), self.input_dim).to(self.device)  # Sample from standard normal
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var) #+ 0.1*z
        return z
    
    def generate_from_label(self, c):
        z = self.target_z(c)        
        # Decode the generated latent variable with the class information
        return self.decode(z, c)    

class CVAE(nn.Module):
    def __init__(self, device, input_dim, latent_dim=128, num_classes=10):
        super(CVAE, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim1 = self.latent_dim*2
        self.hidden_dim2 = self.hidden_dim1*2
        self.num_classes = num_classes
        
        # Encoder: The input size is input_dim + num_classes because class information is concatenated with input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, self.hidden_dim2),  # Concatenate input with class information
            # nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.hidden_dim1),  # Encoded representation of input
            # nn.BatchNorm1d(self.hidden_dim1),
            nn.ReLU(),
        )
        
        # Latent space: Fully connected layers to get mu and log_var for variational inference
        self.fc_mu = nn.Linear(self.hidden_dim1, latent_dim)
        self.fc_log_var = nn.Linear(self.hidden_dim1, latent_dim)
        
        # Decoder: The input size is latent_dim + num_classes because class information is concatenated with latent variable
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, self.hidden_dim1),  # Concatenate latent and class info
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            # nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, input_dim),  # Output reconstructed input
        )
        
    def reparameterize(self, mu, log_var):
        # Reparameterization trick to sample from Gaussian distribution
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, c):
        # Concatenate input `x` with one-hot class encoding `c`
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()  # One-hot class labels
        x_c = torch.cat([x, c_onehot], dim=1)  # Concatenate along the feature dimension
        
        # Pass through encoder layers
        x_enc = self.encoder(x_c)
        
        # Get `mu` and `log_var` for the latent space
        mu = self.fc_mu(x_enc)  # Output of shape [batch_size, latent_dim]
        log_var = self.fc_log_var(x_enc)  # Output of shape [batch_size, latent_dim]
        return mu, log_var
    
    def decode(self, z, c):
        # Concatenate the latent variable `z` with class information (one-hot vector)
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        z_c = torch.cat([z, c_onehot], dim=1)  # Concatenate along the feature dimension
        return self.decoder(z_c)
    
    def forward(self, x, c):
        # Encode and sample z using the reparameterization trick
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        
        # Decode z to reconstruct the input
        recon_x = self.decode(z, c)
        return recon_x, mu, log_var, z
    
    def target_z(self, c):
        # z = torch.randn(c.size(0), self.latent_dim).to(self.device)  # Sample from standard normal
        x = torch.randn(c.size(0), self.input_dim).to(self.device)  # Sample from standard normal
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var) #+ 0.1*z
        return z
    
    def generate_from_label(self, c):
        z = self.target_z(c)        
        # Decode the generated latent variable with the class information
        return self.decode(z, c)    

class CLS(nn.Module):
    def __init__(self, device, input_dim, num_classes=10, **kwargs):
        super(CLS, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = input_dim//2
        self.num_classes = num_classes       
        self.head = nn.Linear(input_dim, self.hidden_dim)
        self.cls = nn.Linear(self.hidden_dim, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.head(x)
        x = self.relu(x)
        x = self.cls(x)
        return x
    
    def get_head_features(self, x):
        return self.head(x)
    
class LCLS(nn.Module):
    def __init__(self, input_dim, num_classes=10, **kwargs):
        super(LCLS, self).__init__()      
        self.cls = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.cls(x)
        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class CLS_V(nn.Module):
    def __init__(self, device, input_dim, num_classes=10, hidden_dim=None, dropout_prob=0.5, activation_fn=nn.ReLU, **kwargs):
        super(CLS_V, self).__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim or input_dim // 2  # Use input_dim//2 if hidden_dim is not provided
        self.dropout_prob = dropout_prob
        self.activation_fn = activation_fn
        
        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            self.activation_fn(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_fn(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
        # Initialize weights with Xavier initialization
        self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

from collections import OrderedDict
# HyperNetwork Definition
class ProtoHN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        embed_dim: int,
        hyper_hidden_dim: int,
        hyper_num_hidden_layers: int,
    ):
        super().__init__()
        self.input_cim = input_dim
        self.embed_dim = embed_dim
        self.hyper_hidden_dim = hyper_hidden_dim
        self.hyper_num_hidden_layers = hyper_num_hidden_layers

        self.embeddng = nn.Linear(input_dim, self.embed_dim)

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

    def forward(self, classifier_params):
        # Flatten and normalize classifier parameters
        flat_params = torch.cat([p.flatten() for p in classifier_params.values()])
        flat_params = F.normalize(flat_params, dim=0)
        embedding = self.embeddng(flat_params)
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
        
class ClassifierEmbedding(nn.Module):
    """Network to embed classifier parameters into latent space"""
    def __init__(self, input_dim, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, classifier_params):
        # Flatten and normalize classifier parameters
        flat_params = torch.cat([p.flatten() for p in classifier_params.values()])
        flat_params = F.normalize(flat_params, dim=0)
        return self.encoder(flat_params)
