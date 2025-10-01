import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class BasicDownscaleBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, negative_slope: float = 0.01
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BasicUpscaleBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, negative_slope: float = 0.1
    ) -> None:
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int = 0) -> None:
        super().__init__()

        # Define convolutions with different kernel sizes
        self.conv3x3 = nn.Conv2d(
            in_channels,
            out_channels // 3 + out_channels % 3,
            kernel_size=3,
            padding=0 + padding,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels,
            out_channels // 3,
            kernel_size=5,
            padding=1 + padding,
        )
        self.conv7x7 = nn.Conv2d(
            in_channels,
            out_channels // 3,
            kernel_size=7,
            padding=2 + padding,
        )

        self.conv4x4 = nn.Conv2d(
            out_channels,
            out_channels // 2 + out_channels % 2,
            kernel_size=4,
            padding=0,
        )
        self.conv6x6 = nn.Conv2d(
            out_channels,
            out_channels // 2,
            kernel_size=6,
            padding=1,
        )

        # BatchNorm and activation (applied after the sum)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolutions
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        out7 = self.conv7x7(x)

        # Sum all the outputs
        out_odd = torch.cat([out3, out5, out7], dim=1)

        out4 = self.conv4x4(out_odd)
        out6 = self.conv6x6(out_odd)

        combined = torch.cat([out4, out6], dim=1)

        # Apply BatchNorm and activation
        combined = self.batch_norm(combined)
        combined = self.activation(combined)

        return combined


class MultiDeconv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int = 0) -> None:
        super().__init__()

        # Define transposed convolutions with different kernel sizes
        self.deconv3x3 = nn.ConvTranspose2d(
            in_channels,
            out_channels // 3 + out_channels % 3,
            kernel_size=3,
            padding=0 + padding,
        )
        self.deconv5x5 = nn.ConvTranspose2d(
            in_channels,
            out_channels // 3,
            kernel_size=5,
            padding=1 + padding,
        )
        self.deconv7x7 = nn.ConvTranspose2d(
            in_channels,
            out_channels // 3,
            kernel_size=7,
            padding=2 + padding,
        )

        # Additional transposed convolutions for refinement
        self.deconv4x4 = nn.ConvTranspose2d(
            out_channels,
            out_channels // 2 + out_channels % 2,
            kernel_size=4,
            stride=1,
            padding=0,
        )

        self.deconv6x6 = nn.ConvTranspose2d(
            out_channels,
            out_channels // 2,
            kernel_size=6,
            stride=1,
            padding=1,
        )

        # BatchNorm and activation (applied after the sum)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply transposed convolutions
        out3 = self.deconv3x3(x)
        out5 = self.deconv5x5(x)
        out7 = self.deconv7x7(x)

        # Sum all the outputs from the transposed convolutions
        out_odd = torch.cat([out3, out5, out7], dim=1)

        # Refine with additional transposed convolutions
        out4 = self.deconv4x4(out_odd)
        out6 = self.deconv6x6(out_odd)

        combined = torch.cat([out4, out6], dim=1)

        # Apply BatchNorm and activation
        combined = self.batch_norm(combined)
        combined = self.activation(combined)

        return combined


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        # Normalize the input for stability
        self.norm = nn.LayerNorm(in_channels)

        # Multihead Attention module
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads, batch_first=True
        )

        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, in_channels, height, width]
        batch_size, channels, height, width = x.size()

        # Reshape to (batch_size, height*width, in_channels) for MultiheadAttention
        x_reshaped = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        # Apply Layer Normalization
        x_normalized = self.norm(x_reshaped)

        # Self-attention
        attention_out, _ = self.mha(x_normalized, x_normalized, x_normalized)

        # Reshape back to original dimensions
        attention_out = attention_out.permute(0, 2, 1).view(
            batch_size, channels, height, width
        )

        # Add the original input (residual connection) with learnable scaling
        out = self.gamma * attention_out + x
        return out


class ANN_Basic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation=None,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        diff = out_channels - in_channels
        step = diff // (num_hidden_layers + 1)

        layers = []

        for i in range(num_hidden_layers):
            layers.append(
                nn.Linear(in_channels + i * step, in_channels + (i + 1) * step)
            )
            layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Linear(in_channels + num_hidden_layers * step, out_channels))

        if activation is not None:
            layers.append(activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        layers = [
            BasicDownscaleBlock(in_channels, out_channels),
            BasicDownscaleBlock(out_channels, out_channels),
            SelfAttention(out_channels),
            ResidualBlock(out_channels),
            SelfAttention(out_channels),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels),
            SelfAttention(out_channels),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return checkpoint_sequential(self.model, 6, x, use_reentrant=False)

        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        layers = [
            SelfAttention(in_channels),
            ResidualBlock(in_channels),
            SelfAttention(in_channels),
            nn.BatchNorm2d(in_channels),
            ResidualBlock(in_channels),
            SelfAttention(in_channels),
            nn.BatchNorm2d(in_channels),
            BasicUpscaleBlock(in_channels, in_channels),
            BasicUpscaleBlock(in_channels, out_channels),
            nn.Sigmoid(),  # For output in range [0, 1] suitable for images
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return checkpoint_sequential(self.model, 6, x, use_reentrant=False)

        return self.model(x)
