import torch
import torch.nn as nn
from flearn.models.utils.VAE_block_utils import ResidualBlock, SelfAttention


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        num_classes: int,
        img_channels: int = 3,
        feature_map_size: int = 64,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size

        # Embedding for label conditioning
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(
                noise_dim + num_classes,
                feature_map_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.05, inplace=True),
            nn.ConvTranspose2d(
                feature_map_size,
                feature_map_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 4x4 -> 8x8
            nn.BatchNorm2d(feature_map_size // 2),
            nn.LeakyReLU(0.05, inplace=True),
            nn.ConvTranspose2d(
                feature_map_size // 2,
                feature_map_size // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 8x8 -> 16x16
            nn.BatchNorm2d(feature_map_size // 4),
            nn.LeakyReLU(0.05, inplace=True),
            ResidualBlock(feature_map_size // 4, feature_map_size // 4),
            nn.ConvTranspose2d(
                feature_map_size // 4,
                img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 16x16 -> 32x32,
            nn.Sigmoid(),
        )

    def forward(self, noise, labels):
        label_embed: torch.Tensor = self.label_embedding(labels)

        # (batch_size, num_classes, 1, 1)
        label_map = label_embed.unsqueeze(2).unsqueeze(3)

        # (batch_size, noise_dim + num_classes, 1, 1)
        x = torch.cat([noise, label_map], dim=1)

        return self.conv_blocks(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_channels: int = 3,
        output_feature_map_size: int = 64,
    ):
        super().__init__()
        self.feature_map_size = output_feature_map_size

        # Embedding for label conditioning
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(
                img_channels + num_classes,
                output_feature_map_size // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(output_feature_map_size // 4, output_feature_map_size // 4),
            nn.Conv2d(
                output_feature_map_size // 4,
                output_feature_map_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 16x16 -> 8x8
            nn.BatchNorm2d(output_feature_map_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                output_feature_map_size // 2,
                output_feature_map_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 8x8 -> 4x4
            nn.BatchNorm2d(output_feature_map_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                output_feature_map_size,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),  # output_feature_map_size x 4x4 -> 1 x 1x1
            nn.Sigmoid(),
        )

    def forward(self, img, labels):

        label_embed: torch.Tensor = self.label_embedding(labels)

        # (batch_size, num_classes, 1, 1)
        label_map = label_embed.unsqueeze(2).unsqueeze(3)

        # (batch_size, num_classes, 32, 32)
        label_map = label_map.expand(-1, -1, img.size(2), img.size(3))

        # (batch_size, img_channels + num_classes, 32, 32)
        x = torch.cat([img, label_map], dim=1)

        # Apply convolutional layers
        return self.conv_blocks(x)


class Generator_SA(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        num_classes: int,
        img_channels: int = 3,
        feature_map_size: int = 64,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size

        # Embedding for label conditioning
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(
                noise_dim + num_classes,
                feature_map_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),  # 1x1 -> 4x4
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.05, inplace=True),
            nn.ConvTranspose2d(
                feature_map_size,
                feature_map_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 4x4 -> 8x8
            nn.BatchNorm2d(feature_map_size // 2),
            nn.LeakyReLU(0.05, inplace=True),
            ResidualBlock(feature_map_size // 2),
            SelfAttention(feature_map_size // 2),
            nn.ConvTranspose2d(
                feature_map_size // 2,
                feature_map_size // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 8x8 -> 16x16
            nn.BatchNorm2d(feature_map_size // 4),
            nn.LeakyReLU(0.05, inplace=True),
            ResidualBlock(feature_map_size // 4),
            SelfAttention(feature_map_size // 4),
            nn.ConvTranspose2d(
                feature_map_size // 4,
                feature_map_size // 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 16x16 -> 32x32,
            ResidualBlock(feature_map_size // 8),
            nn.Conv2d(
                feature_map_size // 8,
                img_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, noise, labels):
        label_embed: torch.Tensor = self.label_embedding(labels)

        # (batch_size, num_classes, 1, 1)
        label_map = label_embed.unsqueeze(2).unsqueeze(3)

        # (batch_size, noise_dim + num_classes, 1, 1)
        x = torch.cat([noise, label_map], dim=1)

        return self.conv_blocks(x)


class Discriminator_SA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_channels: int = 3,
        output_feature_map_size: int = 64,
    ):
        super().__init__()
        self.feature_map_size = output_feature_map_size

        # Embedding for label conditioning
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(
                img_channels + num_classes,
                output_feature_map_size // 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ResidualBlock(output_feature_map_size // 8),
            nn.Conv2d(
                output_feature_map_size // 8,
                output_feature_map_size // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 32x32 -> 16x16
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(output_feature_map_size // 4),
            SelfAttention(output_feature_map_size // 4),
            nn.Conv2d(
                output_feature_map_size // 4,
                output_feature_map_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 16x16 -> 8x8
            nn.BatchNorm2d(output_feature_map_size // 2),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(output_feature_map_size // 2),
            SelfAttention(output_feature_map_size // 2),
            nn.Conv2d(
                output_feature_map_size // 2,
                output_feature_map_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 8x8 -> 4x4
            nn.BatchNorm2d(output_feature_map_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                output_feature_map_size,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),  # output_feature_map_size x 4x4 -> 1 x 1x1
            nn.Sigmoid(),
        )

    def forward(self, img, labels):

        label_embed: torch.Tensor = self.label_embedding(labels)

        # (batch_size, num_classes, 1, 1)
        label_map = label_embed.unsqueeze(2).unsqueeze(3)

        # (batch_size, num_classes, 32, 32)
        label_map = label_map.expand(-1, -1, img.size(2), img.size(3))

        # (batch_size, img_channels + num_classes, 32, 32)
        x = torch.cat([img, label_map], dim=1)

        # Apply convolutional layers
        return self.conv_blocks(x)
