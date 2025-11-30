import torch
import torch.nn as nn
from torchvision import models

#----------------------------------- CNN Architecture -----------------------------------#
class FeatureExtractors(nn.Module):
    """
    Uses the pretrained VGG19 model for style transfer
    """

    def __init__(self, device):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg.to(device).eval()

        self.target_layers = {
            "relu1_1": 1,
            "relu2_1": 6,
            "relu3_1": 11,
            "relu4_1": 20,
            "relu4_2": 22,  # Used for content loss
            "relu5_1": 29,
        }

    def forward(self, x):
        # The spectrograms are only single channel so just repeat to get 3 channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = {}
        current = x

        for i, layer in enumerate(self.vgg):
            current = layer(current)

            names = [name for name, idx in self.target_layers.items() if idx == i]
            if names:
                features[names[0]] = current

        return features


#----------------------------------- UNet Blocks -----------------------------------#
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = torch.nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ----------------------------------- CycleGAN Blocks -----------------------------------#
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        self.inConv = ConvBlock(in_channels, base_channels)
        # Encoder
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        # Bottleneck convolution
        self.out = nn.Sequential(
            OutConv(base_channels, out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.inConv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        out = self.out(u4)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        layers.append(nn.InstanceNorm2d(base_channels * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        layers.append(nn.InstanceNorm2d(base_channels * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(
            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False
            )
        )
        layers.append(nn.InstanceNorm2d(base_channels * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(
            nn.Conv2d(
                base_channels * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def gram_matrix(feature):
    B, C, H, W = feature.shape
    F = feature.view(B, C, H * W)
    G = torch.bmm(F, F.transpose(1, 2))
    return G / (C * H * W)  # normalize before returning
