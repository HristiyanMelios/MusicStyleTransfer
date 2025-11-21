import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractors(nn.Module):
    """
    Uses the pretrained VGG19 model for style transfer
    """

    def __init__(self, device):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Freeze weights
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


def gram_matrix(feature):
    B, C, H, W = feature.shape
    F = feature.view(B, C, H * W)
    G = torch.bmm(F, F.transpose(1, 2))
    return G / (C * H * W)
