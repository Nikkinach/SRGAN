import torch
import torch.nn as nn
from torchvision import models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=35, use_input_norm=True):  # conv5_4 = layer 35
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:feature_layer + 1])
        for param in self.features.parameters():
            param.requires_grad = False

        self.use_input_norm = use_input_norm
        if use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)