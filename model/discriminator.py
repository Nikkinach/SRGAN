import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, clip_value=0.01):
        super().__init__()
        self.clip_value = clip_value  # Optional clipping

        def block(in_feat, out_feat, stride):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_feat, out_feat, 3, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            block(256, 512, 1),
            block(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out
