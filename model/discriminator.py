import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_feat, out_feat, stride):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 3, stride, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            # Spectral Norm ONLY on the first layer
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(64, 64, 2),
            conv_block(64, 128, 1),
            conv_block(128, 128, 2),
            conv_block(128, 192, 1),
            conv_block(192, 192, 2),
            conv_block(192, 256, 1),
            conv_block(256, 256, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)

