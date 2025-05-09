import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(growth_channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(growth_channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(growth_channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(growth_channels, channels, 3, 1, 1)
        )
        self.se = SELayer(channels)

    def forward(self, x):
        return x + 0.2 * self.se(self.body(x))

class RRDB(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels)
        )

    def forward(self, x):
        return x + 0.2 * self.block(x)

class GeneratorRRDB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feat=64, num_blocks=23):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        res = self.trunk_conv(self.body(fea))
        fea = fea + res
        out = self.upsample(fea)
        out = self.conv_last(out)
        return out + F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)