import torch
import torch.nn as nn

def ssim(img1, img2):
    """Simplified SSIM implementation (grayscale, per image)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = nn.functional.avg_pool2d(img1, 3, 1, padding=1)
    mu2 = nn.functional.avg_pool2d(img2, 3, 1, padding=1)

    sigma1_sq = nn.functional.avg_pool2d(img1 * img1, 3, 1, padding=1) - mu1 ** 2
    sigma2_sq = nn.functional.avg_pool2d(img2 * img2, 3, 1, padding=1) - mu2 ** 2
    sigma12 = nn.functional.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def psnr(pred, target, max_val=1.0):
    mse = nn.functional.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))