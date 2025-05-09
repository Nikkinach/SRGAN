import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import autocast, GradScaler

from data_loader import DIV2KDataset
from model.discriminator import Discriminator
from model.generator import GeneratorRRDB
from model.loss import CharbonnierLoss
from model.metrics import psnr, ssim
from model.perceptual import VGGFeatureExtractor


def train_srgan_improved():
    dataset_path = "/content/drive/MyDrive"
    pretrain_path = os.path.join(dataset_path, "checkpoints_pretrain_improved", "gen_pretrain_epoch_20.pth")
    save_model_path = os.path.join(dataset_path, "checkpoints_srgan_improved")
    os.makedirs(save_model_path, exist_ok=True)

    # Hyperparameters
    batch_size = 16
    num_epochs = 40
    lr_gen = 1e-4
    lr_disc = 1e-4
    lambda_content = 1.0
    lambda_adv = 1e-2  # Hinge loss allows stronger weight

    # Data
    train_set = DIV2KDataset(dataset_path, mode='train')
    valid_set = DIV2KDataset(dataset_path, mode='valid')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    # Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = GeneratorRRDB().to(device)
    discriminator = Discriminator().to(device)
    vgg = VGGFeatureExtractor().to(device)
    generator.load_state_dict(torch.load(pretrain_path, map_location=device))

    # Loss functions and optimizers
    pixel_loss = CharbonnierLoss()
    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=lr_gen)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=lr_disc)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs)

    scaler = GradScaler("cuda")
    metrics = []

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        progress_bar = tqdm(train_loader, desc=f"SRGAN Epoch {epoch+1}/{num_epochs}")

        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # ----------- Generator Update -----------
            optimizer_G.zero_grad()
            with autocast("cuda"):
                gen_hr = generator(lr_imgs)
                pred_fake = discriminator(gen_hr)
                pred_real = discriminator(hr_imgs).detach()

                # Hinge RaGAN adversarial loss
                g_adv_loss = -torch.mean(pred_real - pred_fake)

                # Content loss
                real_feat = vgg(hr_imgs).detach()
                fake_feat = vgg(gen_hr)
                content_loss = pixel_loss(fake_feat, real_feat)

                g_loss = lambda_content * content_loss + lambda_adv * g_adv_loss

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)

            # ----------- Discriminator Update -----------
            optimizer_D.zero_grad()
            with autocast("cuda"):
                pred_real = discriminator(hr_imgs)
                pred_fake = discriminator(gen_hr.detach())

                # RaGAN Hinge Loss
                d_loss_real = torch.mean(torch.relu(1.0 - (pred_real - pred_fake)))
                d_loss_fake = torch.mean(torch.relu(1.0 + (pred_fake - pred_real)))
                d_loss = (d_loss_real + d_loss_fake) / 2

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()

            progress_bar.set_postfix({"g_loss": g_loss.item(), "d_loss": d_loss.item()})

        scheduler_G.step()
        scheduler_D.step()
        torch.save(generator.state_dict(), os.path.join(save_model_path, f"gen_epoch_{epoch+1}.pth"))

        # ----------- Validation -----------
        generator.eval()
        avg_psnr, avg_ssim = 0, 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                with autocast("cuda"):
                    gen_hr = generator(lr_imgs)
                avg_psnr += psnr(gen_hr, hr_imgs).item()
                avg_ssim += ssim(gen_hr, hr_imgs).item()

        avg_psnr /= len(valid_loader)
        avg_ssim /= len(valid_loader)
        print(f"Validation PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        metrics.append({
            "epoch": epoch + 1,
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "psnr": avg_psnr,
            "ssim": avg_ssim
        })
        pd.DataFrame(metrics).to_csv(os.path.join(dataset_path, "metrics_srgan_improved.csv"), index=False)

if __name__ == "__main__":
    train_srgan_improved()