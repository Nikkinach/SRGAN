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
from model.generator import GeneratorRRDB
from model.loss import CharbonnierLoss
from model.metrics import psnr, ssim

def pretrain_srgan_improved():
    dataset_path = "/content/drive/MyDrive"
    save_model_path = os.path.join(dataset_path, "checkpoints_pretrain_improved")
    os.makedirs(save_model_path, exist_ok=True)

    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    upscale_factor = 4

    # Dataset and Dataloader
    train_set = DIV2KDataset(dataset_path, mode='train', upscale_factor=upscale_factor)
    valid_set = DIV2KDataset(dataset_path, mode='valid', upscale_factor=upscale_factor)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = GeneratorRRDB().to(device)

    # Loss and optimizer
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(generator.parameters(), lr=learning_rate)
    scaler = GradScaler("cuda")

    metrics = []

    for epoch in range(num_epochs):
        generator.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")

        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                sr_imgs = generator(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        torch.save(generator.state_dict(), os.path.join(save_model_path, f"gen_pretrain_epoch_{epoch+1}.pth"))

        # Validation Phase
        generator.eval()
        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                with autocast("cuda"):
                    sr_imgs = generator(lr_imgs)
                avg_psnr += psnr(sr_imgs, hr_imgs).item()
                avg_ssim += ssim(sr_imgs, hr_imgs).item()

        avg_psnr /= len(valid_loader)
        avg_ssim /= len(valid_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        metrics.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim
        })
        pd.DataFrame(metrics).to_csv(os.path.join(dataset_path, "metrics_pretrain_improved.csv"), index=False)

if __name__ == "__main__":
    pretrain_srgan_improved()