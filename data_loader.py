import os
import random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, mode='train', crop_size=96, upscale_factor=4):
        self.hr_dir = os.path.join(root_dir, f'DIV2K_{mode}_HR')
        self.file_names = sorted(os.listdir(self.hr_dir))
        self.crop_size = crop_size
        self.upscale = upscale_factor
        self.mode = mode

        # Normalize to [-1, 1]
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor()

    def degrade(self, hr_img):
        """ Apply degradation: blur → downscale → noise """
        # Random Gaussian Blur
        if random.random() < 0.5:
            hr_img = hr_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.5)))

        # Resize down to LR
        lr_img = hr_img.resize(
            (hr_img.width // self.upscale, hr_img.height // self.upscale),
            resample=Image.BICUBIC
        )

        # Optional: Add synthetic noise (disabled for now)
        return lr_img

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.file_names[idx])
        hr = Image.open(hr_path).convert('RGB')

        # Random crop for training
        if self.mode == 'train':
            i = random.randint(0, hr.height - self.crop_size * self.upscale)
            j = random.randint(0, hr.width - self.crop_size * self.upscale)
            hr = TF.crop(hr, i, j, self.crop_size * self.upscale, self.crop_size * self.upscale)

        # Degrade HR to generate LR
        lr = self.degrade(hr)

        # Convert to tensor and normalize
        hr_tensor = self.normalize(self.to_tensor(hr))
        lr_tensor = self.normalize(self.to_tensor(lr))

        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.file_names)