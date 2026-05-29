# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Train the unconditional diffusion model on packed binary-image datasets.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from data_aggregation import discover_packed_files, unpack_binary_image
from model.model_diffusion import DiffusionUNet, diffusion_pipeline

###############################################################################
# Configuration
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Binary_Image_Packed"
# Select which packed dataset group to train.
# Available groups are "periodic", "four_fold", and "eight_fold".
DATA_GROUP = "periodic"


###############################################################################
# Geometry preprocessing
###############################################################################

def DisField(p):
    din = distance_transform_edt(p)
    dout = distance_transform_edt(~p)
    return din - dout


def normpn(phi):
    span = phi.max() - phi.min()
    if span == 0:
        return np.zeros_like(phi, dtype=np.float32)
    return -1 + 2 * (phi - phi.min()) / span


def geo_surf(img):
    p = img.astype(bool)
    phi = DisField(p)
    return -normpn(phi)


###############################################################################
# Dataset
###############################################################################

class PackedSurfDataset(Dataset):
    """Memory-mapped packed binary images decoded on demand for training."""

    def __init__(self, data_root, group=DATA_GROUP):
        self.records = discover_packed_files(root_dir=data_root, groups=group)
        self.lengths = np.array([record["length"] for record in self.records], dtype=np.int64)
        self.cumulative_lengths = np.cumsum(self.lengths)
        self._packed_arrays = None

        total = int(self.cumulative_lengths[-1])
        print(f"Loaded index for group '{group}': {len(self.records)} files, {total} images.")

    def __len__(self):
        return int(self.cumulative_lengths[-1])

    def _arrays(self):
        if self._packed_arrays is None:
            self._packed_arrays = [np.load(record["path"], mmap_mode="r") for record in self.records]
        return self._packed_arrays

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_index = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        prev_total = 0 if file_index == 0 else int(self.cumulative_lengths[file_index - 1])
        local_index = idx - prev_total

        packed_img = self._arrays()[file_index][local_index]
        img = unpack_binary_image(packed_img)
        surf_img = geo_surf(img).astype(np.float32)
        return torch.from_numpy(surf_img).unsqueeze(0)


###############################################################################
# Training
###############################################################################

def train():
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    batch_size = 32 * gpu_count if gpu_count > 0 else 32
    lr = 1e-4
    epochs = 100
    save_dir = PROJECT_ROOT / "checkpoints_iso_original"
    save_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = PackedSurfDataset(DATA_ROOT, group=DATA_GROUP)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    unet = DiffusionUNet(time_dim=256, condition_dim=256).to(device)
    if gpu_count > 1:
        print(f"Using DataParallel on {gpu_count} GPUs!")
        unet = nn.DataParallel(unet)

    pipeline = diffusion_pipeline(unet, device).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    print("Start Training...")
    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_images in pbar:
            batch_images = batch_images.to(device)

            optimizer.zero_grad()
            loss = pipeline(image=batch_images, flag=0, type="x0")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            model_to_save = unet.module if isinstance(unet, nn.DataParallel) else unet
            torch.save(model_to_save.state_dict(), save_dir / f"model_epoch_{epoch + 1}.pth")

            pipeline.diffusion_model.eval()
            with torch.no_grad():
                samples = pipeline(B=4, n_samples=1, flag=1, type="x0")
                samples_np = samples.cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            for i in range(4):
                axes[i].imshow(samples_np[i, 0], cmap="viridis")
                axes[i].axis("off")
            plt.savefig(save_dir / f"sample_epoch_{epoch + 1}.png")
            plt.close()

            pipeline.diffusion_model.train()


if __name__ == "__main__":
    train()
