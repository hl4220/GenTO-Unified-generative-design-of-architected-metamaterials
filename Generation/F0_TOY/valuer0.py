# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Score F0 toy samples by direct pixel matching against the target image.

import torch
from matplotlib.image import imread
from pathlib import Path


TARGET_CACHE = {}
TARGET_PATH = Path(__file__).with_name("target.png")


def _target(size, device):
    key = (size, str(device))
    if key not in TARGET_CACHE:
        img = imread(TARGET_PATH)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=-1)
        if img.shape != (size, size):
            raise ValueError(f"target.png has shape {img.shape}, expected {(size, size)}")
        target = torch.tensor(img < 0.5, dtype=torch.bool, device=device)
        TARGET_CACHE[key] = target[None, :, :]
    return TARGET_CACHE[key]


def valuer0_pass(binary_samples, device):
    return torch.ones(binary_samples.shape[0], dtype=torch.bool, device=device)


def valuer0_score(binary_samples, device):
    samples = binary_samples[:, 0].to(device).bool()
    target = _target(samples.shape[-1], device)
    return (samples == target).float().mean(dim=(1, 2))


__all__ = ["valuer0_pass", "valuer0_score"]
