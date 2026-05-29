# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Define the unconditional diffusion U-Net and sampling/training pipeline used by generation tasks.

"""Diffusion U-Net and wrapper used in this project.

The current GD_AC workflow is effectively unconditional. Older task scripts may
still pass `features` or `switch`, so the public pipeline API keeps those
arguments, but they are not used by the active forward path.

The attention branch has been removed because it was not used in the current
workflow. Older checkpoints should therefore be loaded once with
`strict=False`, then re-saved as a new state_dict without attention weights.
"""

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block conditioned only on diffusion time in current use."""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = x + self.time_mlp(t)[:, :, None, None]
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x + residual


class DownBlock(nn.Module):
    """Residual block followed by 2x downsampling."""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, t):
        return self.down(self.res(x, t))


class UpBlock(nn.Module):
    """2x upsampling, skip concatenation, then residual refinement."""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.res = ResidualBlock(in_channels + in_channels // 2, out_channels, time_dim)

    def forward(self, x, skip, t):
        return self.res(torch.cat([self.up(x), skip], dim=1), t)


class DiffusionUNet(nn.Module):
    """U-Net backbone used by the diffusion model."""

    def __init__(self, time_dim=256, condition_dim=256):
        super().__init__()
        del condition_dim  # Kept only so existing constructor calls still work.

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv_in = nn.Conv2d(1, 64, kernel_size=3, padding=1, padding_mode="circular")
        self.down1 = DownBlock(64, 128, time_dim)
        self.down2 = DownBlock(128, 256, time_dim)
        self.down3 = DownBlock(256, 512, time_dim)
        self.bottleneck1 = ResidualBlock(512, 512, time_dim)
        self.bottleneck2 = ResidualBlock(512, 512, time_dim)
        self.up1 = UpBlock(512, 256, time_dim)
        self.up2 = UpBlock(256, 128, time_dim)
        self.up3 = UpBlock(128, 64, time_dim)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, x, t):
        t = self.time_mlp(t.unsqueeze(-1).float())
        x = self.conv_in(x)

        skip1 = x
        x = self.down1(x, t)
        skip2 = x
        x = self.down2(x, t)
        skip3 = x
        x = self.down3(x, t)

        x = self.bottleneck1(x, t)
        x = self.bottleneck2(x, t)
        x = self.up1(x, skip3, t)
        x = self.up2(x, skip2, t)
        x = self.up3(x, skip1, t)
        return self.conv_out(x)


class DiffusionPipeline(nn.Module):
    """Training and sampling wrapper around the U-Net."""

    def __init__(self, model, device):
        super().__init__()
        self.diffusion_model = model
        self.device = device  # Kept for compatibility; runtime tensors use the registered buffers' device.
        self.num_steps = 100
        self.beta = np.linspace(0.0001 ** 0.5, 0.5 ** 0.5, self.num_steps) ** 2
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer("beta_values", torch.tensor(self.beta).float())
        self.register_buffer("alpha_hat_values", torch.tensor(self.alpha_hat).float())
        self.register_buffer("alpha_values", torch.tensor(self.alpha).float())
        self.register_buffer(
            "alpha_torch",
            self.alpha_values.unsqueeze(1).unsqueeze(1),
        )

    def train_process(self, image, type="eps"):
        device = self.alpha_values.device
        image = image.to(device)
        batch_size = image.shape[0]
        t = torch.randint(0, self.num_steps, (batch_size,), device=device)
        current_alpha = self.alpha_values[t][:, None, None, None]

        noise = torch.randn_like(image)
        noisy_image = (
            current_alpha.sqrt() * image
            + (1.0 - current_alpha).sqrt() * noise
        )
        predicted = self.diffusion_model(noisy_image, t)

        if type == "eps":
            residual = noise - predicted
        elif type == "x0":
            residual = image - predicted
        else:
            raise ValueError(f"Unsupported training target: {type}")

        return torch.sum(residual ** 2) / 256 / 256

    def _reverse_step(self, sample_noise, prediction, step, type, explore_scale=1.0):
        alpha = self.alpha_values[step]
        alpha_hat = self.alpha_hat_values[step]

        if type == "eps":
            noise_predicted = prediction
        elif type == "x0":
            noise_predicted = -((prediction * alpha.sqrt() - sample_noise) / (1 - alpha).sqrt())
        else:
            raise ValueError(f"Unsupported sampling target: {type}")

        coeff1 = alpha_hat.rsqrt()
        coeff2 = (1 - alpha_hat) / (1 - alpha).sqrt()
        noise = torch.randn_like(noise_predicted)
        sample_noise = coeff1 * (sample_noise - coeff2 * noise_predicted)
        if step > 0:
            alpha_prev = self.alpha_values[step - 1]
            beta = self.beta_values[step]
            sigma = (((1.0 - alpha_prev) / (1.0 - alpha)) * beta).sqrt()
            sample_noise += explore_scale * sigma * noise
        return sample_noise

    def generation(self, B, n_samples, H=256, W=256, type="eps", explore_scale=1.0):
        device = self.alpha_values.device
        generated_samples = torch.empty(B, n_samples, H, W, device=device)
        for i in range(n_samples):
            sample_noise = torch.randn(B, 1, H, W, device=device)
            for step in range(self.num_steps - 1, -1, -1):
                t = torch.full((B,), step, device=device, dtype=torch.long)
                prediction = self.diffusion_model(sample_noise, t)
                sample_noise = self._reverse_step(sample_noise, prediction, step, type, explore_scale=explore_scale)
            generated_samples[:, i : i + 1, :, :] = sample_noise.detach()
        return generated_samples

    def ddim_generation(self, B, n_samples, H=256, W=256, n_steps=10):
        device = self.alpha_values.device
        generated_samples = torch.empty(B, n_samples, H, W, device=device)
        ddim_timesteps = np.linspace(self.num_steps - 1, 0, n_steps).astype(int)

        for i in range(n_samples):
            sample_noise = torch.randn(B, 1, H, W, device=device)
            for idx, step in enumerate(ddim_timesteps):
                t_tensor = torch.full((B,), int(step), device=device, dtype=torch.long)
                pred_x0 = self.diffusion_model(sample_noise, t_tensor)

                alpha_t = self.alpha_values[step]
                if idx + 1 < len(ddim_timesteps):
                    alpha_t_next = self.alpha_values[ddim_timesteps[idx + 1]]
                else:
                    alpha_t_next = self.alpha_values.new_tensor(1.0)
                pred_epsilon = (sample_noise - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()

                if idx == len(ddim_timesteps) - 1:
                    sample_noise = pred_x0.detach()
                else:
                    sample_noise = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * pred_epsilon

            generated_samples[:, i : i + 1, :, :] = sample_noise.detach()
        return generated_samples

    def forward(
        self,
        features=None,
        switch=1,
        image=None,
        n_samples=None,
        flag=None,
        B=32,
        type="eps",
        explore_scale=1.0,
    ):
        del features, switch

        if flag == 0:
            return self.train_process(image=image, type=type)
        if flag == 1:
            return self.generation(B=B, n_samples=n_samples, type=type, explore_scale=explore_scale)
        if flag == 2:
            return self.ddim_generation(B=B, n_samples=n_samples, n_steps=10)
        if flag == 3:
            return self.alpha_torch
        raise ValueError(f"Unsupported flag: {flag}")


diffusion_pipeline = DiffusionPipeline
ConditionalDiffusionUNet = DiffusionUNet
ConditionalResidualBlock = ResidualBlock


__all__ = [
    "DiffusionUNet",
    "ConditionalDiffusionUNet",
    "ResidualBlock",
    "ConditionalResidualBlock",
    "DiffusionPipeline",
    "diffusion_pipeline",
]
