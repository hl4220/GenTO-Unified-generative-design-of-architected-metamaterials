# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Provide shared generation utilities for binarization, sample collection, symmetry, and plotting.

import matplotlib.pyplot as plt
import numpy as np
import torch

###############################################################################
# Common sample processing
###############################################################################

def Matext_fvol(img_batch, vf=0.5):
    """Binarize each sample by matching the requested volume fraction."""
    if img_batch.ndim == 3:
        img_batch = img_batch.unsqueeze(1)
    B = img_batch.shape[0]
    img_flat = img_batch.reshape(B, -1)
    thresholds = torch.quantile(img_flat, 1.0-vf, dim=1, keepdim=True).reshape(B, 1, 1, 1)
    return (img_batch > thresholds).float()

def generation_ac(pipeline, useddim=True, batch=8, explore_scale=1.0):
    """Generate a batch of continuous samples from the diffusion pipeline."""
    flagddim = 2 if useddim else 1
    with torch.inference_mode():
        samples = pipeline(
            B=batch,
            n_samples=1,
            switch=1,
            flag=flagddim,
            type="x0",
            explore_scale=explore_scale,
        )
    return samples

###############################################################################
# Single-objective sample collection
###############################################################################

def collect_valid_samples(
    pipeline,
    ntake,
    device,
    vf,
    pass_fn,
    score_fn,
    batch=32,
    iteration=0,
    useddim=False,
    explore_scale=1.0,
    sample_transform_fn=None,
    store_transformed_samples=True,
):
    """Collect enough valid samples for a single-objective optimisation step."""
    samples_left = []
    score_left = []
    target_count = ntake * 2
    total_collected = 0

    while total_collected < target_count:
        # Generate, binarize, filter, then score only the valid samples.
        samples = generation_ac(
            pipeline=pipeline,
            useddim=useddim,
            batch=batch,
            explore_scale=explore_scale,
        )
        samples_for_eval = sample_transform_fn(samples) if sample_transform_fn is not None else samples
        samples_binary = Matext_fvol(samples_for_eval, vf=vf)
        pass_list = pass_fn(samples_binary, device=device)
        samples_to_store = samples_for_eval if store_transformed_samples else samples
        valid_samples = samples_to_store[pass_list]
        valid_samples_binary = samples_binary[pass_list]

        if valid_samples.shape[0] > 0:
            valid_scores = score_fn(valid_samples_binary, device=device)
            samples_left.append(valid_samples)
            score_left.append(valid_scores)
            total_collected += valid_samples.shape[0]

        print(f"Iter:{iteration+1}, Collected: {total_collected}/{target_count}")

    return torch.cat(samples_left, dim=0), torch.cat(score_left, dim=0)

###############################################################################
# Multi-objective sample collection and Pareto selection
###############################################################################

def collect_valid_samples_mo(
    pipeline,
    ntake,
    device,
    vf,
    pass_fn,
    score_fn,
    batch=32,
    iteration=0,
    useddim=False,
    explore_scale=1.0,
    sample_transform_fn=None,
    store_transformed_samples=True,
):
    """Collect valid samples and stack multiple objective scores column-wise."""
    samples_left = []
    score_left = []
    target_count = ntake * 2
    total_collected = 0

    while total_collected < target_count:
        samples = generation_ac(
            pipeline=pipeline,
            useddim=useddim,
            batch=batch,
            explore_scale=explore_scale,
        )
        samples_for_eval = sample_transform_fn(samples) if sample_transform_fn is not None else samples
        samples_binary = Matext_fvol(samples_for_eval, vf=vf)
        pass_list = pass_fn(samples_binary, device=device)
        samples_to_store = samples_for_eval if store_transformed_samples else samples
        valid_samples = samples_to_store[pass_list]
        valid_samples_binary = samples_binary[pass_list]

        if valid_samples.shape[0] > 0:
            score_columns = score_fn(valid_samples_binary, device=device)
            valid_scores = torch.stack(score_columns, dim=1)
            samples_left.append(valid_samples.detach().cpu())
            score_left.append(valid_scores.detach().cpu())
            total_collected += valid_samples.shape[0]

        print(f"Iter:{iteration+1}, Collected valid: {total_collected}/{target_count}")

    return torch.cat(samples_left, dim=0), torch.cat(score_left, dim=0)


def pareto_mask(scores):
    """Mark points that are not dominated by any other point."""
    count = scores.shape[0]
    mask = torch.ones(count, dtype=torch.bool)

    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            if (
                scores[j, 0] >= scores[i, 0]
                and scores[j, 1] >= scores[i, 1]
                and (scores[j, 0] > scores[i, 0] or scores[j, 1] > scores[i, 1])
            ):
                mask[i] = False
                break

    return mask


def pareto_layers(samples, scores, ntake):
    """Select samples layer by layer from the Pareto frontier until ntake is filled."""
    remaining_idx = torch.arange(samples.shape[0])
    selected_idx = []

    while len(remaining_idx) > 0:
        total_selected = sum(len(idx) for idx in selected_idx)
        if total_selected >= ntake:
            break

        layer_mask = pareto_mask(scores[remaining_idx])
        layer_idx = remaining_idx[layer_mask]

        if total_selected + len(layer_idx) <= ntake:
            selected_idx.append(layer_idx)
        else:
            n_needed = ntake - total_selected
            perm = torch.randperm(len(layer_idx))[:n_needed]
            selected_idx.append(layer_idx[perm])

        remaining_idx = remaining_idx[~layer_mask]

    selected_idx = torch.cat(selected_idx, dim=0)
    return samples[selected_idx], scores[selected_idx]

###############################################################################
# Geometry helpers
###############################################################################

def make_4fold_symmetric_tensor(img_tensor):
    """Mirror the top-left quadrant into a fourfold-symmetric full image."""
    quarter = img_tensor[..., :128, :128]
    
    top = torch.cat([quarter, torch.flip(quarter, dims=[-1])], dim=-1)
    full = torch.cat([top, torch.flip(top, dims=[-2])], dim=-2)
    return full

def make_8fold_symmetric_tensor(img_tensor):
    """Mirror the upper triangle of the top-left quadrant into an eightfold-symmetric full image."""
    quarter = img_tensor[..., :128, :128]
    quarter_upper = torch.triu(quarter)
    quarter_symmetric = quarter_upper + torch.triu(quarter, diagonal=1).transpose(-1, -2)

    top = torch.cat([quarter_symmetric, torch.flip(quarter_symmetric, dims=[-1])], dim=-1)
    full = torch.cat([top, torch.flip(top, dims=[-2])], dim=-2)
    return full

###############################################################################
# Visualisation helpers
###############################################################################

def _score_frequency_response(freqs, T_dB, regions, threshold=-10.0):
    score = 0.0
    for lower, upper, weight in regions["gap"]:
        mask = (freqs >= lower) & (freqs <= upper)
        if np.any(mask):
            score += np.mean(T_dB[mask] < threshold) * weight
    for lower, upper, weight in regions["pass"]:
        mask = (freqs >= lower) & (freqs <= upper)
        if np.any(mask):
            score += np.mean(T_dB[mask] > threshold) * weight
    return score

def plot_samples(img_batch, save_name='samples.png', vf=0.5):
    """Save continuous samples together with their volume-fraction binarization."""
    batch = img_batch.shape[0]
    binary_batch = Matext_fvol(img_batch, vf=vf)
    
    samples_np = img_batch.cpu().numpy()
    binary_np = binary_batch.cpu().numpy()
    
    if samples_np.ndim == 4:
        samples_np = samples_np[:, 0]
        binary_np = binary_np[:, 0]
    
    ncols = min(4, batch)
    nrows = (batch + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(3 * ncols, 3 * nrows * 2))
    axes = np.array(axes).reshape(nrows * 2, ncols)
    
    for i in range(batch):
        row = (i // ncols) * 2
        col = i % ncols
        
        axes[row, col].imshow(samples_np[i], cmap='viridis')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Sample-{i}", fontsize=8)
        
        axes[row + 1, col].imshow(1-binary_np[i], cmap='gray')
        axes[row + 1, col].axis('off')
    
    for i in range(batch, nrows * ncols):
        row = (i // ncols) * 2
        col = i % ncols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()


def plot_frequency_responses(samples, solver, save_name, iteration, regions, threshold=-10.0, n_examples=4, vf=0.5):
    """Save transmission-curve previews using the same volume-fraction binarization as scoring."""
    count = min(n_examples, samples.shape[0])
    if count == 0:
        return

    binary_samples = Matext_fvol(samples[:count], vf=vf)
    region_bounds = regions["gap"] + regions["pass"]
    freq_min = min(lower for lower, _, _ in region_bounds) / 1e3
    freq_max = max(upper for _, upper, _ in region_bounds) / 1e3

    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4), sharey=True)
    if count == 1:
        axes = [axes]

    for index in range(count):
        sample = binary_samples[index, 0] if binary_samples.ndim == 4 else binary_samples[index]
        ax = axes[index]
        try:
            freqs, T_dB = solver.solve(sample.cpu())
        except Exception:
            ax.set_xlim(freq_min, freq_max)
            ax.set_ylim(-50.0, 20.0)
            ax.set_xlabel("Frequency (kHz)")
            ax.grid(True, alpha=0.35)
            ax.set_title(f"S{index+1}  failed")
            if index == 0:
                ax.set_ylabel("T (dB)")
            continue

        for lower, upper, _ in regions["gap"]:
            ax.axvspan(lower / 1e3, upper / 1e3, alpha=0.10, color="red")
        for lower, upper, _ in regions["pass"]:
            ax.axvspan(lower / 1e3, upper / 1e3, alpha=0.15, color="green")

        ax.plot(freqs / 1e3, T_dB, lw=1.2, color="steelblue")
        ax.axhline(threshold, color="k", ls="--", lw=0.9)
        score = _score_frequency_response(freqs, T_dB, regions, threshold=threshold)
        ax.set_xlim(freq_min, freq_max)
        ax.set_ylim(-50.0, 20.0)
        ax.set_xlabel("Frequency (kHz)")
        ax.grid(True, alpha=0.35)
        ax.set_title(f"S{index+1}  score={score:.2f}")
        if index == 0:
            ax.set_ylabel("T (dB)")

    plt.suptitle(f"Transmission - Iter {iteration}")
    plt.tight_layout()
    plt.savefig(save_name, dpi=100)
    plt.close()


def save_pareto_scatter(scores_all, elite_scores, save_path, iteration):
    """Save a scatter plot of all candidates and the selected Pareto elite set."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        scores_all[:, 0].cpu().numpy(),
        scores_all[:, 1].cpu().numpy(),
        alpha=0.3,
        s=10,
        color="steelblue",
        label="All candidates",
    )
    ax.scatter(
        elite_scores[:, 0].cpu().numpy(),
        elite_scores[:, 1].cpu().numpy(),
        alpha=0.8,
        s=20,
        color="crimson",
        label=f"Elite ({elite_scores.shape[0]})",
    )
    ax.set_xlabel("Fractal Dimension")
    ax.set_ylabel("Min Feature Size")
    ax.set_xlim(1.0, 1.8)
    ax.set_ylim(0, 128)
    ax.set_title(f"Score Distribution - Iteration {iteration}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


__all__ = [
    "Matext_fvol",
    "collect_valid_samples",
    "collect_valid_samples_mo",
    "generation_ac",
    "make_4fold_symmetric_tensor",
    "make_8fold_symmetric_tensor",
    "pareto_layers",
    "pareto_mask",
    "plot_frequency_responses",
    "plot_samples",
    "save_pareto_scatter",
]
