# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Run the F0 toy target-matching optimisation with the pretrained periodic diffusion model.

import json
from collections import OrderedDict
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

###############################################################################
# Paths and imports
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
TASK_ROOT = Path(__file__).resolve().parent

if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from model.model_diffusion import DiffusionUNet, diffusion_pipeline
from model.utilities import Matext_fvol, collect_valid_samples, plot_samples
from valuer0 import valuer0_pass, valuer0_score

###############################################################################
# Model setup
###############################################################################

CKPT_PATH = MODELS_ROOT / "model_ckpt_pretrained" / "periodic_pretrained.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionUNet(time_dim=256, condition_dim=256).to(device)

state_dict = torch.load(CKPT_PATH, map_location=device)
new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True)
model.eval()
Diffusion = diffusion_pipeline(model, device).to(device)

###############################################################################
# Experiment settings and output folders
###############################################################################

TEST_NUM = 1
VOLUME_FRACTION = 0.75
N_ITERATIONS = 100
NTAKE = 8
GEN_BATCH = 16
TRAIN_BATCH = 16
PREVIEW_BATCH = 4
EPOCHS = 1
LEARNING_RATE = 5e-5
EXPLORE_SCALE = 1.0
SAVE_SAMPLE_PREVIEWS = True
SAVE_CKPTS = False

ckpt_dir = TASK_ROOT / f"model_ckpt_tuned_TOY_{TEST_NUM}"
output_dir = TASK_ROOT / f"outputs_TOY_{TEST_NUM}"
sample_previews_dir = output_dir / "sample_previews"
iteration_samples_dir = output_dir / "iteration_samples"
summary_json_path = output_dir / "run_summary.json"
training_stats_path = output_dir / "training_stats.txt"
best_design_png_path = output_dir / "best_design.png"
best_design_pt_path = output_dir / "best_design.pt"
output_dir.mkdir(exist_ok=True)
iteration_samples_dir.mkdir(exist_ok=True)
if SAVE_CKPTS:
    ckpt_dir.mkdir(exist_ok=True)
if SAVE_SAMPLE_PREVIEWS:
    sample_previews_dir.mkdir(exist_ok=True)

with open(training_stats_path, "w") as f:
    f.write("Iteration\tscore_ave\tscore_ave_preview\n")

run_summary = {
    "config": {
        "project_root": str(PROJECT_ROOT),
        "task_root": str(TASK_ROOT),
        "checkpoint_path": str(CKPT_PATH),
        "checkpoint_dir": str(ckpt_dir),
        "output_dir": str(output_dir),
        "sample_previews_dir": str(sample_previews_dir),
        "iteration_samples_dir": str(iteration_samples_dir),
        "best_design_png": str(best_design_png_path),
        "best_design_pt": str(best_design_pt_path),
        "device": str(device),
        "test_num": TEST_NUM,
        "target": "letter_G_void",
        "volume_fraction": VOLUME_FRACTION,
        "n_iterations": N_ITERATIONS,
        "ntake": NTAKE,
        "gen_batch": GEN_BATCH,
        "train_batch": TRAIN_BATCH,
        "preview_batch": PREVIEW_BATCH,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "explore_scale": EXPLORE_SCALE,
        "save_sample_previews": SAVE_SAMPLE_PREVIEWS,
        "save_ckpts": SAVE_CKPTS,
    },
    "metrics": [],
}

with open(summary_json_path, "w") as f:
    json.dump(run_summary, f, indent=2)

###############################################################################
# Optimisation Loop
###############################################################################

elite_samples = None
elite_scores = None

for iteration in range(N_ITERATIONS):
    print(f"\n{'='*60}\nIteration {iteration+1}/{N_ITERATIONS}\n{'='*60}")

    samples_valid_cur, scores_valid_cur = collect_valid_samples(
        pipeline=Diffusion,
        ntake=NTAKE,
        device=device,
        vf=VOLUME_FRACTION,
        pass_fn=valuer0_pass,
        score_fn=valuer0_score,
        batch=GEN_BATCH,
        iteration=iteration,
        useddim=False,
        explore_scale=EXPLORE_SCALE,
    )

    sample_save_count = min(16, samples_valid_cur.shape[0])
    samples_to_save = samples_valid_cur[:sample_save_count]
    iteration_samples_path = iteration_samples_dir / f"samples_iter{iteration+1}.pt"
    torch.save(
        {
            "samples": samples_to_save.detach().cpu().clone(),
            "binary_samples": Matext_fvol(samples_to_save, vf=VOLUME_FRACTION).detach().cpu().clone(),
            "scores": scores_valid_cur[:sample_save_count].detach().cpu().clone(),
        },
        iteration_samples_path,
    )

    candidates_samples = []
    candidates_scores = []
    if elite_samples is not None and elite_samples.numel() > 0:
        candidates_samples.append(elite_samples)
        candidates_scores.append(elite_scores)

    candidates_samples.append(samples_valid_cur.detach())
    candidates_scores.append(scores_valid_cur.detach())
    candidates_samples = torch.cat(candidates_samples, dim=0)
    candidates_scores = torch.cat(candidates_scores, dim=0)

    sorted_indices = torch.argsort(candidates_scores, descending=True)[:NTAKE]
    elite_samples = candidates_samples[sorted_indices]
    elite_scores = candidates_scores[sorted_indices]
    score_ave = elite_scores.mean().item()

    best_sample = elite_samples[:1]
    best_score = elite_scores[0].item()
    plot_samples(best_sample, save_name=best_design_png_path, vf=VOLUME_FRACTION)
    torch.save(
        {
            "score": best_score,
            "sample": best_sample.squeeze(0).detach().cpu().clone(),
        },
        best_design_pt_path,
    )

    train_dataset = TensorDataset(elite_samples)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        bar = tqdm(train_loader)
        loss_sum = 0
        for batch_index, (images,) in enumerate(bar):
            optimizer.zero_grad()
            loss = Diffusion(switch=0, image=images, flag=0, type="x0")
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            bar.set_description(f"Iter {iteration+1}, Epoch {epoch+1}/{EPOCHS}, Loss: {loss_sum/(batch_index+1):.6f}")
    if SAVE_CKPTS:
        torch.save(model.state_dict(), ckpt_dir / f"Model_iter{iteration+1}.pt")

    preview_count = min(PREVIEW_BATCH, samples_valid_cur.shape[0])
    preview_samples = samples_valid_cur[:preview_count]
    preview_scores = scores_valid_cur[:preview_count]
    score_ave_preview = preview_scores.mean().item() if preview_count > 0 else 0.0

    preview_image_path = sample_previews_dir / f"samples_iter{iteration+1}.png" if SAVE_SAMPLE_PREVIEWS else None
    if SAVE_SAMPLE_PREVIEWS:
        plot_samples(preview_samples, save_name=preview_image_path, vf=VOLUME_FRACTION)

    with open(training_stats_path, "a") as f:
        f.write(f"{iteration+1}\t{score_ave:.6f}\t{score_ave_preview:.6f}\n")

    run_summary["metrics"].append(
        {
            "iteration": iteration + 1,
            "score_ave": score_ave,
            "score_ave_preview": score_ave_preview,
            "best_score": best_score,
            "best_design_png": str(best_design_png_path),
            "best_design_pt": str(best_design_pt_path),
            "preview_image": str(preview_image_path) if preview_image_path is not None else None,
            "iteration_samples_pt": str(iteration_samples_path),
            "iteration_samples_count": sample_save_count,
        }
    )
    with open(summary_json_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Iteration {iteration+1} - Score ave: {score_ave:.6f}, Preview: {score_ave_preview:.6f}, Best: {best_score:.6f}")
