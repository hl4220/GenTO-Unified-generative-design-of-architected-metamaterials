# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Run the F4 frequency-response optimisation with fourfold symmetry and transmission previews.

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
from model.utilities import collect_valid_samples, make_4fold_symmetric_tensor, plot_frequency_responses, plot_samples
from valuer4 import ONE_PASS_REGIONS, TransmissionSolver, valuer4_pass, valuer4_score_1

###############################################################################
# Model setup
###############################################################################

CKPT_PATH = MODELS_ROOT / "model_ckpt_pretrained" / "four_fold_pretrained.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionUNet(time_dim=256, condition_dim=256).to(device)

# Remove the "module." prefix if the checkpoint was saved from DataParallel.
state_dict = torch.load(CKPT_PATH, map_location=device)
new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True)
model.eval()
Diffusion = diffusion_pipeline(model, device).to(device)

###############################################################################
# Experiment settings and output folders
###############################################################################

TEST_NUM = 9
VOLUME_FRACTION = 0.5
N_ITERATIONS = 100
NTAKE = 256
GEN_BATCH = min(64, NTAKE*2) 
TRAIN_BATCH = min(64, NTAKE) 
PREVIEW_BATCH = 16
RESPONSE_PLOT_COUNT = 4
EPOCHS = 100
LEARNING_RATE = 1.0e-5
EXPLORE_SCALE = 1.1
SAVE_SAMPLE_PREVIEWS = True
SAVE_CKPTS = True

ckpt_dir = TASK_ROOT / f"model_ckpt_tuned_FQ_1pass_{TEST_NUM}"
output_dir = TASK_ROOT / f"outputs_FQ_1pass_{TEST_NUM}"
sample_previews_dir = output_dir / "sample_previews"
response_previews_dir = output_dir / "response_previews"
summary_json_path = output_dir / "run_summary.json"
training_stats_path = output_dir / "training_stats.txt"
best_design_png_path = output_dir / "best_design.png"
best_design_pt_path = output_dir / "best_design.pt"
output_dir.mkdir(exist_ok=True)
if SAVE_CKPTS:
    ckpt_dir.mkdir(exist_ok=True)
if SAVE_SAMPLE_PREVIEWS:
    sample_previews_dir.mkdir(exist_ok=True)
    response_previews_dir.mkdir(exist_ok=True)

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
        "response_previews_dir": str(response_previews_dir),
        "best_design_png": str(best_design_png_path),
        "best_design_pt": str(best_design_pt_path),
        "device": str(device),
        "test_num": TEST_NUM,
        "volume_fraction": VOLUME_FRACTION,
        "n_iterations": N_ITERATIONS,
        "ntake": NTAKE,
        "gen_batch": GEN_BATCH,
        "train_batch": TRAIN_BATCH,
        "preview_batch": PREVIEW_BATCH,
        "response_plot_count": RESPONSE_PLOT_COUNT,
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

frequency_solver = TransmissionSolver()
elite_samples = None
elite_scores = None

for iteration in range(N_ITERATIONS):
    print(f"\n{'='*60}\nIteration {iteration+1}/{N_ITERATIONS}\n{'='*60}")

    # Generate valid raw samples, then evaluate them after imposing fourfold symmetry.
    samples_valid_cur, scores_valid_cur = collect_valid_samples(
        pipeline=Diffusion,
        ntake=NTAKE,
        device=device,
        vf=VOLUME_FRACTION,
        pass_fn=valuer4_pass,
        score_fn=valuer4_score_1,
        batch=GEN_BATCH,
        iteration=iteration,
        useddim=False,
        explore_scale=EXPLORE_SCALE,
        sample_transform_fn=make_4fold_symmetric_tensor,
        store_transformed_samples=False,
    )

    candidates_samples = []
    candidates_scores = []

    # Keep the previous elite pool and merge it with the current valid samples.
    if elite_samples is not None and elite_samples.numel() > 0:
        candidates_samples.append(elite_samples)
        candidates_scores.append(elite_scores)

    candidates_samples.append(samples_valid_cur.detach().cpu())
    candidates_scores.append(scores_valid_cur.detach().cpu())

    candidates_samples = torch.cat(candidates_samples, dim=0)
    candidates_scores = torch.cat(candidates_scores, dim=0)

    sorted_indices = torch.argsort(candidates_scores, descending=True)[:NTAKE]
    elite_samples = candidates_samples[sorted_indices]
    elite_scores = candidates_scores[sorted_indices]
    score_ave = elite_scores.mean().item()

    # Save the best full fourfold design found so far in both visual and loadable formats.
    best_sample = elite_samples[:1]
    best_design = make_4fold_symmetric_tensor(best_sample)
    best_score = elite_scores[0].item()
    plot_samples(best_design, save_name=best_design_png_path, vf=VOLUME_FRACTION)
    torch.save(
        {
            "score": best_score,
            "sample": best_design.squeeze(0).detach().cpu().clone(),
        },
        best_design_pt_path,
    )

    # Fine-tune the generator on the current elite raw samples.
    train_dataset = TensorDataset(elite_samples)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        bar = tqdm(train_loader)
        Loss = 0
        for batch_index, (images,) in enumerate(bar):
            optimizer.zero_grad()
            loss = Diffusion(switch=0, image=images, flag=0, type="x0")
            loss.backward()
            Loss += loss.item()
            optimizer.step()
            bar.set_description(f"Iter {iteration+1}, Epoch {epoch+1}/{EPOCHS}, Loss: {Loss/(batch_index+1):.6f}")
    if SAVE_CKPTS:
        torch.save(model.state_dict(), ckpt_dir / f"Model_iter{iteration+1}.pt")

    # Preview.
    preview_count = min(PREVIEW_BATCH, samples_valid_cur.shape[0])
    preview_raw_samples = samples_valid_cur[:preview_count]
    preview_samples = make_4fold_symmetric_tensor(preview_raw_samples)
    preview_scores = scores_valid_cur[:preview_count]
    score_ave_preview = preview_scores.mean().item() if preview_count > 0 else 0.0

    preview_image_path = sample_previews_dir / f"samples_iter{iteration+1}.png" if SAVE_SAMPLE_PREVIEWS else None
    response_image_path = response_previews_dir / f"responses_iter{iteration+1}.png" if SAVE_SAMPLE_PREVIEWS else None
    if SAVE_SAMPLE_PREVIEWS:
        plot_samples(preview_samples, save_name=preview_image_path, vf=VOLUME_FRACTION)
        plot_frequency_responses(
            preview_samples,
            frequency_solver,
            response_image_path,
            iteration + 1,
            regions=ONE_PASS_REGIONS,
            n_examples=RESPONSE_PLOT_COUNT,
            vf=VOLUME_FRACTION,
        )

    # Record the iteration statistics for later plotting or comparison.
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
            "response_preview": str(response_image_path) if response_image_path is not None else None,
        }
    )
    with open(summary_json_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Iteration {iteration+1} - Score ave: {score_ave:.6f}, Preview: {score_ave_preview:.6f}, Best: {best_score:.6f}")
