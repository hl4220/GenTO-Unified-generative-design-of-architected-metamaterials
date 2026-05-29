# Generation

This folder contains the generation and task-adaptation code for GenTO. It uses pretrained diffusion checkpoints as reusable topology priors, then steers the generated topology distribution toward task-specific objectives and constraints.

## Folder Structure

```text
Generation/
├── F0_TOY/                  # Lightweight toy target-matching example
├── F1_HE/                   # Heat-conductivity optimisation
├── F2_MO/                   # Multi-objective morphology optimisation
├── F3_ES/                   # Elasticity-targeted optimisation
├── F4_FQ/                   # Frequency-response / transmission optimisation
└── models/
    ├── model/               # Diffusion model and shared utilities
    └── model_ckpt_pretrained/
        ├── periodic_pretrained.pt
        ├── four_fold_pretrained.pt
        └── eight_fold_pretrained.pt
```

## Download Checkpoints

The pretrained checkpoints are not included in this GitHub version because of file-size limits. Please download them from Google Drive:

https://drive.google.com/drive/folders/1iLriVSi7aBi89xtdFyk8-OWidYB0o3ys?usp=drive_link

Place the checkpoint files here:

```text
Generation/models/model_ckpt_pretrained/
```

Expected files:

```text
periodic_pretrained.pt
four_fold_pretrained.pt
eight_fold_pretrained.pt
```

## Run Generation Examples

Run each example from its own task folder:

```bash
cd Generation/F0_TOY
python Djob_TOY.py
```

```bash
cd Generation/F1_HE
python Djob_HE_max.py
```

```bash
cd Generation/F2_MO
python Djob_MO.py
```

```bash
cd Generation/F3_ES
python Djob_ES_periodic.py
python Djob_ES_fourfold.py
python Djob_ES_eightfold.py
```

```bash
cd Generation/F4_FQ
python Djob_FQ.py
```

Each script writes optimisation outputs into a local `outputs_*` folder. If checkpoint saving is enabled in the script, tuned model checkpoints are written into a local `model_ckpt_tuned_*` folder.

