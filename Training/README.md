# Training

This folder contains the data-loading and **pretraining** code for the GenTO geometric diffusion topology prior. The training script reads packed binary topology datasets, converts them into signed-distance-function (SDF) fields, and trains the unconditional diffusion model.

The pretrained GenTO checkpoints used in this repository have already been trained and are provided through the download links below. Most users do not need to run pretraining before using the generation examples. This code is released for researchers who want to train a new topology prior on their own geometric datasets, or continue from the provided pretrained prior with additional data for transfer learning.

## Folder Structure

```text
Training/
├── train.py                         # Main diffusion-model training script
├── data_aggregation.py              # Packed-data discovery and decoding utilities
├── model/                           # Local copy of the diffusion model used for training
└── Binary_Image_Packed/
    ├── periodic/                    # Periodic packed datasets
    ├── four_fold/                   # Fourfold-symmetric packed datasets
    └── eight_fold/                  # Eightfold-symmetric packed datasets
```

## Download Training Data and Checkpoints

The packed `.npy` datasets and pretrained checkpoints are not included in this GitHub version because of file-size limits. Please download them from Google Drive:

https://drive.google.com/drive/folders/1iLriVSi7aBi89xtdFyk8-OWidYB0o3ys?usp=drive_link

They are also archived on Zenodo:

https://doi.org/10.5281/zenodo.20452173

Place the packed data files into the corresponding folders:

```text
Training/Binary_Image_Packed/periodic/
Training/Binary_Image_Packed/four_fold/
Training/Binary_Image_Packed/eight_fold/
```

Place the pretrained checkpoint files into:

```text
Generation/models/model_ckpt_pretrained/
```

Expected layout:

```text
Binary_Image_Packed/
├── periodic/
│   ├── CHPF_P_2D.npy
│   ├── GRF_P_2D.npy
│   ├── PR_P_2D.npy
│   └── TS_P_2D.npy
├── four_fold/
│   ├── CHPF_S_2D.npy
│   ├── GRF_S_2D.npy
│   ├── PR_S_2D.npy
│   └── TS_S_2D.npy
└── eight_fold/
    ├── CHPF_I_2D.npy
    ├── GRF_I_2D.npy
    ├── PR_I_2D.npy
    └── TS_I_2D.npy
```

## Run Training

Select the dataset group in `train.py`:

```python
DATA_GROUP = "periodic"   # Options: "periodic", "four_fold", "eight_fold"
```

Then run:

```bash
cd Training
python train.py
```

Training checkpoints and sample previews are saved to:

```text
Training/checkpoints_iso_original/
```
