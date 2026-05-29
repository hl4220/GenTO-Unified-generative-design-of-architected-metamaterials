# Training

This folder contains the data-loading and training code for the GenTO diffusion topology prior. The training script reads packed binary topology datasets, converts them into signed-distance-like surface fields, and trains the unconditional diffusion model.

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

## Download Training Data

The packed `.npy` datasets are not included in this GitHub version because of file-size limits. Please download them from Google Drive:

https://drive.google.com/drive/folders/1iLriVSi7aBi89xtdFyk8-OWidYB0o3ys?usp=drive_link

Place the packed data files into the corresponding folders:

```text
Training/Binary_Image_Packed/periodic/
Training/Binary_Image_Packed/four_fold/
Training/Binary_Image_Packed/eight_fold/
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

