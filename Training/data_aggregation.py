# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Discover packed binary image datasets and decode packed numpy arrays for training.

from pathlib import Path

import numpy as np

###############################################################################
# Packed binary dataset layout
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Binary_Image_Packed"
STRUCTURE_MAP = {"periodic": 0, "four_fold": 1, "eight_fold": 2}
MATERIAL_MAP = {"CHPF": 0, "GRF": 1, "PR": 2, "TS": 3}


def _as_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _material_name(path):
    return path.stem.split("_")[0]


###############################################################################
# Discovery and decoding
###############################################################################

def discover_packed_files(root_dir=DATA_ROOT, groups=None, materials=None):
    """Return metadata for packed npy files without loading image data into RAM."""
    root_dir = Path(root_dir)
    selected_groups = _as_list(groups, STRUCTURE_MAP.keys())
    selected_materials = set(_as_list(materials, MATERIAL_MAP.keys()))
    records = []

    for group in selected_groups:
        if group not in STRUCTURE_MAP:
            raise ValueError(f"Unknown structure group: {group}")

        group_dir = root_dir / group
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing packed data folder: {group_dir}")

        for path in sorted(group_dir.glob("*.npy")):
            material = _material_name(path)
            if material not in selected_materials:
                continue
            if material not in MATERIAL_MAP:
                raise ValueError(f"Unknown material prefix in {path.name}")

            packed = np.load(path, mmap_mode="r")
            if packed.ndim != 3:
                raise ValueError(f"Expected {path} to have shape (N, H, W/8), got {packed.shape}")

            n_images, height, packed_width = packed.shape
            records.append(
                {
                    "path": path,
                    "group": group,
                    "material": material,
                    "length": int(n_images),
                    "packed_shape": tuple(int(v) for v in packed.shape),
                    "image_shape": (int(height), int(packed_width * 8)),
                    "dtype": str(packed.dtype),
                    "structure_label": STRUCTURE_MAP[group],
                    "material_label": MATERIAL_MAP[material],
                }
            )
            del packed

    if not records:
        raise ValueError(f"No packed npy files found under {root_dir}")

    return records


def unpack_binary_image(packed_image):
    """Decode one packed row from (H, W/8) uint8 back to a (H, W) bool image."""
    return np.unpackbits(packed_image, axis=-1).astype(bool)


###############################################################################
# Optional manifest export
###############################################################################

def build_manifest(root_dir=DATA_ROOT, save_path=PROJECT_ROOT / "packed_data_manifest.npy", groups=None, materials=None):
    records = discover_packed_files(root_dir=root_dir, groups=groups, materials=materials)
    manifest = {
        "files": [{**record, "path": str(record["path"])} for record in records],
        "label_maps": {
            "structure": STRUCTURE_MAP,
            "material": MATERIAL_MAP,
        },
    }
    np.save(save_path, manifest)
    return manifest


if __name__ == "__main__":
    manifest = build_manifest()
    total = sum(record["length"] for record in manifest["files"])
    print(f"Saved packed_data_manifest.npy with {len(manifest['files'])} files and {total} images.")
