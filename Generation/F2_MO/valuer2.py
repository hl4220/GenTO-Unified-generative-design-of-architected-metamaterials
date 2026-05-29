# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Evaluate morphology designs with periodic connectivity, fractal-dimension, and feature-size metrics.

import torch
import numpy as np
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries


def _binary_sample_arrays(binary_samples):
    """Convert a batch of tensors to 2D binary numpy arrays once per call."""
    if isinstance(binary_samples, np.ndarray):
        sample_batch = binary_samples
    else:
        sample_batch = binary_samples.detach().cpu().numpy()

    if sample_batch.ndim == 4:
        sample_batch = sample_batch[:, 0]

    return [(sample > 0.5).astype(int) for sample in sample_batch]


# *******************
# Valuer 2 Entry Points
# Defines the pass rule and score rule for the morphology-analysis task.
# *******************
def valuer2_pass(binary_samples, device):
    """Check whether each sample satisfies the periodic connectivity filter."""
    connectivity_results = torch.tensor(evaluate_connectivity(binary_samples, device), dtype=torch.bool)
    pass_list = connectivity_results
    return pass_list


def valuer2_score(binary_samples, device):
    """Return fractal-dimension and feature-size descriptors for each sample."""
    del device
    binary_arrays = _binary_sample_arrays(binary_samples)
    fractal_results = _compute_fractal_dimension_boundary(binary_arrays)
    minfeature_results = _compute_min_feature_size(binary_arrays)

    fractal_dims = torch.tensor([r[0] for r in fractal_results], dtype=torch.float32)
    minfeature_sizes = torch.tensor([r if not np.isnan(r) else 0.0 for r in minfeature_results], dtype=torch.float32)

    return fractal_dims.cpu(), minfeature_sizes.cpu()


# *******************
# Connectivity Check
# Tests whether the material phase forms a valid periodic connected network.
# *******************
def evaluate_connectivity(binary_samples, device=None):
    """Evaluate whether the design is fully periodic in both horizontal and vertical directions."""
    del device
    binary_arrays = _binary_sample_arrays(binary_samples)
    results = []
    for binary_np in binary_arrays:
        labeled_np = label(binary_np, connectivity=1)

        if labeled_np.max() == 0:
            results.append(False)
            continue

        parent = np.arange(labeled_np.max() + 1)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        periodic_rows = np.where(binary_np[:, 0] & binary_np[:, -1])[0]
        for row in periodic_rows:
            union(labeled_np[row, 0], labeled_np[row, -1])

        periodic_cols = np.where(binary_np[0, :] & binary_np[-1, :])[0]
        for col in periodic_cols:
            union(labeled_np[0, col], labeled_np[-1, col])

        material_labels = np.unique(labeled_np[labeled_np > 0])
        material_components = {find(lbl) for lbl in material_labels}
        left_labels = {find(lbl) for lbl in np.unique(labeled_np[:, 0]) if lbl > 0}
        right_labels = {find(lbl) for lbl in np.unique(labeled_np[:, -1]) if lbl > 0}
        top_labels = {find(lbl) for lbl in np.unique(labeled_np[0, :]) if lbl > 0}
        bottom_labels = {find(lbl) for lbl in np.unique(labeled_np[-1, :]) if lbl > 0}

        horizontal_components = left_labels & right_labels
        vertical_components = top_labels & bottom_labels
        supported_components = horizontal_components | vertical_components
        spanning_components = horizontal_components & vertical_components

        results.append(len(spanning_components) > 0 and material_components.issubset(supported_components))

    return results


# *******************
# Minimum Feature Size
# Measures the narrowest material bridge between neighboring void regions.
# *******************
def _compute_min_feature_size(binary_arrays):
    results = []
    for material in binary_arrays:
        H, W = material.shape
        material_tiled = np.tile(material, (3, 3))

        void = material_tiled == 0
        labeled_voids = label(void)
        n_voids = labeled_voids.max()

        if n_voids < 2:
            results.append(np.nan)
            continue

        dist_map, indices = distance_transform_edt(~void, return_indices=True)
        nearest_void_label = labeled_voids[indices[0], indices[1]]
        material_mask = material_tiled > 0

        min_dist = np.inf

        for dr, dc in [(0, 1), (1, 0)]:
            p_label = nearest_void_label[:-dr or None, :-dc or None]
            n_label = nearest_void_label[dr:, dc:]
            p_dist = dist_map[:-dr or None, :-dc or None]
            n_dist = dist_map[dr:, dc:]
            p_mat = material_mask[:-dr or None, :-dc or None]
            n_mat = material_mask[dr:, dc:]

            diff_mask = p_mat & n_mat & (p_label != n_label) & (p_label > 0) & (n_label > 0)
            if not diff_mask.any():
                continue

            total_dist = p_dist + n_dist
            d = total_dist[diff_mask].min()

            if d < min_dist:
                min_dist = d

        if min_dist == np.inf or np.isinf(min_dist):
            results.append(np.nan)
        else:
            results.append(min_dist)

    return results


def compute_min_feature_size_2(binary_samples):
    """Estimate the minimum feature size of each sample using periodic tiling."""
    return _compute_min_feature_size(_binary_sample_arrays(binary_samples))


# *******************
# Boundary Fractal Dimension
# Estimates geometric complexity from the material boundary by box counting.
# *******************
def _compute_fractal_dimension_boundary(binary_arrays, min_box_size=2, max_box_size=None):
    results = []
    for binary in binary_arrays:
        boundary = find_boundaries(binary > 0, mode="inner")

        if max_box_size is None:
            max_box_size_i = min(binary.shape) // 4
        else:
            max_box_size_i = max_box_size

        scales, counts = [], []
        box_size = min_box_size

        while box_size <= max_box_size_i:
            n_boxes_x = int(np.ceil(boundary.shape[1] / box_size))
            n_boxes_y = int(np.ceil(boundary.shape[0] / box_size))
            count = 0

            for ii in range(n_boxes_y):
                for j in range(n_boxes_x):
                    y_start, y_end = ii * box_size, min((ii + 1) * box_size, boundary.shape[0])
                    x_start, x_end = j * box_size, min((j + 1) * box_size, boundary.shape[1])
                    if np.any(boundary[y_start:y_end, x_start:x_end]):
                        count += 1

            scales.append(box_size)
            counts.append(count)
            box_size = int(box_size * 1.5)
            if box_size == int(box_size / 1.5 * 1.5):
                box_size += 1

        if len(scales) < 3:
            results.append((0.0, 0.0))
            continue

        scales = np.array(scales, dtype=float)
        counts = np.array(counts, dtype=float)
        valid = counts > 0
        scales, counts = scales[valid], counts[valid]

        if len(scales) < 3:
            results.append((0.0, 0.0))
            continue

        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        fractal_dim = -coeffs[0]

        fitted = np.polyval(coeffs, np.log(scales))
        ss_res = np.sum((np.log(counts) - fitted) ** 2)
        ss_tot = np.sum((np.log(counts) - np.mean(np.log(counts))) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results.append((fractal_dim, r_squared))

    return results


def compute_fractal_dimension_boundary(binary_samples, min_box_size=2, max_box_size=None):
    """Compute boundary fractal dimension and fit quality for each sample."""
    return _compute_fractal_dimension_boundary(
        _binary_sample_arrays(binary_samples),
        min_box_size=min_box_size,
        max_box_size=max_box_size,
    )


__all__ = [
    "compute_fractal_dimension_boundary",
    "compute_min_feature_size_2",
    "evaluate_connectivity",
    "valuer2_pass",
    "valuer2_score",
]
