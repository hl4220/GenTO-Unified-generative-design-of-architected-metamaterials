# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Evaluate heat-conductivity designs with connectivity, feature-size, and FFT homogenization metrics.

import torch
import torch.nn as nn
import numpy as np
from skimage.measure import label
from scipy.ndimage import distance_transform_edt


# *******************
# Runtime Cache
# Stores the heat solver instance so it can be reused across calls.
# *******************
MODEL_HEAT = None


def _binary_sample_arrays(binary_samples):
    """Convert a batch of tensors to 2D binary numpy arrays once per call."""
    if isinstance(binary_samples, np.ndarray):
        sample_batch = binary_samples
    else:
        sample_batch = binary_samples.detach().cpu().numpy()

    if sample_batch.ndim == 4:
        sample_batch = sample_batch[:, 0]

    return [(sample > 0.5) for sample in sample_batch]


# *******************
# Valuer 1 Entry Points
# Defines the pass rule and score rule for the heat-conductivity task.
# *******************
def valuer1_pass(binary_samples, device, min_feature_threshold=5.0):
    """Check whether each sample passes the connectivity and feature-size filters."""
    connectivity_results = torch.tensor(evaluate_connectivity(binary_samples, device), dtype=torch.bool)
    min_feature_sizes = compute_min_feature_size_2(binary_samples)
    min_feature_results = torch.tensor(
        [(not np.isnan(r)) and (r > min_feature_threshold) for r in min_feature_sizes],
        dtype=torch.bool,
    )
    pass_list = connectivity_results & min_feature_results
    return pass_list


def valuer1_score(binary_samples, device):
    """Score each sample by effective heat conductivity with a small anisotropy penalty."""
    global MODEL_HEAT
    N = binary_samples.shape[2]

    if MODEL_HEAT is None:
        MODEL_HEAT = FFT_Heat(N=N, device=device)

    phases = binary_samples[:, 0, :, :].to(device)
    H = MODEL_HEAT.forward(phases)

    k_mean = (H[:, 0] + H[:, 3]) / 2.0
    anisotropy = torch.abs(H[:, 0] - H[:, 3])
    score = k_mean - 0.1 * anisotropy
    return score


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
def compute_min_feature_size_2(binary_samples):
    """Estimate the minimum feature size of each sample using periodic tiling."""
    results = []
    for i in range(binary_samples.shape[0]):
        img = binary_samples[i, 0].cpu().numpy() if binary_samples.ndim == 4 else binary_samples[i].cpu().numpy()
        binary = (img > 0.5).astype(int)
        material = binary

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


# *******************
# FFT Heat Solver
# Computes homogenized heat response for binary material layouts.
# *******************
class FFT_Heat(nn.Module):
    """FFT-based homogenization model for effective heat conductivity."""

    def __init__(self, N, device):
        """Initialize the spectral solver and precompute its Green operator."""
        super(FFT_Heat, self).__init__()
        self.L = 1.0
        self.device = device
        self.K1 = 1.0
        self.K2 = 1e-3
        self.N = N

        half = self.N // 2
        first_half = torch.arange(0, half, dtype=torch.float32)
        second_half = torch.arange(-half, 0, dtype=torch.float32)
        self.e1t = 2.0 * torch.pi * torch.cat([first_half, second_half]) / self.L
        self.e2t = 2.0 * torch.pi * torch.cat([first_half, second_half]) / self.L
        self.GreenFunction()

    def forward(self, phase):
        """Return the homogenized conductivity tensor for a batch of phases."""
        self.INF = phase
        q1x, q1y = self.solving([1.0, 0])
        q2x, q2y = self.solving([0, 1.0])
        Q = torch.cat((q1x[:, None, :, :], q1y[:, None, :, :], q2x[:, None, :, :], q2y[:, None, :, :]), dim=1)
        H = torch.mean(Q, dim=(-2, -1))
        return H

    def solving(self, Load):
        """Solve the FFT fixed-point iteration for a given macroscopic thermal load."""
        K_lib = self.Global_Constitute().to(self.device)
        batch = K_lib.shape[0]
        RVE_N1, RVE_N2 = self.N, self.N
        K_0 = self.K_0

        unit_matrix = torch.ones((batch, RVE_N1, RVE_N2)).to(self.device)
        RVE_x11 = Load[0] * unit_matrix
        RVE_x22 = Load[1] * unit_matrix
        RVE_q11 = K_lib * RVE_x11
        RVE_q22 = K_lib * RVE_x22

        for _ in range(500):
            RVE_tao11 = RVE_q11 - K_0 * RVE_x11
            RVE_tao22 = RVE_q22 - K_0 * RVE_x22

            F_RVE_tao11 = torch.fft.fft2(RVE_tao11, dim=(-2, -1))
            F_RVE_tao22 = torch.fft.fft2(RVE_tao22, dim=(-2, -1))

            RVE_XF11 = -self.P_11_lib * F_RVE_tao11 - self.P_12_lib * F_RVE_tao22
            RVE_XF11[:, 0, 0] = Load[0] * RVE_N1 * RVE_N2
            RVE_XF22 = -self.P_21_lib * F_RVE_tao11 - self.P_22_lib * F_RVE_tao22
            RVE_XF22[:, 0, 0] = Load[1] * RVE_N1 * RVE_N2

            RVE_x11 = torch.fft.ifft2(RVE_XF11, dim=(-2, -1)).real
            RVE_x22 = torch.fft.ifft2(RVE_XF22, dim=(-2, -1)).real

            RVE_q11 = K_lib * RVE_x11
            RVE_q22 = K_lib * RVE_x22

        return RVE_q11.real, RVE_q22.real

    def Global_Constitute(self):
        """Build the scalar conductivity field from the binary phase tensor."""
        Phase1 = self.INF > 0.5
        Phase2 = ~Phase1
        Phase = torch.zeros((self.INF.size(0), self.N, self.N), dtype=torch.float32)
        Phase[Phase1] = self.K1
        Phase[Phase2] = self.K2
        return Phase

    def GreenFunction(self):
        """Precompute the Fourier-space Green operator used by the solver."""
        RVE_ksi = self.e1t
        RVE_eta = self.e2t
        RVE_L1 = self.L
        RVE_L2 = self.L
        RVE_N1 = self.N
        RVE_N2 = self.N

        self.K_0 = 0.5 * (self.K1 + self.K2)
        K_0_matrix = torch.tensor([[self.K_0, 0], [0, self.K_0]])

        P_11_lib = torch.zeros((self.N, self.N), dtype=torch.float32)
        P_12_lib = torch.zeros((self.N, self.N), dtype=torch.float32)
        P_21_lib = torch.zeros((self.N, self.N), dtype=torch.float32)
        P_22_lib = torch.zeros((self.N, self.N), dtype=torch.float32)

        for ni in range(RVE_N1):
            for nj in range(RVE_N2):
                pr1 = RVE_ksi[ni]
                pr2 = RVE_eta[nj]

                r1 = (RVE_N1 / (0.5 * RVE_L1)) * torch.sin(0.5 * pr1 * RVE_L1 / RVE_N1) * torch.cos(0.5 * pr2 * RVE_L2 / RVE_N2)
                r2 = (RVE_N2 / (0.5 * RVE_L2)) * torch.cos(0.5 * pr1 * RVE_L1 / RVE_N1) * torch.sin(0.5 * pr2 * RVE_L2 / RVE_N2)

                d2 = r1**2 + r2**2
                all_r = torch.tensor([r1, r2])
                down_d = 0.0
                for n_down_i in range(2):
                    for n_down_j in range(2):
                        down_d += all_r[n_down_i] * all_r[n_down_j] * K_0_matrix[n_down_i, n_down_j]

                if d2 > 0:
                    P_11_lib[ni, nj] = r1 * r1 / down_d
                    P_12_lib[ni, nj] = r1 * r2 / down_d
                    P_21_lib[ni, nj] = r2 * r1 / down_d
                    P_22_lib[ni, nj] = r2 * r2 / down_d

        P_11_lib[0, 0] = 0
        P_12_lib[0, 0] = 0
        P_21_lib[0, 0] = 0
        P_22_lib[0, 0] = 0

        self.P_11_lib = P_11_lib[None, :, :].to(self.device)
        self.P_12_lib = P_12_lib[None, :, :].to(self.device)
        self.P_21_lib = P_21_lib[None, :, :].to(self.device)
        self.P_22_lib = P_22_lib[None, :, :].to(self.device)


__all__ = [
    "FFT_Heat",
    "compute_min_feature_size_2",
    "evaluate_connectivity",
    "valuer1_pass",
    "valuer1_score",
]
