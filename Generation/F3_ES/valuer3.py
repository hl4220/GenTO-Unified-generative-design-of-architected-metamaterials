# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Evaluate elasticity designs with connectivity, feature-size, and FFT homogenization metrics.

import torch
import numpy as np
import itertools
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, binary_opening, disk, remove_small_objects


# *******************
# Runtime Cache
# Stores the elasticity solver instance so it can be reused across calls.
# *******************
MODEL_ELASTICITY = None
MODEL_ELASTICITY_KEY = None


PERIODIC_TARGET = [
    [0.5919223, -0.23617198, -0.01492361],
    [-0.23617198, 0.29432666, 0.00355324],
    [-0.01492361, 0.00355324, 0.06040507],
]

FOURFOLD_TARGET = [
    [0.91168091, -0.15954416, 0.0],
    [-0.15954416, 0.22792023, 0.0],
    [0.0, 0.0, 0.08],
]

EIGHTFOLD_TARGET = [
    [0.7977208, -0.27920228, 0.],
    [-0.27920228, 0.7977208, 0.],
    [ 0., 0., 0.25]
]


def _binary_sample_arrays(binary_samples):
    """Convert a batch of tensors to 2D binary numpy arrays once per call."""
    if isinstance(binary_samples, np.ndarray):
        sample_batch = binary_samples
    else:
        sample_batch = binary_samples.detach().cpu().numpy()

    if sample_batch.ndim == 4:
        sample_batch = sample_batch[:, 0]

    return [(sample > 0.5).astype(int) for sample in sample_batch]


def _get_elasticity_model(N, device):
    """Reuse the elasticity solver when the problem size and device stay the same."""
    global MODEL_ELASTICITY, MODEL_ELASTICITY_KEY
    model_key = (N, str(device))
    if MODEL_ELASTICITY is None or MODEL_ELASTICITY_KEY != model_key:
        MODEL_ELASTICITY = FFT_Elasticity(2, N, l1=8.33, l2=3.86, device=device)
        MODEL_ELASTICITY_KEY = model_key
    return MODEL_ELASTICITY


def _score_against_target(binary_samples, target_values, device):
    """Score each sample by the negative mean absolute error to a target tensor."""
    N = binary_samples.shape[2]
    model = _get_elasticity_model(N, device)
    H_target = torch.tensor(target_values, device=device)[None, :, :]

    phases = binary_samples[:, 0, :, :].to(device)
    H = torch.stack([model.forward(phase) for phase in phases], dim=0)
    error = torch.mean(torch.abs(H - H_target), dim=(1, 2))
    return -error


# *******************
# Valuer 3 Entry Points
# Defines the pass rule and score rules for the elasticity-design task.
# *******************
def valuer3_pass(binary_samples, device):
    """Check whether each sample passes the connectivity and feature-size filters."""
    binary_arrays = _binary_sample_arrays(binary_samples)
    connectivity_results = torch.tensor(evaluate_connectivity(binary_samples, device), dtype=torch.bool)
    min_feature_size_results = _compute_min_feature_size(binary_arrays)
    pass_list = connectivity_results & min_feature_size_results
    return pass_list


def valuer3_score_periodic(binary_samples, device):
    """Score each sample against the periodic target elasticity tensor."""
    return _score_against_target(binary_samples, PERIODIC_TARGET, device)


def valuer3_score_fourfold(binary_samples, device):
    """Score each sample against the fourfold target elasticity tensor."""
    return _score_against_target(binary_samples, FOURFOLD_TARGET, device)


def valuer3_score_eightfold(binary_samples, device):
    """Score each sample against the eightfold target elasticity tensor."""
    return _score_against_target(binary_samples, EIGHTFOLD_TARGET, device)


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
# Uses a skeleton-based thickness estimate to reject overly thin structures.
# *******************
def _compute_min_feature_size(binary_arrays, threshold=10.0, remove_noise=True, noise_size=5, opening_radius=2):
    results = []
    for material in binary_arrays:
        H, W = material.shape
        material_tiled = np.tile(material, (3, 3))

        if remove_noise:
            material_tiled = binary_opening(material_tiled, disk(opening_radius))
            material_tiled = remove_small_objects(material_tiled.astype(bool), min_size=noise_size).astype(int)

        dist_map_full = distance_transform_edt(material_tiled)
        skeleton_full = skeletonize(material_tiled > 0)

        center_dist = dist_map_full[H:2 * H, W:2 * W]
        center_skeleton = skeleton_full[H:2 * H, W:2 * W]
        skeleton_distances = center_dist[center_skeleton]

        if len(skeleton_distances) == 0:
            results.append(False)
            continue

        min_radius = np.min(skeleton_distances)
        min_feature = 2 * min_radius
        results.append(min_feature > threshold)

    return torch.tensor(results, dtype=torch.bool)


def compute_min_feature_size(binary_samples, threshold=5.0, remove_noise=True, noise_size=5, opening_radius=2):
    """Return whether each sample keeps a minimum skeleton-based feature thickness."""
    return _compute_min_feature_size(
        _binary_sample_arrays(binary_samples),
        threshold=threshold,
        remove_noise=remove_noise,
        noise_size=noise_size,
        opening_radius=opening_radius,
    )


# *******************
# FFT Elasticity Solver
# Computes homogenized elastic properties for binary material layouts.
# *******************
class FFT_Elasticity:
    """FFT-based homogenization model for effective elasticity."""

    def __init__(self, ndim, N, l1, l2, device):
        """Initialize the elasticity solver and its spectral operators."""
        super(FFT_Elasticity, self).__init__()
        self.dtype = torch.float32
        self.dtype2 = torch.complex64
        self.device = device
        self.ndim = ndim
        self.N = N

        i = torch.eye(ndim)
        self.I = torch.einsum("ij,xy", i, torch.ones([N, N]))
        self.I4 = torch.einsum("ijkl,xy->ijklxy", torch.einsum("il,jk", i, i), torch.ones([N, N]))
        self.I4rt = torch.einsum("ijkl,xy->ijklxy", torch.einsum("ik,jl", i, i), torch.ones([N, N]))
        self.I4s = (self.I4 + self.I4rt) / 2.0
        self.II = torch.einsum("ijxy,klxy->ijklxy", self.I, self.I)
        self.I4 = self.I4.to(self.device)
        self.I4s = self.I4s.to(self.device)
        self.II = self.II.to(self.device)

        self.ddot42 = lambda A4, B2: torch.einsum("ijklxy,lkxy -> ijxy", A4, B2)
        self.Ptran = lambda A2, B3: torch.einsum("hl,lxy -> hxy", A2, B3)
        self.fft = lambda x: torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x), [N, N]))
        self.ifft = lambda x: torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x), [N, N]))
        self.G = lambda A2: torch.real(self.ifft(self.ddot42(self.Ghat4, self.fft(A2)))).reshape(-1)
        self.K_deps = lambda depsm: self.ddot42(self.K4, depsm.reshape(self.solsize))
        self.G_K_deps = lambda depsm: self.G(self.K_deps(depsm))
        self.param = lambda M0, M1: M0 * torch.ones([self.N, self.N], dtype=self.dtype).to(self.device) * (1.0 - self.phase) + M1 * torch.ones([self.N, self.N], dtype=self.dtype).to(self.device) * self.phase

        self.l1 = l1
        self.l2 = l2

        eo = (torch.pi * torch.arange(-self.N / 2, self.N / 2) / self.N).to(self.dtype).to(self.device)
        ex, ey = torch.meshgrid(eo, eo, indexing="ij")
        freqx, freqy = torch.sin(ex) * torch.cos(ey), torch.cos(ex) * torch.sin(ey)
        self.xi = torch.stack([freqx, freqy], dim=0)
        self.Ghat4 = self.GreenOperator()

    def forward(self, phase):
        """Return the homogenized elasticity tensor for one binary phase field."""
        self.solsize = [self.ndim, self.ndim, self.N, self.N]
        self.constitutive(phase)
        eps1, sig1 = self.solving([1.0, 0, 0], phase)
        eps2, sig2 = self.solving([0, 1.0, 0], phase)
        eps3, sig3 = self.solving([0, 0, 1.0], phase)
        homo_stress = self.homo_properties(sig1, sig2, sig3)
        return homo_stress

    def solving(self, macro_eps, phase):
        """Solve the local elasticity field for one prescribed macroscopic strain."""
        sig = torch.zeros(self.solsize, dtype=self.dtype).to(self.device)
        eps = torch.zeros(self.solsize, dtype=self.dtype).to(self.device)

        DE = torch.zeros(self.solsize, dtype=self.dtype).to(self.device)
        DE[0, 0] += macro_eps[0]
        DE[1, 1] += macro_eps[1]
        DE[1, 0] += macro_eps[2]
        DE[0, 1] += macro_eps[2]

        eps += DE
        sig = self.ddot42(self.K4, eps.to(self.dtype))
        b = -self.G(sig)

        En = torch.linalg.norm(eps).to(self.device)
        for iiter in range(10):
            depsm, _ = self.minres(self.G_K_deps, b, maxit=100, tol=1e-4)
            eps += depsm.reshape(self.solsize)
            sig = self.ddot42(self.K4, eps.to(self.dtype))
            b = -self.G(sig)
            if torch.linalg.norm(depsm) / En < 1.0e-5 and iiter > 0:
                break

        return eps, sig

    def constitutive(self, phase):
        """Build the local elastic constitutive tensor field from the phase map."""
        self.phase = phase
        self.K = self.param(self.l1 * 0.0, self.l1)[None, None, None, None, :, :]
        self.mu = self.param(self.l2 * 0.0, self.l2)[None, None, None, None, :, :]
        self.K4 = self.K * self.II + 2.0 * self.mu * (self.I4s - 1.0 / 3.0 * self.II)

    def GreenOperator(self):
        """Precompute the Fourier-space Green operator for elasticity."""
        qdotq = self.xi[0] ** 2 + self.xi[1] ** 2
        mask = qdotq != 0
        I = torch.eye(self.ndim, dtype=self.dtype, device=self.device)
        Ghat4 = torch.zeros(self.ndim, self.ndim, self.ndim, self.ndim, self.N, self.N, dtype=self.dtype, device=self.device)
        for i, j, l, m in itertools.product(range(self.ndim), repeat=4):
            t1 = -(self.xi[i] * self.xi[j] * self.xi[l] * self.xi[m]) / (qdotq ** 2)
            t2 = (
                I[j, l] * self.xi[i] * self.xi[m]
                + I[j, m] * self.xi[i] * self.xi[l]
                + I[i, l] * self.xi[j] * self.xi[m]
                + I[i, m] * self.xi[j] * self.xi[l]
            ) / (2 * qdotq)
            Ghat4[i, j, l, m][mask] = (t1 + t2)[mask]
        return Ghat4.to(self.dtype2)

    def homo_properties(self, sig1, sig2, sig3):
        """Average local stresses to obtain the homogenized elasticity tensor."""
        p11 = torch.mean(sig1[0, 0, :, :], dim=(-2, -1))
        p21 = torch.mean(sig1[1, 1, :, :], dim=(-2, -1))
        p31 = (torch.mean(sig1[0, 1, :, :], dim=(-2, -1)) + torch.mean(sig1[1, 0, :, :], dim=(-2, -1))) / 2

        p12 = torch.mean(sig2[0, 0, :, :], dim=(-2, -1))
        p22 = torch.mean(sig2[1, 1, :, :], dim=(-2, -1))
        p32 = (torch.mean(sig2[0, 1, :, :], dim=(-2, -1)) + torch.mean(sig2[1, 0, :, :], dim=(-2, -1))) / 2

        p13 = torch.mean(sig3[0, 0, :, :], dim=(-2, -1)) / 2
        p23 = torch.mean(sig3[1, 1, :, :], dim=(-2, -1)) / 2
        p33 = (torch.mean(sig3[0, 1, :, :], dim=(-2, -1)) + torch.mean(sig3[1, 0, :, :], dim=(-2, -1))) / 4

        homo_stress = torch.stack([
            torch.stack([p11, p12, p13], dim=0),
            torch.stack([p21, p22, p23], dim=0),
            torch.stack([p31, p32, p33], dim=0),
        ], dim=1)
        return homo_stress

    def minres(self, A, b, maxit, tol):
        """Solve the linear system with a lightweight MINRES-style iteration."""
        tol2 = torch.tensor(tol ** 2)
        x = torch.zeros_like(b)
        r = b - A(x)
        p0 = r.clone()
        s0 = A(p0)
        p1 = p0.clone()
        s1 = s0.clone()

        for i in range(maxit):
            p2, p1 = p1, p0
            s2, s1 = s1, s0

            alpha = torch.dot(r, s1) / (torch.dot(s1, s1) + 1e-8)
            if torch.isnan(alpha):
                break

            x += alpha * p1
            r -= alpha * s1

            jud = torch.dot(r, r)
            if jud.item() < tol2:
                break

            p0 = s1.clone()
            s0 = A(s1)

            beta1 = torch.dot(s0, s1) / (torch.dot(s1, s1) + 1e-8)
            p0 -= beta1 * p1
            s0 -= beta1 * s1

            if i > 1:
                beta2 = torch.dot(s0, s2) / (torch.dot(s2, s2) + 1e-8)
                p0 -= beta2 * p2
                s0 -= beta2 * s2

        return x, r


__all__ = [
    "FFT_Elasticity",
    "compute_min_feature_size",
    "evaluate_connectivity",
    "valuer3_pass",
    "valuer3_score_eightfold",
    "valuer3_score_fourfold",
    "valuer3_score_periodic",
]
