# Authors: Haolin Li, Yuyang Miao
# Date: 2026-05-29
# Description: Evaluate acoustic transmission designs with connectivity checks and finite-element frequency response.

import torch
import numpy as np
from skimage.measure import label
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, spsolve


# *******************
# Runtime Cache
# Stores the transmission solver instance so it can be reused across calls.
# *******************
_FREQUENCY_SOLVER = None
_FREQUENCY_SOLVER_KEY = None


ONE_PASS_REGIONS = {
    "gap": [(1e3, 2e3, 0.25), (4e3, 5e3, 0.25)],
    "pass": [(2e3, 4e3, 0.5)],
}

TWO_PASS_REGIONS = {
    "gap": [(2e3, 4e3, 0.5)],
    "pass": [(1e3, 2e3, 0.25), (4e3, 5e3, 0.25)],
}



def _binary_sample_arrays(binary_samples):
    """Convert a batch of tensors to 2D binary numpy arrays once per call."""
    if isinstance(binary_samples, np.ndarray):
        sample_batch = binary_samples
    else:
        sample_batch = binary_samples.detach().cpu().numpy()

    if sample_batch.ndim == 4:
        sample_batch = sample_batch[:, 0]

    return [(sample > 0.5) for sample in sample_batch]


def _get_frequency_solver(binary_shape):
    """Reuse the transmission solver when the binary image size stays the same."""
    global _FREQUENCY_SOLVER, _FREQUENCY_SOLVER_KEY
    solver_key = tuple(binary_shape)
    if _FREQUENCY_SOLVER is None or _FREQUENCY_SOLVER_KEY != solver_key:
        _FREQUENCY_SOLVER = TransmissionSolver(nelx=binary_shape[1], nely=binary_shape[0])
        _FREQUENCY_SOLVER_KEY = solver_key
    return _FREQUENCY_SOLVER


def _score_transmission_curve(freqs, T_dB, regions, threshold=-10.0):
    """Evaluate a transmission curve against weighted gap and passband targets."""
    score = 0.0
    for lower, upper, weight in regions["gap"]:
        mask = (freqs >= lower) & (freqs <= upper)
        score += np.mean(T_dB[mask] < threshold) * weight
    for lower, upper, weight in regions["pass"]:
        mask = (freqs >= lower) & (freqs <= upper)
        score += np.mean(T_dB[mask] > threshold) * weight
    return score


def _score_binary_samples(binary_arrays, regions):
    """Score a batch of binary samples with the cached transmission solver."""
    if len(binary_arrays) == 0:
        return torch.empty(0, dtype=torch.float32)

    solver = _get_frequency_solver(binary_arrays[0].shape)
    scores = []
    for binary_img in binary_arrays:
        try:
            freqs, T_dB = solver.solve(binary_img)
            score = _score_transmission_curve(freqs, T_dB, regions, threshold=-10.0)
        except Exception:
            score = -100.0
        scores.append(score)

    return torch.tensor(scores, dtype=torch.float32).cpu()


# *******************
# Valuer 4 Entry Points
# Defines the pass rule and score rules for the transmission-spectrum task.
# *******************
def valuer4_pass(binary_samples, device):
    """Check whether each sample satisfies the vertical support requirement."""
    connectivity_results = torch.tensor(evaluate_vertical_connectivity(binary_samples, device), dtype=torch.bool)
    pass_list = connectivity_results
    return pass_list


def valuer4_score_1(binary_samples, device):
    """Score each sample for one passband between two target bandgaps."""
    del device
    return _score_binary_samples(_binary_sample_arrays(binary_samples), ONE_PASS_REGIONS)


def valuer4_score_2(binary_samples, device):
    """Score each sample for two separated passbands within the frequency range."""
    del device
    return _score_binary_samples(_binary_sample_arrays(binary_samples), TWO_PASS_REGIONS)


# *******************
# Vertical Connectivity Check
# Requires at least one spanning component and forbids unsupported floating components.
# *******************
def evaluate_vertical_connectivity(binary_samples, device=None):
    """Evaluate whether the design spans vertically without unsupported floating components."""
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

        material_labels = np.unique(labeled_np[labeled_np > 0])
        material_components = {find(lbl) for lbl in material_labels}
        top_labels = {find(lbl) for lbl in np.unique(labeled_np[0, :]) if lbl > 0}
        bottom_labels = {find(lbl) for lbl in np.unique(labeled_np[-1, :]) if lbl > 0}
        supported_components = top_labels | bottom_labels
        spanning_components = top_labels & bottom_labels
        results.append(len(spanning_components) > 0 and material_components.issubset(supported_components))

    return results


# *******************
# Transmission Solver
# Computes the transmission spectrum of a binary layout with a finite-element model.
# *******************
class TransmissionSolver:
    """Finite-element transmission solver used by the frequency-domain valuers."""

    @staticmethod
    def _get_ke(E, nu):
        """Build the elemental stiffness matrix for a solid quadrilateral element."""
        D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        ke = np.zeros((8, 8))
        for xi in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            for et in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
                dN = 0.25 * np.array([
                    [-(1 - et), (1 - et), (1 + et), -(1 + et)],
                    [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
                ])
                J = dN @ np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
                dNxy = np.linalg.solve(J, dN)
                B = np.zeros((3, 8))
                B[0, 0::2] = dNxy[0]
                B[1, 1::2] = dNxy[1]
                B[2, 0::2] = dNxy[1]
                B[2, 1::2] = dNxy[0]
                ke += B.T @ D @ B * np.linalg.det(J)
        return ke

    @staticmethod
    def _get_me(rho):
        """Build the elemental consistent mass matrix for one quadrilateral element."""
        return rho / 36 * np.array([
            [4, 0, 2, 0, 1, 0, 2, 0],
            [0, 4, 0, 2, 0, 1, 0, 2],
            [2, 0, 4, 0, 2, 0, 1, 0],
            [0, 2, 0, 4, 0, 2, 0, 1],
            [1, 0, 2, 0, 4, 0, 2, 0],
            [0, 1, 0, 2, 0, 4, 0, 2],
            [2, 0, 1, 0, 2, 0, 4, 0],
            [0, 2, 0, 1, 0, 2, 0, 4],
        ], dtype=float)

    @staticmethod
    def _assemble(img, mat_solid, mat_void):
        """Assemble the global sparse matrix from elemental material matrices."""
        nely, nelx = img.shape
        iy, ix = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
        iy, ix = iy.ravel(), ix.ravel()
        n1, n2 = iy * (nelx + 1) + ix, iy * (nelx + 1) + ix + 1
        n3, n4 = (iy + 1) * (nelx + 1) + ix + 1, (iy + 1) * (nelx + 1) + ix
        dofs = np.stack(
            [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1],
            axis=1,
        )
        mask = img.ravel().astype(bool)
        vals = np.where(mask[:, None, None], mat_solid[None], mat_void[None])
        I = dofs[:, :, None] * np.ones((1, 1, 8), int)
        J = dofs[:, None, :] * np.ones((1, 8, 1), int)
        n = 2 * (nelx + 1) * (nely + 1)
        return coo_matrix((vals.ravel(), (I.ravel(), J.ravel())), shape=(n, n)).tocsr()

    def __init__(self, E_s=50e6, nu=0.3, rho=1050.0, eta=0.1,
                 h=5e-4, n_modes=50, nelx=256, nely=256,
                 f_min=1e3, f_max=5e3, n_freq=200):
        """Initialize mesh, constraints, modal settings, and frequency sweep."""
        self.eta, self.n_modes = eta, n_modes
        self.freqs = np.linspace(f_min, f_max, n_freq)
        self.om = 2 * np.pi * self.freqs
        self.ke = self._get_ke(E_s, nu)
        self.me = h**2 * self._get_me(rho)
        n_nodes = (nelx + 1) * (nely + 1)
        n_dofs = 2 * n_nodes

        def nd(iy, ix):
            return iy * (nelx + 1) + ix

        tl, tr = nd(0, 0), nd(0, nelx)
        top_inn = np.array([nd(0, ix) for ix in range(1, nelx)])
        left_inn = np.array([nd(iy, 0) for iy in range(1, nely)])
        right_inn = np.array([nd(iy, nelx) for iy in range(1, nely)])
        self.bot_nodes = np.array([nd(nely, ix) for ix in range(nelx + 1)])
        boundary = np.unique(np.concatenate([
            [tl, tr, nd(nely, 0), nd(nely, nelx)],
            top_inn,
            self.bot_nodes[1:-1],
            left_inn,
            right_inn,
        ]))
        interior = np.setdiff1d(np.arange(n_nodes), boundary)

        mdofs = np.array(sorted(
            [d for n in interior for d in (2 * n, 2 * n + 1)]
            + [2 * n + 1 for n in np.concatenate([[tl], top_inn])]
            + [2 * n + 1 for n in left_inn]
        ))
        midx = {d: i for i, d in enumerate(mdofs)}

        r, c, v = [], [], []
        for n in interior:
            for d in (2 * n, 2 * n + 1):
                r.append(d)
                c.append(midx[d])
                v.append(1.0)
        for n in np.concatenate([[tl], top_inn]):
            d = 2 * n + 1
            r.append(d)
            c.append(midx[d])
            v.append(1.0)
        for n in left_inn:
            d = 2 * n + 1
            r.append(d)
            c.append(midx[d])
            v.append(1.0)
        for rn, ln in zip(right_inn, left_inn):
            r.append(2 * rn + 1)
            c.append(midx[2 * ln + 1])
            v.append(1.0)
        r.append(2 * tr + 1)
        c.append(midx[2 * tl + 1])
        v.append(1.0)

        self._T = csr_matrix((v, (r, c)), shape=(n_dofs, len(mdofs)))
        self._up = np.zeros(n_dofs)
        self._up[2 * self.bot_nodes + 1] = 1.0
        self._top_raw = np.array([midx[2 * n + 1] for n in np.concatenate([[tl], top_inn])])

    def solve(self, img):
        """Solve the modal transmission problem and return frequency and T(dB) curves."""
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.asarray(img, dtype=float)

        K = self._assemble(img, self.ke, np.zeros_like(self.ke))
        M = self._assemble(img, self.me, np.zeros_like(self.me))

        K_red = self._T.T @ K @ self._T
        M_red = self._T.T @ M @ self._T

        act = np.where(np.array(K_red.diagonal()) > 0)[0]
        if act.size < 2:
            raise ValueError("Not enough active reduced degrees of freedom for modal analysis.")
        act_map = {old: new for new, old in enumerate(act)}
        K_red = K_red[np.ix_(act, act)]
        M_red = M_red[np.ix_(act, act)]
        K_fp = (self._T.T @ (K @ self._up))[act]
        M_fp = (self._T.T @ (M @ self._up))[act]
        top_idx = np.array([act_map[i] for i in self._top_raw if i in act_map])
        if top_idx.size == 0:
            raise ValueError("No active top degrees of freedom for modal response.")

        k = min(self.n_modes, max(K_red.shape[0] - 1, 1))
        evals, Phi = eigsh(K_red, k=k, M=M_red, sigma=1.0)
        omega_n = np.sqrt(np.abs(evals))

        phi_top = Phi[top_idx].mean(axis=0)
        gK = Phi.T @ K_fp
        gM = Phi.T @ M_fp
        u_st = spsolve(K_red, -K_fp)
        u_top_exact = u_st[top_idx].mean()
        u_top_modal = phi_top @ (-gK / omega_n**2)
        residual = u_top_exact - u_top_modal

        num = -(1 + 1j * self.eta) * gK[:, None] + self.om[None, :]**2 * gM[:, None]
        den = omega_n[:, None]**2 * (1 + 1j * self.eta) - self.om[None, :]**2
        T_dB = 20 * np.log10(np.abs(phi_top @ (num / den) + residual) + 1e-40)
        return self.freqs, T_dB


__all__ = [
    "ONE_PASS_REGIONS",
    "TransmissionSolver",
    "TWO_PASS_REGIONS",
    "evaluate_vertical_connectivity",
    "valuer4_pass",
    "valuer4_score_1",
    "valuer4_score_2",
]
