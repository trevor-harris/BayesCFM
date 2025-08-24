
import torch
from typing import Dict, Optional

@torch.no_grad()
def _kth_nn_radius(X: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    """Return per-point distance to its k-th nearest neighbor within X (excluding self).
    Uses chunked torch.cdist to save memory. X: [N,D]
    """
    device, dtype = X.device, X.dtype
    N = X.size(0)
    radii = torch.empty(N, device=device, dtype=dtype)
    for i in range(0, N, chunk):
        xs = X[i:i+chunk]  # [B,D]
        # cdist xs to X: [B,N]
        Dmat = torch.cdist(xs, X)  # includes zero self-distances for overlapping rows
        # sort along last dim; take k+1 because of self (0.0)
        vals, _ = torch.topk(Dmat, k=k+1, largest=False, dim=1)
        radii[i:i+chunk] = vals[:, -1]  # k-th NN (excluding self)
    return radii

@torch.no_grad()
def _min_dist_to_set(A: torch.Tensor, B: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """For each row in A, compute min_j ||a - b_j||_2 over B. Returns [A_N]."""
    device, dtype = A.device, A.dtype
    N = A.size(0)
    mins = torch.empty(N, device=device, dtype=dtype)
    for i in range(0, N, chunk):
        a = A[i:i+chunk]
        d = torch.cdist(a, B)  # [b, M]
        mins[i:i+chunk], _ = d.min(dim=1)
    return mins

@torch.no_grad()
def _count_within_radii(A: torch.Tensor, B: torch.Tensor, radii_B: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """For each a in A, count how many b in B satisfy ||a-b|| <= r_b (per-b radius). Returns [A_N] int32."""
    device = A.device
    N = A.size(0)
    counts = torch.zeros(N, device=device, dtype=torch.int32)
    M = B.size(0)
    for i in range(0, N, chunk):
        a = A[i:i+chunk]                               # [b,D]
        d = torch.cdist(a, B)                          # [b,M]
        # Compare to per-b radii
        within = d <= radii_B[None, :]                 # [b,M]
        counts[i:i+chunk] = within.sum(dim=1).to(torch.int32)
    return counts

@torch.no_grad()
def prdc(real_feats: torch.Tensor, fake_feats: torch.Tensor, k: int = 5, chunk: int = 4096) -> Dict[str, float]:
    """Compute Precision, Recall, Density, Coverage (PRDC) from features (N,D).
    Based on Naeem et al. (2019). Torch-only, chunked for memory.
    """
    assert real_feats.ndim == 2 and fake_feats.ndim == 2, "Use [N,D] features"
    device = real_feats.device
    real_feats = real_feats.to(device)
    fake_feats = fake_feats.to(device)

    # Per-set kNN radii
    r_radii = _kth_nn_radius(real_feats, k=k, chunk=chunk)  # [Nr]
    f_radii = _kth_nn_radius(fake_feats, k=k, chunk=chunk)  # [Nf]

    # Precision: fraction of fakes inside real manifold: min_dist(fake, real) <= r_radii (per real)
    # Implement as: count real anchors covering each fake (>=1 => in-manifold)
    cov_counts_f = _count_within_radii(fake_feats, real_feats, r_radii, chunk=max(1024, chunk//2))  # [Nf]
    precision = (cov_counts_f > 0).float().mean().item()

    # Recall: fraction of reals inside fake manifold
    cov_counts_r = _count_within_radii(real_feats, fake_feats, f_radii, chunk=max(1024, chunk//2))  # [Nr]
    recall = (cov_counts_r > 0).float().mean().item()

    # Coverage: fraction of reals whose nearest fake is within its real kNN radius
    min_rf = _min_dist_to_set(real_feats, fake_feats, chunk=chunk)
    coverage = (min_rf <= r_radii).float().mean().item()

    # Density: average number of covering real anchors per fake, normalized by k
    density = (cov_counts_f.float() / float(k)).mean().item()

    return {"precision": float(precision), "recall": float(recall), "density": float(density), "coverage": float(coverage)}

