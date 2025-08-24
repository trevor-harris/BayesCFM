
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple

@torch.no_grad()
def compute_dataset_stats_from_loader(loader: DataLoader, feature_extractor, device=None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    feats_list = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch.get("images", None)
            if x is None:
                continue
        else:
            x = batch
        feats, _ = feature_extractor(x.to(device))
        feats_list.append(feats.cpu())
    feats_all = torch.cat(feats_list, dim=0).to(torch.double)
    mu = feats_all.mean(dim=0)
    xm = feats_all - mu
    sigma = (xm.t() @ xm) / (feats_all.size(0) - 1)
    return mu, sigma, feats_all.size(0)

@torch.no_grad()
def compute_activations_from_images(x: torch.Tensor, feature_extractor) -> torch.Tensor:
    feats, _ = feature_extractor(x)
    return feats

@torch.no_grad()
def _trace_sqrt_product(C1: torch.Tensor, C2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C1 = C1.to(torch.double)
    C2 = C2.to(torch.double)
    I1 = torch.eye(C1.size(0), device=C1.device, dtype=C1.dtype)
    I2 = torch.eye(C2.size(0), device=C2.device, dtype=C2.dtype)
    C1 = C1 + eps * I1
    C2 = C2 + eps * I2
    s1, U1 = torch.linalg.eigh(C1)
    s1 = s1.clamp_min(0)
    C1_half = (U1 * s1.sqrt().unsqueeze(0)) @ U1.t()
    M = C1_half @ C2 @ C1_half
    s, _ = torch.linalg.eigh((M + M.t()) * 0.5)
    s = s.clamp_min(0)
    return s.sqrt().sum()

@torch.no_grad()
def fid_from_stats(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6) -> float:
    dmu = (mu1 - mu2).to(torch.double)
    tr_sqrt = _trace_sqrt_product(sigma1, sigma2, eps=eps)
    fid = (dmu.dot(dmu) + torch.trace(sigma1.to(torch.double) + sigma2.to(torch.double) - 2.0 * tr_sqrt)).item()
    return float(fid)

@torch.no_grad()
def _poly_mmd2_unbiased(X: torch.Tensor, Y: torch.Tensor, degree: int = 3) -> torch.Tensor:
    n, d = X.shape
    m, _ = Y.shape
    c = 1.0 / d
    Kxx = (X @ X.t()) * c + 1.0
    Kyy = (Y @ Y.t()) * c + 1.0
    Kxy = (X @ Y.t()) * c + 1.0
    Kxx.pow_(degree); Kyy.pow_(degree); Kxy.pow_(degree)
    sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    sum_Kxy = Kxy.mean()
    return sum_Kxx + sum_Kyy - 2 * sum_Kxy

@torch.no_grad()
def kid_from_activations(X: torch.Tensor, Y: torch.Tensor, n_subsets: int = 100, subset_size: int = 1000, degree: int = 3):
    device = X.device
    X = X.to(device); Y = Y.to(device)
    rng = torch.Generator(device=device)
    vals = []
    for _ in range(n_subsets):
        idx_x = torch.randint(0, X.size(0), (subset_size,), generator=rng, device=device)
        idx_y = torch.randint(0, Y.size(0), (subset_size,), generator=rng, device=device)
        vals.append(_poly_mmd2_unbiased(X[idx_x], Y[idx_y], degree=degree).item())
    import numpy as np
    return float(np.mean(vals)), float(np.std(vals))

@torch.no_grad()
def inception_score_from_probs(P: torch.Tensor, splits: int = 10, eps: float = 1e-16):
    N = P.size(0)
    split_size = max(1, N // splits)
    scores = []
    for i in range(splits):
        part = P[i * split_size : min(N, (i + 1) * split_size)]
        if part.size(0) == 0:
            continue
        py = part.mean(dim=0, keepdim=True)
        kl = part * (part.add(eps).log() - py.add(eps).log())
        kl = kl.sum(dim=1).mean()
        scores.append(torch.exp(kl).item())
    import numpy as np
    return float(np.mean(scores)), float(np.std(scores))

