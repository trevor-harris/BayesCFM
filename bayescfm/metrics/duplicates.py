
import torch
from typing import Optional, Tuple

@torch.no_grad()
def duplicate_rate(features: torch.Tensor, k: int = 1, threshold: float = 1e-6, chunk: int = 4096) -> float:
    """Fraction of samples whose nearest neighbor (excluding self) lies within 'threshold' L2 distance.
    features: [N,D] tensor (can be raw pixels flattened or embeddings).
    """
    N = features.size(0)
    dup = 0
    for i in range(0, N, chunk):
        xs = features[i:i+chunk]
        D = torch.cdist(xs, features)           # [b,N]
        # ignore self by making self-distance inf
        b = xs.size(0)
        row = torch.arange(b, device=features.device)
        col = torch.arange(i, i+b, device=features.device)
        D[row, col] = float("inf")
        vals, _ = D.min(dim=1)
        dup += (vals <= threshold).sum().item()
    return float(dup) / float(N)

@torch.no_grad()
def duplicate_rate_vs_ref(features: torch.Tensor, ref: torch.Tensor, threshold: float, chunk: int = 4096) -> float:
    """Fraction of 'features' that have a nearest neighbor in 'ref' within threshold distance."""
    N = features.size(0)
    c = 0
    for i in range(0, N, chunk):
        xs = features[i:i+chunk]
        D = torch.cdist(xs, ref)
        vals, _ = D.min(dim=1)
        c += (vals <= threshold).sum().item()
    return float(c) / float(N)

@torch.no_grad()
def birthday_paradox_test(features: torch.Tensor, subset: int = 1024, threshold: float = 1e-6, trials: int = 10) -> Tuple[float, float]:
    """Monte Carlo birthday test for mode collapse (near-duplicates).
    Samples 'trials' subsets of size 'subset' (or up to N) and computes collision rate among pairs.
    Returns (collision_rate, estimated_support) where support ~ subset^2 / (2 * collisions) if collisions>0 else inf.
    """
    N = features.size(0)
    subset = min(subset, N)
    g = torch.Generator(device=features.device)
    collisions = 0
    total_pairs = 0
    for _ in range(trials):
        idx = torch.randint(0, N, (subset,), generator=g, device=features.device)
        X = features.index_select(0, idx)
        D = torch.cdist(X, X)
        # exclude diagonal
        b = X.size(0)
        D[torch.arange(b, device=features.device), torch.arange(b, device=features.device)] = float("inf")
        # count pairs under threshold (each pair counted twice; divide by 2)
        c = (D <= threshold).sum().item() / 2.0
        collisions += c
        total_pairs += (b * (b - 1)) / 2.0
    collision_rate = float(collisions / max(1.0, total_pairs))
    support = float((subset * subset) / (2.0 * (collisions / max(1, trials))) ) if collisions > 0 else float("inf")
    return collision_rate, support

