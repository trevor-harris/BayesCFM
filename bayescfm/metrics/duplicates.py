import torch

@torch.no_grad()
def duplicate_rate(features, threshold=0.1, k=1):
    D = torch.cdist(features, features)
    D.fill_diagonal_(float('inf'))
    nn = D.topk(k, largest=False).values[:, -1]
    return (nn <= threshold).float().mean().item()

@torch.no_grad()
def duplicate_rate_vs_ref(features, ref, threshold=0.1):
    D = torch.cdist(features, ref)
    m = D.min(dim=1).values
    return (m <= threshold).float().mean().item()

@torch.no_grad()
def birthday_paradox_test(features, subset=1024, threshold=0.1, trials=10):
    import numpy as np
    rng = np.random.default_rng(0)
    n = features.size(0)
    hits = []
    for _ in range(trials):
        idx = torch.tensor(rng.choice(n, min(subset, n), replace=False), device=features.device)
        D = torch.cdist(features.index_select(0, idx), features.index_select(0, idx))
        D.fill_diagonal_(float('inf'))
        hits.append((D.min(1).values <= threshold).float().mean().item())
    return float(sum(hits)/len(hits))
