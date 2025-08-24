
import torch
from typing import Tuple

@torch.no_grad()
def _roc_auc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute ROC AUC from scores via Mann-Whitney U (no sklearn).

    labels: 1 for 'member', 0 for 'non-member'. Higher scores => more likely to be member.

    """
    scores = scores.detach().cpu()
    labels = labels.detach().cpu().to(torch.long)
    # rank the scores (average ranks for ties)
    sorted_scores, indices = torch.sort(scores)
    ranks = torch.empty_like(indices, dtype=torch.float)
    # average ranks for ties
    i = 0
    n = scores.numel()
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0  # average 1-based rank
        ranks[indices[i:j]] = rank
        i = j
    pos = labels.sum().item()
    neg = (labels.numel() - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    sum_pos_ranks = ranks[labels == 1].sum().item()
    U = sum_pos_ranks - pos * (pos + 1) / 2.0
    auc = U / (pos * neg)
    return float(auc)

@torch.no_grad()
def mia_auc_nn_distance(member_feats: torch.Tensor, nonmember_feats: torch.Tensor, reference_feats: torch.Tensor, chunk: int = 4096, invert: bool = True) -> Tuple[float, torch.Tensor]:
    """Membership-Inference Attack AUC using nearest-neighbor distance to a reference set (e.g., generated samples or training set itself).

    For each example z in member/nonmember, compute score = -min_dist(z, reference) if invert=True else +min_dist.

    Returns (AUC, scores) where scores are aligned as [members..., nonmembers...]."""
    def min_dists(A, B):
        N = A.size(0)
        mins = torch.empty(N, device=A.device, dtype=A.dtype)
        for i in range(0, N, chunk):
            a = A[i:i+chunk]
            d = torch.cdist(a, B)
            mins[i:i+chunk], _ = d.min(dim=1)
        return mins
    m = min_dists(member_feats, reference_feats)
    u = min_dists(nonmember_feats, reference_feats)
    scores = torch.cat([m, u], dim=0)
    if invert:
        scores = -scores
    labels = torch.cat([torch.ones_like(m, dtype=torch.long), torch.zeros_like(u, dtype=torch.long)], dim=0)
    auc = _roc_auc_from_scores(scores, labels)
    return auc, scores

