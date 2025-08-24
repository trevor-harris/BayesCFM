
import torch
from typing import Dict, Tuple, Optional
from .core import fid_from_stats
from .prdc import prdc

@torch.no_grad()
def intrafid_per_class(
    feats_real: torch.Tensor, labels_real: torch.Tensor,
    feats_fake: torch.Tensor, labels_fake: torch.Tensor,
    eps: float = 1e-6
) -> Dict[int, float]:
    """Compute FID per class between real and fake features. Returns dict class->FID."""
    assert feats_real.size(0) == labels_real.size(0)
    assert feats_fake.size(0) == labels_fake.size(0)
    classes = torch.unique(labels_real.cpu())
    result = {}
    for c in classes.tolist():
        idx_r = (labels_real == c).nonzero(as_tuple=True)[0]
        idx_f = (labels_fake == c).nonzero(as_tuple=True)[0]
        if idx_r.numel() < 2 or idx_f.numel() < 2:
            continue
        Xr = feats_real.index_select(0, idx_r).to(torch.double)
        Xf = feats_fake.index_select(0, idx_f).to(torch.double)
        mu_r = Xr.mean(dim=0)
        mu_f = Xf.mean(dim=0)
        Sr = ((Xr - mu_r).t() @ (Xr - mu_r)) / (Xr.size(0) - 1)
        Sf = ((Xf - mu_f).t() @ (Xf - mu_f)) / (Xf.size(0) - 1)
        fid = fid_from_stats(mu_r, Sr, mu_f, Sf, eps=eps)
        result[c] = float(fid)
    return result

@torch.no_grad()
def rare_class_mask(labels_real: torch.Tensor, percentile: float = 20.0) -> torch.Tensor:
    """Return a boolean mask (over classes in labels_real) for 'rare' classes at or below given percentile of frequency."""
    classes, counts = labels_real.unique(return_counts=True)
    thresh = torch.quantile(counts.float(), percentile / 100.0)
    rare = classes[counts.float() <= thresh]
    return rare

@torch.no_grad()
def pr_coverage_by_class(
    feats_real: torch.Tensor, labels_real: torch.Tensor,
    feats_fake: torch.Tensor, labels_fake: torch.Tensor,
    k: int = 5, chunk: int = 4096,
    only_classes: Optional[torch.Tensor] = None
) -> Dict[int, Dict[str, float]]:
    """Compute PRDC per class present in both real and fake. Optionally restrict to 'only_classes' (tensor of class ids)."""
    assert feats_real.size(0) == labels_real.size(0)
    assert feats_fake.size(0) == labels_fake.size(0)
    result = {}
    classes = torch.unique(labels_real.cpu())
    if only_classes is not None:
        only = set(only_classes.cpu().tolist())
        classes = torch.tensor([c for c in classes.tolist() if c in only])
    for c in classes.tolist():
        idx_r = (labels_real == c).nonzero(as_tuple=True)[0]
        idx_f = (labels_fake == c).nonzero(as_tuple=True)[0]
        if idx_r.numel() < 3 or idx_f.numel() < 3:
            continue
        Xr = feats_real.index_select(0, idx_r)
        Xf = feats_fake.index_select(0, idx_f)
        result[c] = prdc(Xr, Xf, k=k, chunk=chunk)
    return result

