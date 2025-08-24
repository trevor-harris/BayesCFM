from typing import List, Dict
import torch
from .prdc import prdc

@torch.no_grad()
def prdc_frontier(real_feats: torch.Tensor, fake_feats: torch.Tensor, ks: List[int] | None = None, chunk: int = 4096):
    N = min(real_feats.size(0), fake_feats.size(0))
    if ks is None: ks = [1,2,3,5,8,10,15,20,30,50,75,100]
    ks = [int(k) for k in ks if 1 <= k < N]
    out = []
    for k in ks:
        out.append({"k": k, **prdc(real_feats, fake_feats, k=k, chunk=chunk)})
    return out
