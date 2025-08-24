import torch

@torch.no_grad()
def _pairwise_distances(x, y, chunk=4096):
    Ds = []
    for i in range(0, x.size(0), chunk):
        Ds.append(torch.cdist(x[i:i+chunk], y))
    return torch.cat(Ds, dim=0)

@torch.no_grad()
def prdc(real_feats, fake_feats, k=5, chunk=4096):
    Dx = _pairwise_distances(real_feats, real_feats, chunk)
    Dy = _pairwise_distances(fake_feats, fake_feats, chunk)
    Dxy = _pairwise_distances(real_feats, fake_feats, chunk)
    r_k = Dx.kthvalue(k+1, dim=1).values  # exclude self
    s_k = Dy.kthvalue(k+1, dim=1).values
    precision = (Dxy.min(1).values <= r_k).float().mean().item()
    recall    = (Dxy.min(0).values <= s_k).float().mean().item()
    density   = ( (Dxy <= r_k.unsqueeze(1)).float().sum(1) / k ).mean().item()
    coverage  = ( Dx.min(1).values <= Dxy.min(0).values ).float().mean().item()
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}
