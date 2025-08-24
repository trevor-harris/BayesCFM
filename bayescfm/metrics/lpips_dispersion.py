
import torch
from typing import Tuple, Optional

@torch.no_grad()
def lpips_dispersion(
    images: torch.Tensor,
    max_pairs: int = 10000,
    backend: str = "lpips",   # "lpips" or "vgg"
    device: Optional[torch.device] = None,
) -> float:
    """Average pairwise perceptual distance over randomly sampled pairs.
    images: [N,3,H,W] in [-1,1] or [0,1]; we internally map to [-1,1] for LPIPS.
    backend='lpips' requires the 'lpips' package. backend='vgg' uses a simple VGG16 feature L2 proxy.
    Returns scalar dispersion.
    """
    N = images.size(0)
    if N < 2:
        return 0.0
    device = device or (images.device if images.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    n_pairs = min(max_pairs, (N * (N - 1)) // 2)
    g = torch.Generator(device=device).manual_seed(0)

    if backend.lower() == "lpips":
        try:
            import lpips
        except Exception as e:
            raise ImportError("backend='lpips' requires the 'lpips' package: pip install lpips") from e
        net = lpips.LPIPS(net='alex').to(device).eval()
        # LPIPS expects [-1,1]
        x = images.to(device)
        if x.min() < -1.01 or x.max() > 1.01:
            x = x.clamp(0,1) * 2 - 1
        # sample pairs
        idx_i = torch.randint(0, N, (n_pairs,), generator=g, device=device)
        idx_j = torch.randint(0, N, (n_pairs,), generator=g, device=device)
        mask = idx_i != idx_j
        idx_i = idx_i[mask]; idx_j = idx_j[mask]
        # compute in mini-batches
        bs = 64
        vals = []
        for s in range(0, idx_i.numel(), bs):
            a = x.index_select(0, idx_i[s:s+bs])
            b = x.index_select(0, idx_j[s:s+bs])
            d = net(a, b)  # [bs,1,1,1]
            vals.append(d.view(-1).detach().cpu())
        import numpy as np
        return float(torch.cat(vals).mean().item())
    else:
        # VGG16 feature L2 proxy
        import torchvision.models as models
        import torch.nn.functional as F
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        x = images.to(device)
        if x.min() < -0.01 or x.max() > 1.01:
            x = (x.clamp(-1,1) + 1.0) * 0.5  # map to [0,1]
        # normalize
        mean = torch.tensor([0.485,0.456,0.406], device=device)[None,:,None,None]
        std  = torch.tensor([0.229,0.224,0.225], device=device)[None,:,None,None]
        x = (x - mean) / std

        def feats(z):
            # take a few layers
            outs = []
            cur = z
            for i,layer in enumerate(vgg):
                cur = layer(cur)
                if i in (3,8,15,22):  # relu1_2, relu2_2, relu3_3, relu4_3 approx
                    outs.append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(cur, (7,7)), 1))
            return torch.cat(outs, dim=1)

        FZ = feats(x)

        # sample pairs
        idx_i = torch.randint(0, N, (n_pairs,), generator=g, device=device)
        idx_j = torch.randint(0, N, (n_pairs,), generator=g, device=device)
        mask = idx_i != idx_j
        idx_i = idx_i[mask]; idx_j = idx_j[mask]
        diffs = FZ.index_select(0, idx_i) - FZ.index_select(0, idx_j)
        d = (diffs.pow(2).sum(dim=1)).sqrt()
        return float(d.mean().item())

