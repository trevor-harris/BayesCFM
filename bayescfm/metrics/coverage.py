
import torch
from typing import Optional

@torch.no_grad()
def _lpips_or_vgg_distance(a: torch.Tensor, b: torch.Tensor, backend: str = "lpips", device: Optional[torch.device] = None) -> torch.Tensor:
    device = device or (a.device if a.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    a = a.to(device); b = b.to(device)
    if backend.lower() == "lpips":
        try:
            import lpips
        except Exception as e:
            raise ImportError("backend='lpips' requires the 'lpips' package: pip install lpips") from e
        net = lpips.LPIPS(net='alex').to(device).eval()
        if a.min() < -1.01 or a.max() > 1.01:
            a = a.clamp(0,1) * 2 - 1
            b = b.clamp(0,1) * 2 - 1
        return net(a, b).view(-1)  # [B]
    else:
        import torchvision.models as models
        import torch.nn.functional as F
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        def prep(x):
            if x.min() < -0.01 or x.max() > 1.01:
                x = (x.clamp(-1,1) + 1.0) * 0.5
            mean = torch.tensor([0.485,0.456,0.406], device=device)[None,:,None,None]
            std  = torch.tensor([0.229,0.224,0.225], device=device)[None,:,None,None]
            x = (x - mean) / std
            return x
        def feats(z):
            z = prep(z)
            outs = []
            cur = z
            for i,layer in enumerate(vgg):
                cur = layer(cur)
                if i in (3,8,15,22):
                    outs.append(torch.flatten(F.adaptive_avg_pool2d(cur, (7,7)), 1))
            return torch.cat(outs, dim=1)
        Fa = feats(a); Fb = feats(b)
        return torch.norm(Fa - Fb, dim=1)  # [B]

@torch.no_grad()
def perceptual_coverage_k(real_images: torch.Tensor, cand_images: torch.Tensor, threshold: float, backend: str = "lpips", device: Optional[torch.device] = None) -> float:
    """Perceptual coverage: fraction of targets where at least one of K candidates is within 'threshold'.

    real_images: [N,3,H,W]

    cand_images: [N,K,3,H,W] (aligned by target), or [N*K,3,H,W] with K provided via shape.

    Returns scalar in [0,1]."""
    if cand_images.dim() == 4:
        raise ValueError("cand_images must be [N,K,C,H,W]; got [N,C,H,W].")
    N, K = cand_images.size(0), cand_images.size(1)
    device = device or (real_images.device if real_images.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    real = real_images.to(device)

    hits = 0
    bs = 32
    for i in range(N):
        r = real[i].unsqueeze(0).repeat(1,1,1,1)  # [1,C,H,W]
        # chunk over K
        best = torch.tensor(float("inf"), device=device)
        for s in range(0, K, bs):
            c = cand_images[i, s:s+bs].to(device)  # [b,C,H,W]
            r_rep = r.expand(c.size(0), -1, -1, -1)
            d = _lpips_or_vgg_distance(r_rep, c, backend=backend, device=device)  # [b]
            m = d.min()
            if m < best:
                best = m
        hits += int(best.item() <= threshold)
    return float(hits / N)

