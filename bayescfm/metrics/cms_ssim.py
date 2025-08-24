
import torch
from typing import Dict, Optional, Tuple

def _try_ms_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return MS-SSIM(x,y) using pytorch-msssim if available; else a light SSIM fallback.

    x,y: [B,3,H,W] in [-1,1] or [0,1].

    Output: [B] similarity in [0,1]."""
    try:
        from pytorch_msssim import ms_ssim
        # map to [0,1] for ms-ssim default
        def norm(z):
            if z.min() < -0.01 or z.max() > 1.01:
                z = (z.clamp(-1,1) + 1.0) * 0.5
            return z
        return ms_ssim(norm(x), norm(y), data_range=1.0, size_average=False)
    except Exception:
        # Fallback: simple SSIM-like using a Gaussian blur proxy; not multi-scale
        import torch.nn.functional as F
        def norm(z):
            if z.min() < -0.01 or z.max() > 1.01:
                z = (z.clamp(-1,1) + 1.0) * 0.5
            return z
        x1 = norm(x); y1 = norm(y)
        # quick gaussian blur
        def blur(z):
            return F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        mu_x = blur(x1); mu_y = blur(y1)
        sigma_x = blur(x1 * x1) - mu_x * mu_x
        sigma_y = blur(y1 * y1) - mu_y * mu_y
        sigma_xy = blur(x1 * y1) - mu_x * mu_y
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        # clamp and average over spatial and channels
        ssim = ssim_map.clamp(0,1).mean(dim=(1,2,3))
        return ssim

@torch.no_grad()
def cms_ssim_by_class(images: torch.Tensor, labels: torch.Tensor, pairs_per_class: int = 1000, device: Optional[torch.device] = None) -> Dict[int, float]:
    """Class-conditional MS-SSIM: for each class, sample random pairs within class and average MS-SSIM.

    Lower values => more diversity.

    images: [N,3,H,W]; labels: [N]."""
    device = device or (images.device if images.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    X = images.to(device)
    Y = labels.to(device)
    classes = torch.unique(Y)
    out = {}
    g = torch.Generator(device=device).manual_seed(0)
    for c in classes.tolist():
        idx = (Y == c).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < 2:
            continue
        m = min(pairs_per_class, (n*(n-1))//2)
        # sample pairs with replacement for speed
        i = idx[torch.randint(0, n, (m,), generator=g, device=device)]
        j = idx[torch.randint(0, n, (m,), generator=g, device=device)]
        mask = i != j
        i = i[mask]; j = j[mask]
        if i.numel() == 0:
            continue
        bs = 64
        vals = []
        for s in range(0, i.numel(), bs):
            a = X.index_select(0, i[s:s+bs])
            b = X.index_select(0, j[s:s+bs])
            sim = _try_ms_ssim(a, b)  # [b]
            vals.append(sim.detach().cpu())
        out[c] = float(torch.cat(vals).mean().item())
    return out

