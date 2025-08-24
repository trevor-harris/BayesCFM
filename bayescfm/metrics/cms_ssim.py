import torch

def _try_ms_ssim(x, y):
    try:
        from pytorch_msssim import ms_ssim
        def norm(z):
            if z.min() < -0.01 or z.max() > 1.01: z = (z.clamp(-1,1)+1.0)*0.5
            return z
        return ms_ssim(norm(x), norm(y), data_range=1.0, size_average=False)
    except Exception:
        import torch.nn.functional as F
        def norm(z):
            if z.min() < -0.01 or z.max() > 1.01: z = (z.clamp(-1,1)+1.0)*0.5
            return z
        x1 = norm(x); y1 = norm(y)
        def blur(z): return F.avg_pool2d(z,3,1,1)
        mu_x = blur(x1); mu_y = blur(y1)
        sigma_x = blur(x1*x1) - mu_x*mu_x
        sigma_y = blur(y1*y1) - mu_y*mu_y
        sigma_xy = blur(x1*y1) - mu_x*mu_y
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.clamp(0,1).mean(dim=(1,2,3))

@torch.no_grad()
def cms_ssim_by_class(images, labels, pairs_per_class=1000):
    device = images.device
    classes = torch.unique(labels)
    out = {}
    g = torch.Generator(device=device).manual_seed(0)
    for c in classes.tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < 2: continue
        m = min(pairs_per_class, (n*(n-1))//2)
        i = idx[torch.randint(0,n,(m,),generator=g,device=device)]
        j = idx[torch.randint(0,n,(m,),generator=g,device=device)]
        mask = i != j
        i = i[mask]; j = j[mask]
        bs = 64; vals = []
        for s in range(0, i.numel(), bs):
            a = images.index_select(0, i[s:s+bs])
            b = images.index_select(0, j[s:s+bs])
            sim = _try_ms_ssim(a,b)
            vals.append(sim.detach().cpu())
        out[c] = float(torch.cat(vals).mean().item())
    return out
